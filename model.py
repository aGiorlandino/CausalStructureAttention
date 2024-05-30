#model.py
import jax.numpy as jnp
from flax import linen as nn
from utils import causal_mask


class MultiHeadSelfAttention(nn.Module):
    embed_dim: int
    num_heads: int

    def setup(self):
        self.head_dim = self.embed_dim 
        self.qkv = nn.Dense(features=self.embed_dim * 3 * self.num_heads, use_bias=False)
        self.out = nn.Dense(features=self.embed_dim)

    def __call__(self, x, mask=None):
        batch_size, seq_length,_ = x.shape
        qkv = self.qkv(x) #qkb are not the params, theyre W_Q, W_K, W_V applied to the batch !
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3, self.head_dim)
        qkv = qkv.transpose((2, 0, 1, 3, 4))  # (num_heads, batch_size, seq_length, 3, head_dim)
        q, k, v = qkv[:, :, :, 0, :], qkv[:, :, :, 1, :], qkv[:, :, :, 2, :]
        attn_weights = jnp.einsum('hbqd,hbkd->hbqk', q, k) / jnp.sqrt(self.head_dim) # einstein summation
        #then check if we want to normalize or not / jnp.sqrt(self.head_dim)

        if mask is not None:
            attn_weights = jnp.where(mask[None, None, :, :], attn_weights, -1e10)

        attn_weights = jax.nn.softmax(attn_weights, axis=-1) #axis=-1 is the last axis i.e. 'row-wise'
        attn_output = jnp.einsum('hbqk,hbvd->hbqd', attn_weights, v) # (num_heads, batch_size, seq_length, head_dim)
        attn_output = attn_output.transpose((1, 2, 0, 3))  # (batch_size, seq_length, num_heads, head_dim)
        attn_output = attn_output.reshape(batch_size, seq_length, self.num_heads * self.head_dim)
        return self.out(attn_output)


class TransformerDecoderLayer(nn.Module):
    embed_dim: int
    num_heads: int

    def setup(self):
        self.self_attn = MultiHeadSelfAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)
        self.ln = nn.LayerNorm()

    def __call__(self, x, mask=None):
        attn_output = self.self_attn(x, mask=mask)
        x = x + attn_output
        x = self.ln(x)
        return x

class TransformerDecoder(nn.Module):
    #vocab_size: int
    layer_dims: list
    num_heads: list

    def setup(self):
        #self.embedding = nn.Embed(num_embeddings=self.vocab_size, features=self.layer_dims[0]) # no embedding!
        self.layers = [TransformerDecoderLayer(embed_dim=layer_dim, num_heads=num_heads) 
                       for layer_dim, num_heads in zip(self.layer_dims, self.num_heads)]
        self.ln = nn.LayerNorm()

    def __call__(self, x, mask=None):
        #x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.ln(x)
        return x

class NextTokenPredictor(nn.Module):
    vocab_size: int
    layer_dims: list
    num_heads: list

    def setup(self):
        self.decoder = TransformerDecoder(
            #vocab_size=self.vocab_size,
            layer_dims=self.layer_dims,
            num_heads=self.num_heads
        )
        self.out = nn.Dense(features=self.vocab_size)

    def __call__(self, x):
        seq_length = x.shape[1]
        mask = causal_mask(seq_length)
        decoder_output = self.decoder(x, mask=mask)
        logits = self.out(decoder_output)
        return logits
