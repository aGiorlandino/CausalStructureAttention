# utils.py

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

# Define utility functions for attention mechanism
def causal_mask(size):
    mask = np.tril(np.ones((size, size), dtype=np.bool_), k=0)
    return jnp.array(mask)

def cross_entropy_loss(logits, labels):
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.sum(labels * log_probs) / labels.shape[0]

def embed(sequences, S, d_2):
    embedded_sequences = np.zeros((sequences.shape[0], sequences.shape[1], d_2), dtype=np.int32)
    for i in range(sequences.shape[0]):
        for j in range(sequences.shape[1]):
            embedded_sequences[i, j, sequences[i, j]] = 1
            embedded_sequences[i, j, S + j] = 1
    return embedded_sequences