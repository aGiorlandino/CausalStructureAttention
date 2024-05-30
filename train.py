# train.py

import os
import jax
import jax.numpy as jnp
from jax import random
from flax import serialization
import optax
import numpy as np
from utils import cross_entropy_loss, causal_mask, embed
from model import NextTokenPredictor
from data_generation import create_3gram_transition_matrix, generate_sequence


def train(T, S, m1, m2):
    # Define constants and parameters
    num_heads = [m1, m2]  # Number of heads for each layer

    d_0 = S + T # Dimension of the input sequence
    d_1 = (1 + m1) * d_0  
    d_2 = (1 + m2) * d_1  # embedding dimension  

    vocab_size = S
    layer_dims = [d_2, d_2]  

    model = NextTokenPredictor(
        vocab_size=vocab_size,
        layer_dims=layer_dims,
        num_heads=num_heads
    )

    # Directory to save model parameters
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)

    # Lists to store training loss and model parameters
    training_losses = []
    model_params_list = []

    num_epochs = 2048
    num_batches = 16
    batch_size = 1024

    # Define optimizer with cosine decay schedule
    num_train_steps = num_epochs * num_batches
    lr_schedule = optax.cosine_decay_schedule(0.3, num_train_steps)
    optimizer = optax.chain(optax.adam(learning_rate=lr_schedule), optax.clip_by_global_norm(1.0))

    # Initialize model and optimizer state
    rng = random.PRNGKey(0)
    params = model.init(rng, jnp.zeros((1, T, d_2), dtype=jnp.int32))

    optimizer_state = optimizer.init(params)

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        transition_matrix = create_3gram_transition_matrix(S) # 3-gram transition fixed for the epoch

        for batch_idx in range(num_batches):
            sequences = np.array([generate_sequence(transition_matrix, T) for _ in range(batch_size)])

            # Generate target sequences by shifting the input sequences
            target_sequences = np.zeros_like(sequences)
            target_sequences[:, :-1] = sequences[:, 1:]

            # Convert target sequences to one-hot encoding
            sequences_onehot = np.zeros((sequences.shape[0], sequences.shape[1], S), dtype=np.int32)
            target_sequences_onehot = np.zeros((target_sequences.shape[0], target_sequences.shape[1], S), dtype=np.int32)
            for i in range(sequences.shape[0]):
                for j in range(sequences.shape[1]):
                    sequences_onehot[i, j, sequences[i, j]] = 1
                    target_sequences_onehot[i, j, target_sequences[i, j]] = 1

            # Embed input sequences
            embedded_sequences = embed(sequences, S, d_2)

            # Convert embedded sequences and targets to JAX arrays
            batch_sequences = jnp.array(embedded_sequences)
            batch_targets = jnp.array(target_sequences_onehot)

            # Compute gradients and loss
            def loss_fn(params):
                logits = model.apply(params, batch_sequences)
                loss = cross_entropy_loss(logits, batch_targets)
                return loss, logits

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, _), grads = grad_fn(params)
            epoch_loss += loss

            # Update parameters
            updates, optimizer_state = optimizer.update(grads, optimizer_state)
            params = optax.apply_updates(params, updates)

        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / num_batches
        training_losses.append(avg_epoch_loss)

        # Save model parameters
        if (epoch + 1) % 250 == 0:
            model_params_list.append(params)
            model_path = os.path.join(save_dir, f"model_epoch_{epoch+1}_num_head_{num_heads[0]}.params")

            with open(model_path, "wb") as f:
                f.write(serialization.to_bytes(params))

        # Print epoch loss
        print(f"Epoch {epoch+1}, Loss: {avg_epoch_loss}")

    # Save training losses to file
    losses_path = os.path.join(save_dir, "training_losses_num_head_{num_heads[0]}.npy")
    np.save(losses_path, np.array(training_losses))

