# data_generation.py

import numpy as np

    
def create_3gram_transition_matrix(S):
    """
    Create a transition matrix for a 3-gram model by sampling from a Dirichlet prior.

    Returns:
    dict: A dictionary where keys are 2-grams (tuples of integers) and values are the probability distributions
          over the next words (integers from 0 to S-1).
    """
    # Create a list of all possible 2-grams from the vocabulary
    two_grams = [(i, j) for i in range(S) for j in range(S)]

    transition_matrix = {}     # Initialize the transition matrix as a dictionary
    alpha = 1.0     # Dirichlet parameter alpha
 
    for two_gram in two_grams:
        next_word_probs = np.random.dirichlet([alpha] * S)         # Sample a probability distribution over the next words
        transition_matrix[two_gram] = next_word_probs
    return transition_matrix


def generate_sequence(transition_matrix, T):
    """
    Generate a sequence of length T given a transition matrix.
    
    Parameters:
    transition_matrix (dict): The transition matrix where keys are 2-grams (tuples of integers)
                              and values are the probability distributions over the next words.
    T (int): Length of the sequence to generate.
    
    Returns:
    list: A list of integers representing the generated sequence.
    """
    two_grams = list(transition_matrix.keys())     # Extract the list of 2-grams from the transition matrix
    
    # Randomly choose an initial 2-gram
    current_2gram = two_grams[np.random.choice(len(two_grams))]
    
    sequence = list(current_2gram)     # Initialize the sequence with the chosen 2-gram
    
    # Generate the sequence
    for _ in range(T - 2):
        next_word_probs = transition_matrix[current_2gram]      # Get the next word probability distribution for the current 2-gram
        next_word = np.random.choice(range(len(next_word_probs)), p=next_word_probs)         # Sample the next word 
        sequence.append(next_word)
        # Update the current 2-gram
        current_2gram = (current_2gram[1], next_word)
    return sequence