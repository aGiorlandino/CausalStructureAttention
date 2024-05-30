# main.py

from train import train


def main():
    # Define constants
    T = 20  # Sequence length
    S = 10  # Cardinality of the alphabet
    m1 = 2  # Heads in the first layer
    m2 = 1  # Heads in the second layer

    # Train the model
    train(T, S, m1, m2)

if __name__ == "__main__":
    main()
