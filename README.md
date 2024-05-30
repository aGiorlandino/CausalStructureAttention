# CausalStructureAttention
 
# JAX Implementation of How Transformers Learn Causal Structure with Gradient Descent 

This repository contains a JAX implementation of the experiments described in the paper "[How Transformers Learn Causal Structure with Gradient Descent](https://arxiv.org/abs/2402.14735)" by [Eshaan Nichani, Alex Damian, Jason D. Lee].

**Codes under construction**

## Overview

The paper investigates how transformers learn causal structure using gradient descent. It explores the mechanisms by which transformers, a type of neural network architecture, can infer causal relationships from observational data generated according to a causally structured process.
This code focuses on the 3-gram structure.

## Requirements

- Python 3.x
- JAX
- Flax
- NumPy
- [List any other dependencies] ...(Still working on this)

## Installation

Clone the repository:

```bash
https://github.com/aGiorlandino/CausalStructureAttention
cd CausalStructureAttention
```

Install the required dependencies: #still have to finish this

```bash
pip install -r requirements.txt
```

## Usage

[Explain how to use your code. Include instructions on how to run the experiments, train models, and reproduce the results of the paper.]

```bash
python main.py
```

## Structure

- `main.py`: Main script to run the experiments.
- `models.py`: Implementation of the neural network models used in the paper.
- `train.py`: Implementation of cross-entropy minimization
- `data_generation.py`: generate 3-grams
- `utils.py`: Utility functions.
[- `experiments/`: Directory containing scripts for specific experiments.]:
[- `results/`: Directory to store experimental results.]:

<!---

## Contributing

[Explain how others can contribute to your project, such as filing issues or submitting pull requests.]

## License

[Include the license information of your project.]

## Acknowledgments

[Optional: Acknowledge any contributors, libraries, or resources that you used in your project.]
-->
