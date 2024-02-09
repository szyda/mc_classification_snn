# Multi-Criteria Classification using Single-Layer Neural Network

## Overview
This project focuses on implementing a single-layer neural network for the multicriteria classification of animals based on various features represented by numerical values. The core objective is to train the neural network to classify animals into three categories: mammals, birds, and fish, utilizing a set of input features and a teacher matrix for expected outcomes.

## Key Features
- **Neural Network Model:** A single-layer neural network model with a unipolar sigmoid activation function.
- **Learning and training:** The model uses a random set of weights initially, which are adjusted over time through a learning process involving backpropagation and the gradient descent method.
- **Input and target data:** The input matrix (P) represents different animal features, while the target matrix (T) represents the expected classification outcomes.
- **Verification:** Includes a verification function to test the model's performance on new data sets.

## Implementation
The implementation involves:
- Defining the neural network structure and its activation function.
- Randomly generating initial weights.
- Iteratively adjusting weights based on the learning rate, epochs, and error correction through the training process.
- Verifying the trained model's accuracy with new input data.

## Dependencies:
**NumPy**
```
pip install numpy
```

## How to Run:
1. Clone the repository.
2. Install dependencies.
3. Run the script (python main.py) to see the perceptron in action.

### Note
This project was made for an Introduction to Artificial Intelligence course at school, showcasing practical application of AI principles in a controlled educational environment.
