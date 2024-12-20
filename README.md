# Neural Network from Scratch

## Overview
This project implements a neural network from scratch. The network itself was built using numpy and is designed to be as modular and scalable as possible given the constraints. I took huge inspiration from the TensorFlow library. My primarmy goal in developing this project was to gain a deep understanding of the inner workings of neural networks by building one from the ground up.
## Key Highlights
- **Custom Implementation**:
  - Forward propagation, backpropagation, and gradient descent coded manually.
  - Activation functions include ReLU, Sigmoid, and Softmax.
- **Results**:
  - Achieved **97% accuracy** on the MNIST test set.
  - Successfully classified the **Iris dataset** (multiclass classification).
  - Solved the **XOR problem** using a two-layer network.
## Tools and Technologies
- **Programming Language**: Python
- **Libraries**: NumPy (no pre-built ML frameworks used)
- **Datasets**:
  - **MNIST**: Handwritten digit classification.
  - **Iris**: Multiclass flower classification.
  - **XOR**: Demonstration of non-linear separability in neural networks.
## Installation and Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/neural-network-from-scratch.git
2. Navigate to the project directory:
   ```bash
   cd neural-network-from-scratch
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
## How to Use
### Option 1: Train the Model Yourself
1. Ensure you have the MNIST dataset in the `data` folder.
2. Run the training script:
   ```bash
   python train.py
3. The model will train on the MNIST dataset, and weights will be saved in the models folder.
### Option 2: Use the Pretrained Model
1. Skip the training step and use the pre-trained model. You dont have to anything for this step.
### Test model
1. This step is the same regardless of if you trained the model or if you're using the pre trained version
   ```bash
   python accuracy.py
2. This step will load the weights and display the model's accuracy on the MNIST test set.
### Display the Dataset
1. To view examples of numbers from the MNIST dataset, run the following:
   ```bash
   python display_numbers.py
2. This step will load the weights and display the model's accuracy on the MNIST test set.
### Test with a Custom Image
1. To see how the model performs on an image not in the dataset or training set:
   ```bash
   python imageclassification.py
2. You can see the original image in the handwritten_digits folder.
## Learning objectives
- Built a complete neural network from scratch, coding core ML algorithms manually.
- Demonstrated proficiency in training and evaluating models on real-world datasets.
- Applied optimization techniques to improve performance.

