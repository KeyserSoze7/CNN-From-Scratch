
# Convolutional Neural Network (CNN) from Scratch

This project implements a Convolutional Neural Network (CNN) from scratch using Python and NumPy. The aim is to understand the inner workings of CNN layers and functions, including forward and backward passes, and build a modular structure for easy extensibility.


## **Overview**

This project demonstrates a fully functional CNN with:
- **Convolutional Layer**: Extracts features using filters.
- **MaxPooling Layer**: Reduces spatial dimensions while retaining significant features.
- **Fully Connected Layer**: Maps extracted features to class scores.
- **Activation Functions**:
  - **ReLU**: Introduces non-linearity.
  - **Softmax**: Outputs class probabilities.

---

## **Features**

- Modular structure: Each layer is implemented in a separate file for better organization and reusability.
- End-to-end learning:
  - Forward pass for predictions.
  - Backward pass for parameter updates.
- Minimal external dependencies: Only Python and NumPy are used.

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/KeyserSoze7/CNN-From-Scratch.git
   cd CNN-From-Scratch
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(As We’re only using Python and NumPy, this step may not be necessary.)*

---

## **Usage**

### **Training the CNN**
Run the `cnn.py` file to train the network:
```bash
python cnn.py
```

- Modify hyperparameters such as learning rate, batch size, and number of epochs in the `main.py` file.
- Customize the dataset loading function in `utils/helper.py`.

---

## **Explanation of Key Components**

1. **Convolutional Layer (`conv.py`)**:
   - Implements the forward and backward pass for extracting spatial features.
   - Supports adjustable filter size, stride, and padding.

2. **MaxPooling Layer (`maxpool.py`)**:
   - Reduces spatial dimensions to enhance computational efficiency.
   - Retains only the most significant features.

3. **Fully Connected Layer (`fullyconnected.py`)**:
   - Flattens feature maps and maps them to output class scores.

4. **Activation Functions (`activations/`)**:
   - **ReLU (`relu.py`)**: Applies non-linearity.
   - **Softmax (`softmax.py`)**: Converts logits into probabilities for classification.

---

## **Dataset**
The project is designed to work with grayscale or RGB image datasets. To use your dataset:
- Replace the data preprocessing functions in `utils/data_loader.py`.

---

## **Next Steps**
- Add additional layers (e.g., dropout, batch normalization).
- Test with larger datasets.
- Optimize for performance using external libraries like TensorFlow or PyTorch after validating the functionality.

---

## **Contributing**
Contributions are welcome! Please fork the repository and submit a pull request with your improvements or suggestions.

---

## **License**
This project is licensed under the MIT License. See `LICENSE` for more details.
