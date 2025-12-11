# Wise CNN Model for Stock Prediction

The 'Wise CNN' page in the WiseTrader application implements a Convolutional Neural Network (CNN) for stock price movement prediction. This model is designed to analyze sequential patterns in historical stock data to forecast future price trends (up or down).

## How to Use

1.  **Navigate to 'Wise CNN'**: Select 'Wise CNN' from the sidebar navigation within the WiseTrader application.
2.  **Enter Stock Ticker**: Input the ticker symbol of the desired stock (e.g., `NV20.NS` for Nifty 20, or `AAPL` for Apple). The system will fetch 1 year of historical data at a 1-hour interval for the specified stock.
3.  **Model Training**: The application will automatically preprocess the fetched data, train the CNN model, and evaluate its performance. This process might take some time depending on data size and computational resources.
4.  **View Results**: Once training is complete, the page will display the model's accuracy, a plot of training and validation accuracy over epochs, and a plot of loss.

## Model Details

### Data Preprocessing

Prior to model training, the stock data (`Open`, `High`, `Low`, `Close`, `Volume`) is loaded and scaled using `MinMaxScaler` to normalize the input features. Sequences of `sequence_length = 10` are created for the input `X`, and the target `y` is a binary classification (1 if closing price of the next sequence is higher, 0 otherwise).

### Model Architecture

The CNN model is built using Keras/TensorFlow `Sequential` API and consists of the following layers:

*   **Conv1D Layer**: A 1D convolutional layer with 64 filters, a kernel size of 3, `relu` activation, and `input_shape=(10, 5)` (sequence length, number of features).
*   **MaxPooling1D Layer**: Reduces the dimensionality of the feature maps.
*   **Flatten Layer**: Flattens the output for dense layers.
*   **Dense Layers**: Two fully connected layers with `relu` and `sigmoid` activations respectively, for classification.

### Training and Evaluation

The model is compiled with the `adam` optimizer and `binary_crossentropy` loss function. It is trained for 10 epochs with a batch size of 64 and a validation split of 0.2. The training progress and final accuracy are displayed.

## Interpretation of Results

*   **Accuracy**: Indicates the proportion of correct predictions the model made on unseen test data.
*   **Accuracy/Loss Plots**: These graphs visualize the model's learning process over epochs, showing how well the model generalizes and helps identify overfitting (`val_accuracy` diverging from `train_accuracy`).

## Example Usage

To use the Wise CNN model, simply enter a preferred stock ticker in the input box, for instance, `GOOGL`. The system will then proceed to fetch data, train the model, and present the results as described above.

```python
# Example of how to interact in Streamlit (Conceptual)
import streamlit as st

# ... (Code snippet for text input and button press in Streamlit)
# (This code is handled internally by the Streamlit app)
```