"""This module trains three neural network models on 
the dataset recordings and saves the X and y features."""

import os
import joblib
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (
    Dense,
    Conv1D,
    Flatten,
    Dropout,
    Activation,
    MaxPooling1D,
    BatchNormalization,
    LSTM,
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from config import features_PATH  
from config import images_PATH  
from config import models_PATH  
from config import recordings_PATH  


#NEW

def mlp_classifier(X, y):
    """
    Function to train and evaluate a Multilayer Perceptron (MLP) classifier. 
    The data is first split, the MLP model is initialized with a single hidden layer,
    trained, and then evaluated for accuracy. A classification report is generated
    and saved to CSV for detailed analysis.
    """
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit the MLP model
    model = MLPClassifier(hidden_layer_sizes=(100,), solver='adam', alpha=0.001, 
                          shuffle=True, verbose=True, momentum=0.8)
    model.fit(X_train, y_train)

    # Predict and evaluate the model's accuracy
    predictions = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    print(f"MLP Accuracy: {accuracy * 100:.2f}%")

    # Generate and export classification report
    report = classification_report(y_test, predictions, output_dict=True)
    report_df = pd.DataFrame(report).T
    report_df.to_csv(os.path.join(features_PATH, "mlp_classification_report.csv"))
    print(report_df)

def lstm_model(X, y):
    """
    Function to train and evaluate an LSTM model. Data is reshaped to fit the LSTM's
    input requirements, the model is built with two LSTM layers and dense layers,
    then compiled and trained. Post-training, the model's accuracy is evaluated.
    """
    # Prepare the data for LSTM network
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    # Build, compile, and train the LSTM model
    model = Sequential()
    model.add(LSTM(64, input_shape=(40, 1), return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(8, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, to_categorical(y_train, num_classes=8), epochs=100, batch_size=32)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, to_categorical(y_test, num_classes=8), verbose=0)
    print(f"LSTM Test Accuracy: {accuracy * 100:.2f}%")

def cnn_model(X, y):
    """
    Train a Convolutional Neural Network (CNN) on feature set X and target y.
    The function first splits the dataset into training and testing sets, 
    then reshapes the training set to fit the CNN requirements.
    It defines a CNN architecture with three convolutional layers followed by batch normalization and flattening before
    the final classification layer. The model uses ReLU activation and categorical crossentropy as the loss function.
    After compiling the model, it is trained and validated on the test data.
    """

    # Split the dataset into training and testing sets for model validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape the input features to add an additional dimension for the CNN
    x_traincnn = np.expand_dims(X_train, axis=2)
    x_testcnn = np.expand_dims(X_test, axis=2)

    # Define the CNN model architecture
    model = Sequential()
    # Add convolutional layers with 'relu' activation
    model.add(Conv1D(16, 5, padding="same", input_shape=(40, 1), activation="relu"))
    model.add(Conv1D(8, 5, padding="same", activation="relu"))
    # Optional dropout and max pooling layers are commented out; could be used to reduce overfitting
    # Add another convolutional layer followed by batch normalization for training stability
    model.add(Conv1D(8, 5, padding="same", activation="relu"))
    model.add(BatchNormalization())
    # Flatten the output of the conv layers to connect to dense layers
    model.add(Flatten())
    # Output layer with softmax activation for multi-class classification
    model.add(Dense(8, activation="softmax"))

    # Compile the model with adam optimizer and accuracy metrics
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Convert labels to categorical format for the softmax output
    y_train = to_categorical(y_train, num_classes=8)
    y_test = to_categorical(y_test, num_classes=8)

    # Train the CNN and store its history for plotting
    cnn_history = model.fit(x_traincnn, y_train, batch_size=50, epochs=100, validation_data=(x_testcnn, y_test))

    # The model training history can be used to plot accuracy and loss over epochs



    # plot_model(
    #     model,
    #     to_file="speech_emotion_recognition/images/cnn_model_summary.png",
    #     show_shapes=True,
    #     show_layer_names=True,
    # )

    # Plot model loss
    plt.plot(cnn_history.history["loss"])
    plt.plot(cnn_history.history["val_loss"])
    plt.title("CNN model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"])
    cnn_loss_path = os.path.join(images_PATH, "cnn_loss2.png")
    plt.savefig(cnn_loss_path)
    plt.close()

    # Plot model accuracy
    plt.plot(cnn_history.history["accuracy"])
    plt.plot(cnn_history.history["val_accuracy"])
    plt.title("CNN model accuracy")
    plt.ylabel("acc")
    plt.xlabel("epoch")
    plt.legend(["train", "test"])
    cnn_accuracy_path = os.path.join(images_PATH, "cnn_accuracy2.png")
    plt.savefig(cnn_accuracy_path)

    # Evaluate the model
    y_test_int = np.argmax(y_test, axis=-1)

    # If your model's predictions are probabilities, convert them to class indices
    cnn_pred = model.predict(x_testcnn)
    cnn_pred_classes = np.argmax(cnn_pred, axis=-1)

    # Now you can use y_test_int and cnn_pred_classes with confusion_matrix
    matrix = confusion_matrix(y_test_int, cnn_pred_classes)
    print(matrix)

    plt.figure(figsize=(12, 10))
    emotions = [
        "neutral",
        "calm",
        "happy",
        "sad",
        "angry",
        "fearful",
        "disgusted",
        "surprised",
    ]
    cm = pd.DataFrame(matrix)
    ax = sns.heatmap(
        matrix,
        linecolor="white",
        cmap="crest",
        linewidth=1,
        annot=True,
        fmt="",
        xticklabels=emotions,
        yticklabels=emotions,
    )
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title("CNN Model Confusion Matrix", size=20)
    plt.xlabel("predicted emotion", size=14)
    plt.ylabel("actual emotion", size=14)
    cnn_confusion_matrix_path = os.path.join(images_PATH, "CNN_confusionmatrix.png")
    plt.savefig(cnn_confusion_matrix_path)
    plt.show()

    # predictions_array = np.array([cnn_pred, y_test])
    # predictions_df = pd.DataFrame(data=predictions_array)  # .flatten())
    # predictions_df = predictions_df.T
    # predictions_df = predictions_df.rename(columns={0: "cnn_pred", 1: "y_test"})

    cnn_pred_probabilities = model.predict(x_testcnn)
    cnn_pred_classes = np.argmax(cnn_pred_probabilities, axis=-1)
    clas_report = pd.DataFrame(
        classification_report(y_test_int, cnn_pred_classes, output_dict=True)
    ).transpose()
    cnn_clas_report_path = os.path.join(features_PATH, "cnn_clas_report.csv")
    clas_report.to_csv(cnn_clas_report_path)
    print(classification_report(y_test_int, cnn_pred_classes))

    if not os.path.isdir(models_PATH):
        os.makedirs(models_PATH)

    model_path = os.path.join(models_PATH, "cnn_model.h5")
    model.save(model_path)
    print("Saved trained model at %s " % model_path)


if __name__ == "__main__":
    print("Training started")
    X = joblib.load(features_PATH + '\\X_MFCC.joblib')
    y = joblib.load(features_PATH + '\\y_MFCC.joblib')
 #   X = joblib.load(features_PATH + '\\X.joblib')
 #   y = joblib.load(features_PATH + '\\y.joblib')
    cnn_model(X=X, y=y)
#    lstm_model(X=X, y=y)
    print("Model finished.")
