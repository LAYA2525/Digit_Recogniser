from sklearn.neural_network import MLPClassifier
import joblib
import numpy as np

def build_model(input_shape=(784,), num_classes=10):
    """
    Build a neural network model using scikit-learn's MLPClassifier
    """
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),  # Two hidden layers with 128 and 64 neurons
        activation='relu',
        solver='adam',
        max_iter=300,
        random_state=42,
        verbose=True
    )
    return model

def train_model(model, x_train, y_train):
    """
    Train the model
    """
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x_test, y_test):
    """
    Evaluate the model and return accuracy
    """
    accuracy = model.score(x_test, y_test)
    return accuracy

def save_model(model, filepath):
    """
    Save the trained model
    """
    joblib.dump(model, filepath)

def load_model(filepath):
    """
    Load a saved model
    """
    return joblib.load(filepath)
