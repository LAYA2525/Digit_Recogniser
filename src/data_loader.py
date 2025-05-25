import numpy as np
from sklearn.datasets import fetch_openml

def load_data(flatten=True):
    # Load MNIST from sklearn (use different approach)
    try:
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    except:
        # Alternative: use the default MNIST dataset
        mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')
    
    X, y = mnist.data, mnist.target.astype(int)
    
    # Split into train/test (first 60000 for train, rest for test)
    x_train, x_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    
    # Normalize
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    if not flatten:
        x_train = x_train.reshape(-1, 28, 28)
        x_test = x_test.reshape(-1, 28, 28)
    
    return x_train, y_train, x_test, y_test
