from src.data_loader import load_data
from src.model import build_model, train_model, evaluate_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def main():
    print("Loading data...")
    x_train, y_train, x_test, y_test = load_data()

    print("Building model...")
    model = build_model()

    print("Training model...")
    model = train_model(model, x_train, y_train)

    print("Evaluating model...")
    accuracy = evaluate_model(model, x_test, y_test)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    print("Generating classification report...")
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
