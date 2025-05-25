import streamlit as st
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
import joblib

# --- Streamlit page config MUST be first ----------------------------------- #
st.set_page_config(page_title="Digit Classifier", page_icon="✍️", layout="centered")

MODEL_PATH = "mnist_model.pkl"

# --- Load or train model ---------------------------------------------------- #
@st.cache_resource  # one-time, survives reruns
def get_or_train_model():
    if Path(MODEL_PATH).exists():
        return joblib.load(MODEL_PATH)

    # Load data and train model
    try:
        mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')
    except:
        return None
    
    X, y = mnist.data, mnist.target.astype(int)
    x_train = X[:60000]  # Use first 60k samples for training
    y_train = y[:60000]
    
    x_train = x_train.astype("float32") / 255.0

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=100,
        random_state=42,
        verbose=False
    )
    
    model.fit(x_train, y_train)
    joblib.dump(model, MODEL_PATH)
    return model

# Load model
model = get_or_train_model()

# --- Streamlit UI ----------------------------------------------------------- #
st.title("✍️ Hand-drawn Digit Classifier")

if model is None:
    st.error("Failed to load MNIST dataset. Please check your internet connection.")
    st.stop()

# Show training message if model is being trained for first time
if not Path(MODEL_PATH).exists():
    st.info("Training model for the first time... This may take a few minutes.")

st.write("Draw a **single digit (0-9)** in the canvas, then click **Predict**.")

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=12,
    stroke_color="black",
    background_color="white",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

col1, col2 = st.columns(2)
with col1:
    clear = st.button("Clear", type="secondary")
with col2:
    predict = st.button("Predict", type="primary")

if clear and canvas_result.image_data is not None:
    canvas_result.image_data[:] = 255  # white out
    st.rerun()

if predict and model is not None:
    if canvas_result.image_data is None:
        st.warning("Please draw a digit first.")
    else:
        # Pre-process drawing ➜ (1, 784) vector
        img = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")
        img = ImageOps.invert(img)                # black digit on white bg
        img = img.resize((28, 28))                # MNIST size
        img_array = np.array(img).astype("float32") / 255.0
        img_array = img_array.reshape(1, 784)

        pred = model.predict(img_array)[0]
        st.success(f"**Predicted digit → {pred}**")
        st.image(img.resize((140, 140)), caption="28×28 preview", channels="L")

st.caption("First run? The model trains automatically and is cached in *mnist_model.pkl*.")
