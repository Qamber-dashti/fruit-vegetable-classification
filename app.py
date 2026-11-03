
import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification

# Hugging Face model
MODEL_ID = "Qamber-Dashti/fruit-vegetable-classification"

st.set_page_config(
    page_title="Fruit & Vegetable Classifier",
    page_icon="🍎",
    layout="centered"
)

@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = ViTForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model

st.title("🍎 Fruit & Vegetable Classification")
st.write("Upload an image to identify the fruit or vegetable.")

try:
    processor, model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=350)

    if st.button("Classify Image"):
        with st.spinner("Analyzing image..."):
            inputs = processor(
                images=image,
                return_tensors="pt"
            )

            with torch.inference_mode():
                outputs = model(**inputs)
                probabilities = torch.softmax(
                    outputs.logits, dim=-1
                )[0]

            predicted_id = probabilities.argmax().item()

            id2label = model.config.id2label

            predicted_label = id2label.get(
                predicted_id,
                id2label.get(str(predicted_id), f"Class {predicted_id}")
            )

            confidence = probabilities[predicted_id].item() * 100

            st.success(f"Prediction: {predicted_label}")
            st.metric("Confidence", f"{confidence:.2f}%")

            st.subheader("Top 5 Predictions")

            top_values, top_indices = torch.topk(
                probabilities,
                min(5, len(probabilities))
            )

            for value, index in zip(top_values, top_indices):
                label = id2label.get(
                    index.item(),
                    id2label.get(str(index.item()), f"Class {index.item()}")
                )
                st.write(f"{label}: {value.item()*100:.2f}%")
                st.progress(value.item())
