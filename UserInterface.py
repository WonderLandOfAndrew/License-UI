import streamlit as st
from PIL import Image
import numpy as np
import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from keras import mixed_precision
import requests
from io import BytesIO

#mixed_precision.set_global_policy('mixed_float16')

# --- Paths ---
MODELS = {
    "DenseNet": "densenet_fishingnet_model.h5",
    "Inception": "inception_fishingnet_model.h5"
}
LABELS_PATH = "class_labels.txt"
TEST_DIR = "C:/Anul 4/Sem I/ML/MyDataset/test"
IMAGE_SIZE = (224, 224)

# --- Load class labels ---
# @st.cache_data
def load_class_names():
    with open(LABELS_PATH, "r") as f:
        return [line.strip() for line in f.readlines()]

# --- Load models without cache to avoid mixed precision errors ---
def load_all_models():
    loaded = {}
    for name, path in MODELS.items():
        try:
            tf.keras.backend.clear_session()
            loaded[name] = tf.keras.models.load_model(path, compile=False)
        except Exception as e:
            st.error(f"Failed to load model '{name}': {e}")
    return loaded

# --- Prediction logic ---
def predict_from_image(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image = np.array(image).astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    preds = model.predict(image)
    return np.argmax(preds), np.max(preds) * 100


# --- Prediction from URL ---
def predict_from_url(model, url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        img_array = np.expand_dims(np.array(img).astype('float32') / 255.0, axis=0)
        preds = model.predict(img_array)
        return np.argmax(preds), np.max(preds) * 100, img
    except Exception as e:
        st.error(f"Could not process image from URL: {e}")
        return None, None, None

# --- UI Layout ---
st.set_page_config(layout="wide", page_title="Fishing Net Analyser")
st.markdown("""
    <style>
        .main-title {
            font-size: 2.5em;
            font-weight: bold;
            color: rgb(0, 75, 191);
        }
        .section-title {
            font-size: 1.4em;
            margin-top: 30px;
            color: #1DB954;
        }
            
        .stats-for-geeks img {
            height: 200px !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>Smart Predictive Maintenance for Fishing Nets 🐟</div>", unsafe_allow_html=True)
st.markdown("This Streamlit app compares multiple deep learning models to detect damage in fishing nets.")

with st.expander("🛠️ Model Selection", expanded=True):
    selected_models = st.multiselect(
        "Select models to evaluate:",
        options=list(MODELS.keys()),
        default=list(MODELS.keys()),
        help="Choose one or more trained models to evaluate on the test dataset"
    )

    image_url = st.text_input("📷 Image URL for Prediction (Optional)", "")

if selected_models:
    class_names = load_class_names()
    all_models = load_all_models()
    results = []

    if not all_models:
        st.warning("❌ None of the selected models could be loaded. Please check the files.")
    else:
        if image_url:
            st.markdown("### 🔍 Prediction from Image URL")
            for model_name in selected_models:
                model = all_models.get(model_name)
                pred_index, confidence, img = predict_from_url(model, image_url)
                if pred_index is not None:
                    st.image(img, caption=f"{model_name} Prediction", width=224)
                    st.success(f"{model_name}: {class_names[pred_index]} ({confidence:.2f}%)")

                     # Stats for Geeks for Image URL Prediction
                    with st.expander(f"🧑‍🔬 Stats for Geeks: {model_name} (Image URL Prediction)"):
                        st.markdown(f"**Predicted Class:** {class_names[pred_index]}")
                        st.markdown(f"**Confidence:** {confidence:.2f}%")
                        # Show the image again, but larger for analysis
                        st.image(img, caption="Analysed Image", width=320)
                        # Show softmax probabilities if available
                        try:
                            img_array = np.expand_dims(np.array(img).astype('float32') / 255.0, axis=0)
                            probs = model.predict(img_array)[0]
                            fig, ax = plt.subplots(figsize=(8, 3))
                            ax.bar(class_names, probs * 100, color='teal')
                            ax.set_ylabel("Probability (%)")
                            ax.set_title("Class Probabilities")
                            ax.set_ylim([0, 100])
                            for i, v in enumerate(probs * 100):
                                ax.text(i, v + 1, f"{v:.2f}%", ha='center', fontsize=8)
                            st.pyplot(fig)
                        except Exception as e:
                            st.info(f"Could not display class probabilities: {e}")

    # Add button to start evaluation
    evaluate = st.button("🚀 Evaluate Models on Test Dataset")

    if evaluate:
        with st.spinner("🔍 Evaluating models on test dataset..."):
            for class_dir in os.listdir(TEST_DIR):
                class_path = os.path.join(TEST_DIR, class_dir)
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    for model_name in selected_models:
                        model = all_models.get(model_name)
                        if model is None:
                            continue
                        pred_index, confidence = predict_from_image(model, img_path)
                        pred_label = class_names[pred_index]
                        results.append({
                            "Image": img_name,
                            "Actual": class_dir,
                            "Model": model_name,
                            "Predicted": pred_label,
                            "Confidence": f"{confidence:.2f}%"
                        })

        if not results:
            st.warning("⚠️ No predictions were generated. Make sure at least one model works.")
        else:
            df = pd.DataFrame(results)
            st.markdown("<div class='section-title'>📊 Prediction Table</div>", unsafe_allow_html=True)
            
            st.markdown("""
                <div style='max-height: 300px; overflow-y: auto;'>
                """ + df.to_html(escape=False, index=False, justify='center') + """
                </div>
                """, unsafe_allow_html=True)
            # st.markdown(df.to_html(escape=False, index=False, justify='center'), unsafe_allow_html=True)

            st.markdown("<div class='section-title'>📈 Summary Statistics</div>", unsafe_allow_html=True)
            for model_name in selected_models:
                subset = df[df['Model'] == model_name]
                correct = (subset['Actual'] == subset['Predicted']).sum()
                total = len(subset)
                if total == 0:
                    st.warning(f"{model_name}: No data available for this model.")
                else:
                    st.success(f"{model_name}: {correct}/{total} correct ({(correct/total)*100:.2f}%)")

            with st.expander("📊 Stats for Geeks (Advanced Visualisation)"):
                st.markdown("<div class='stats-for-geeks'>", unsafe_allow_html=True)
                st.markdown("#### Confusion Matrix")
                for model_name in selected_models:
                    subset = df[df['Model'] == model_name]
                    if subset.empty:
                        st.info(f"No data available for {model_name}.")
                        continue
                    cm_data = pd.crosstab(subset['Actual'], subset['Predicted'])
                    if cm_data.empty:
                        st.warning(f"{model_name}: Not enough prediction data for confusion matrix.")
                        continue
                    fig, ax = plt.subplots()
                    ax.matshow(cm_data, cmap='Blues')
                    for (i, j), val in np.ndenumerate(cm_data.values):
                        ax.text(j, i, val, ha='center', va='center')
                    ax.set_xticks(range(len(cm_data.columns)))
                    ax.set_yticks(range(len(cm_data.index)))
                    ax.set_xticklabels(cm_data.columns)
                    ax.set_yticklabels(cm_data.index)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    ax.set_title(f"{model_name} - Confusion Matrix")
                    st.pyplot(fig)

                st.markdown("### Model Prediction Breakdown")

                st.markdown("#### Accuracy Bar Chart")
                acc_data = []
                for model_name in selected_models:
                    subset = df[df['Model'] == model_name]
                    correct = (subset['Actual'] == subset['Predicted']).sum()
                    total = len(subset)
                    acc = (correct / total) * 100 if total else 0
                    acc_data.append((model_name, acc))

                if acc_data:
                    labels, values = zip(*acc_data)
                    fig, ax = plt.subplots()
                    ax.bar(labels, values, color=['skyblue', 'salmon'])
                    ax.set_ylabel("Accuracy (%)")
                    ax.set_title("Model Accuracy Comparison")
                    ax.set_ylim([0, 100])
                    for i, v in enumerate(values):
                        ax.text(i, v + 1, f"{v:.2f}%", ha='center')
                    st.pyplot(fig)

                st.markdown("#### Accuracy Line Chart")
                if acc_data:
                    fig, ax = plt.subplots()
                    ax.plot(labels, values, marker='o', linestyle='-', color='green')
                    ax.set_ylabel("Accuracy (%)")
                    ax.set_title("Model Accuracy Trend")
                    ax.set_ylim([0, 100])
                    for i, v in enumerate(values):
                        ax.text(i, v + 1, f"{v:.2f}%", ha='center')
                    st.pyplot(fig)

                st.markdown("#### Per-Class Accuracy")
                for model_name in selected_models:
                    st.markdown(f"**{model_name}**")
                    subset = df[df['Model'] == model_name]
                    class_accuracies = []
                    for class_name in sorted(df['Actual'].unique()):
                        class_subset = subset[subset['Actual'] == class_name]
                        total = len(class_subset)
                        correct = (class_subset['Actual'] == class_subset['Predicted']).sum()
                        acc = (correct / total) * 100 if total else 0
                        class_accuracies.append((class_name, acc))
                    class_labels, class_values = zip(*class_accuracies)
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.bar(class_labels, class_values, color='mediumpurple')
                    ax.set_ylim([0, 100])
                    ax.set_ylabel("Accuracy (%)")
                    ax.set_title(f"Per-Class Accuracy for {model_name}")
                    for i, v in enumerate(class_values):
                        ax.text(i, v + 1, f"{v:.2f}%", ha='center')
                    st.pyplot(fig)
                for model_name in selected_models:
                    subset = df[df['Model'] == model_name]
                    pie_data = subset['Predicted'].value_counts()
                    if pie_data.empty:
                        st.info(f"No predictions to show for {model_name}.")
                        continue
                    fig, ax = plt.subplots()
                    ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
                    ax.set_title(f"{model_name} - Prediction Distribution")
                    st.pyplot(fig)
else:
    st.warning("⚠️ Please select at least one model to evaluate.")
