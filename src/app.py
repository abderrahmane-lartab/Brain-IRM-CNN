# app.py - Interface Streamlit pour BrainScan AI
import os
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Brain MRI Classification", page_icon="🧠", layout="wide")

CATEGORIES = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = 224

@st.cache_resource
def load_trained_model():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(app_dir)
    models_dir = os.path.join(project_dir, 'models')
    for name in ['best_model.keras', 'brain_mri_cnn.keras', 'brain_mri_cnn.h5']:
        p = os.path.join(models_dir, name)
        if os.path.exists(p):
            return load_model(p)
    raise FileNotFoundError('Aucun modèle trouvé dans models/')

def preprocess_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def main():
    st.title("Brain MRI Tumor Classification")
    st.markdown("Aide au diagnostic : glioma, meningioma, notumor, pituitary")
    try:
        model = load_trained_model()
    except Exception as e:
        st.error(str(e))
        return
    uploaded_file = st.file_uploader("Choisir une image IRM (jpg, jpeg, png, bmp)", type=['jpg','jpeg','png','bmp'])
    if uploaded_file is not None:
        img = preprocess_image(uploaded_file)
        if img is None:
            st.error("Impossible de lire l'image.")
            return
        st.image((img[0]*255).astype(np.uint8), caption="Image sélectionnée", width=300)
        preds = model.predict(img, verbose=0)[0]
        pred_idx = np.argmax(preds)
        st.success(f"Prédiction : **{CATEGORIES[pred_idx]}** ({preds[pred_idx]*100:.2f}%)")
        st.subheader("Probabilités")
        for i, c in enumerate(CATEGORIES):
            st.write(f"- {c}: {preds[i]*100:.2f}%")

if __name__ == "__main__":
    main()
