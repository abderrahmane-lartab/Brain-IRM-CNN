# Classification d’IRM cérébrales par CNN pour l’aide au diagnostic des tumeurs

## Description
Ce projet développe un modèle CNN avec TensorFlow/Keras pour classifier des images IRM cérébrales et détecter la présence de tumeurs (glioma, meningioma, notumor, pituitary). Le prétraitement est réalisé avec OpenCV. Une interface Streamlit permet de tester des prédictions.

## Structure des fichiers
```
Brain-IRM-CNN/
├── data/                        # Dataset par classe
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
├── models/                      # Modèles sauvegardés (.keras, .h5)
├── notebooks/
│   ├── pretraitement.ipynb      # Chargement, prétraitement, exploration, split
│   └── architecture_cnn.ipynb  # CNN, entraînement, évaluation, déploiement
├── src/
│   └── app.py                   # Interface Streamlit
├── requirements.txt
└── README.md
```

## Environnement
- Python 3.12 recommandé (TensorFlow CPU 2.21.0)
- Installer les dépendances :
```bash
python -m venv .venv
source .venv/bin/activate  # Linux
pip install -r requirements.txt
```

## Exécution
1. Lancer le notebook de prétraitement :
   - `jupyter notebook notebooks/pretraitement.ipynb` (ou jupyterlab)
2. Lancer le notebook CNN :
   - `jupyter notebook notebooks/architecture_cnn.ipynb`

## Interface Streamlit
Après avoir entraîné et sauvegardé le modèle (dans `models/`) :
```bash
cd /home/abdo/Documents/Brain-IRM-CNN
streamlit run src/app.py
```

## Détails
- Redimensionnement : 224×224
- Normalisation : [0,1]
- Encodage : one-hot
- Optimiseur : Adam (lr 1e-4), perte : categorical_crossentropy
- Callbacks : ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
- Rééquilibrage via `class_weight`
