# MedImage Lab

Application Streamlit pédagogique de prétraitement, segmentation et comparaison d’images médicales, construite à partir des TP du projet universitaire.

> Prototype de formation et de recherche uniquement. Il ne produit aucun diagnostic et n’est pas un dispositif médical.

## Fonctionnalités

- Import PNG, JPEG et TIFF avec validation du contenu, limite de 25 Mo et 12 mégapixels.
- Exemple BAGLS fourni avec masque de référence aligné.
- Prétraitements : médian, gaussien, bilatéral, CLAHE et égalisation d’histogramme.
- Segmentations TP fidèles : seuil manuel/Otsu TP2 et K-means/Mean Shift/Chan–Vese TP3.
- Watershed automatique clairement identifié comme extension web non équivalente au Watershed guidé du TP2.
- Comparaison visuelle : original, prétraité, masque et superposition.
- Histogramme, durées, Dice et IoU lorsque la vérité terrain est disponible et alignée.
- Export des masques en PNG et des résultats en CSV.
- Architecture U-Net BUSI du TP4 et normalisation originale préservées. L’inférence n’est activée qu’avec TensorFlow et un vrai checkpoint compatible.

## Installation vérifiée

Prérequis : Python 3.11.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip==24.3.1
.\.venv\Scripts\python.exe -m pip install -r requirements-baseline-lock.txt
.\.venv\Scripts\python.exe -m pip check
```

Lancer l’application :

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Puis ouvrir l’adresse locale affichée, généralement `http://localhost:8501`.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -vv
```

La livraison fidèle aux TP avec l'intégration du checkpoint collecte 12 tests et obtient 12 réussites. Les preuves et limites sont détaillées dans `FINAL_STATUS.md`.

## U-Net BUSI

Le checkpoint TP4 fourni est chargé par défaut depuis :

```text
models/unet_breast_batch16_epochs50.hdf5
```

Trois modes de configuration sont disponibles :

1. déposer ce vrai fichier dans `models/` ;
2. définir `MEDIMAGE_UNET_CHECKPOINT` vers un fichier local administré ;
3. sélectionner « U-Net BUSI » et uploader temporairement un `.h5`, `.hdf5` ou `.keras` dans l’interface.

Installer TensorFlow séparément si nécessaire :

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dl.txt
```

L’application charge le fichier avec `tf.keras.models.load_model(path, compile=False)` et vérifie que l’entrée et la sortie sont `(256, 256, 1)`. L'inférence affiche l'image originale, la probabilité, le masque strict `> 0.5` et la superposition. Elle ne crée jamais de prédiction avec des poids aléatoires. Une architecture seule n’est pas un modèle préentraîné.

## Intégrité scientifique

- Le U-Net cible les échographies mammaires BUSI, pas BAGLS, FIRE, IRM ou d’autres modalités.
- Les scores BUSI et BAGLS ne doivent pas être réunis dans un même benchmark.
- La normalisation TP4 reste une normalisation L2 `axis=1`, pas une division par 255.
- Dice et IoU exigent un masque de référence de mêmes dimensions et correctement aligné.
- Les segmentations conventionnelles détectent des régions d’intensité et ne garantissent pas une anatomie correcte.

Voir [DEMO_GUIDE.md](DEMO_GUIDE.md) pour une démonstration guidée et [FINAL_STATUS.md](FINAL_STATUS.md) pour l’état de livraison.

La traçabilité scientifique complète, cellule par cellule, se trouve dans [TP_PARITY_REPORT.md](TP_PARITY_REPORT.md).
