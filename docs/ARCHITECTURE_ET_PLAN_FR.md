# Projet — Plateforme web d'analyse et traitement d'images médicales

## Objectif
Concevoir une application **pédagogique**, utilisable depuis un navigateur, pour charger une image médicale, lui appliquer des opérations conventionnelles, réaliser des segmentations et comparer leurs résultats avec un modèle entraîné lorsque celui-ci est compatible avec la modalité.

## Inventaire issu des TP exécutés

| Source | Réutilisable | Réalité de l'exécution | Réserve |
|---|---|---|---|
| TP1 — introduction | lecture couleur/gris, dimensions, histogramme | cellules exécutées | `frame856.png`, `frame442.png`, `patient4.mp4` ne sont pas nécessaires au web MVP |
| TP3 — segmentation | filtrage médian, Mean Shift, K-means, nettoyage, Chan–Vese, Dice/IoU | sorties et scores enregistrés | image glottique BAGLS, pas BUSI |
| TP4 — U-Net | architecture Keras, préparation BUSI, normalisation axis=1, évaluation | 10 et 50 époques exécutées | poids .hdf5 sauvegardés dans Google Drive, absents de la bibliothèque vérifiée |
| TP FIRE — recalage | extraction de points SIFT/ORB, RANSAC, similarité/affine/homographie | notebook exécuté, 134 paires, 402 évaluations | le recalage est une extension, pas une méthode de segmentation |

## Modules livrés dans le MVP
1. **Acquisition et contrôle** : PNG/JPG/TIFF, conversion gris et image de démonstration BAGLS ; validation d'une taille maximale de 12 Mpixels.
2. **Prétraitement** : Gaussien, médian, bilatéral, CLAHE et égalisation d'histogramme. Histogramme visuel.
3. **Segmentation conventionnelle** : Otsu, seuillage manuel, K-means, Mean Shift, Chan–Vese, watershed. Choix de région sombre ou claire, nettoyage facultatif, récupération masque PNG.
4. **Segmentation Deep Learning** : architecture U-Net exact TP4, modèle entraîné BUSI 50 époques **uniquement si ses poids réels sont présents**. Aucune prédiction simulée. Prétraitement de réseau fidèlement reproduit, entrée 256×256×1, post-traitement mask.
5. **Comparaison** : résultat original / prétraité / superposition / masque ; Dice, IoU si vérité terrain ; CSV de comparaison ; temps de calcul des méthodes conventionnelles.

## Schéma

```text
Navigateur — Streamlit
       │
       ├── Import image + type de modalité + éventuelle annotation
       │                │
       │                ├── Branche conventionnelle
       │                │       filtre / contraste → segmentation → masque
       │                │
       │                └── Branche U-Net BUSI (écho mammaire uniquement)
       │                        image ORIGINALE → normalisation TP4
       │                        → poids .hdf5 → prédiction 256×256 → masque
       │
       └── Visualisation & comparaison (image, masque, overlay, métriques)
```

**Important :** le filtre sélectionné dans l'interface n'est pas injecté par défaut dans le U-Net. Le modèle a été entraîné avec une normalisation déterminée ; une nouvelle amélioration de contraste pourrait changer la distribution d'entrée et perturber sa prédiction.

## Plan de travail recommandé

**Phase A — Récupération scientifique** (réalisée dans ce paquet) : inventaire des notebooks exécutés, extraction du code, reconstruction du U-Net fidèle et récupération d'un exemple BAGLS.

**Phase B — MVP conventionnel** (code fourni et tests unitaires passés) : import, prétraitement, 6 méthodes conventionnelles, visualisation et export. Le lancement de l'interface Streamlit doit encore être validé sur un poste équipé de Streamlit.

**Phase C — Activation Deep Learning** (bloquée par les fichiers de poids) : récupérer les `.hdf5` depuis le Drive qui a servi lors du TP4, charger le modèle à 50 époques, effectuer des tests de non-régression sur BUSI.

**Phase D — Évaluation équitable** : images/masques issus de la même modalité, exécuter toutes les méthodes pertinentes sur le même jeu de test, calculer Dice/IoU **par image**, moyenne et écart-type, renseigner temps CPU/GPU et configuration. Ne pas comparer directement le Dice BAGLS du TP3 au Dice BUSI du TP4.

**Phase E — Extension facultative** : recalage de fonds d'œil (TP FIRE), prise en charge DICOM avec anonymisation, étude de nouveaux modèles de segmentation préentraînés propres aux modalités ciblées, sauvegarde de sessions.

## Sécurité et périmètre
L'application n'est pas un dispositif médical et n'effectue pas de diagnostic. Le MVP est à exécuter localement, avec des images ouvertes / anonymisées. Avant déploiement accessible à d'autres utilisateurs, ajouter restrictions d'accès, suppression des uploads, journalisation minimale, limites de durée/ressources, contrôle de formats et politique sur les données de santé.

## Vérifications effectuées
- Les `.ipynb` originaux ont été analysés comme JSON pour distinguer code et sorties exécutées.
- `core/unet_tp4.py` reprend le **corps original** de `simple_unet_model` du TP4.
- `preprocess_busi_like_tp` reproduit mathématiquement `normalize(image_dataset, axis=1)` sur le batch en 3D avant ajout du canal : normalisation selon l'axe hauteur.
- Tests automatisés sur images synthétiques : forme binaire, métriques, toutes les méthodes et normalisation TP4.
- Pas de TensorFlow installé dans l'environnement de test ici : l'inférence ne peut pas être validée sans dépendance **et** sans checkpoint.


## Ressources ajoutées — TP2 et extensions pédagogiques
Les notebooks non exécutés de `Livraison_Notebooks.zip` sont inclus dans `reference_notebooks/`. Le TP2 aborde filtrage, morphologie, contours et Watershed ; l’atelier multi-modalités ajoute la comparaison du débruitage/seuillage en endoscopie, radiographie et IRM. Les implémentations existantes de l’application ne changent pas dans cette version. Le détail des sources et doublons figure dans `MANIFEST_COMPLET.json`.
