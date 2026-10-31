# Guide de démonstration

## Démarrage rapide — BAGLS

1. Lancez `streamlit run app.py` avec le Python de `.venv`.
2. Laissez « Utiliser l’exemple BAGLS » activé.
3. Conservez la modalité « Larynx / BAGLS ».
4. Choisissez un prétraitement. L’onglet **Aperçu & histogramme** affiche immédiatement l’original, le résultat et le masque annoté.
5. Sélectionnez une ou plusieurs méthodes conventionnelles.
6. Ouvrez **Segmentations** pour consulter masques, superpositions, temps, Dice et IoU.
7. Téléchargez un masque PNG ou le tableau CSV.
8. Ouvrez **Traçabilité** pour voir les paramètres et les méthodes réellement exécutées.

## Démonstration recommandée

- Prétraitement : `Médian`, noyau 3.
- Méthodes : `Otsu`, `K-means`, `Mean Shift`, `Chan–Vese`, `Watershed` et `Seuil manuel`.
- TP3 sélectionne toujours la classe sombre, exactement comme le notebook; ce choix n’est pas modifiable.
- Le contrôle de région concerne uniquement l’extension Watershed; il ne modifie pas Mean Shift, K-means ou Chan–Vese TP3.

Les scores affichés décrivent uniquement l’image BAGLS de démonstration et son masque. Ils ne sont pas une estimation générale de performance.

## Tester un fichier utilisateur

1. Désactivez l’exemple BAGLS.
2. Choisissez la modalité déclarée.
3. Uploadez un PNG, JPEG ou TIFF réel de moins de 25 Mo et 12 mégapixels.
4. Ajoutez éventuellement un masque de référence exactement aligné.

Le contenu réel du fichier est contrôlé par Pillow; changer seulement son extension ne contourne pas la validation. Si les dimensions du masque diffèrent, les métriques sont désactivées au lieu de redimensionner artificiellement la vérité terrain.

## U-Net BUSI

1. Déclarez « Échographie mammaire / BUSI ».
2. Sélectionnez « U-Net BUSI ».
3. Vérifiez que TensorFlow est installé et fournissez un checkpoint authentique via `models/`, `MEDIMAGE_UNET_CHECKPOINT` ou l’upload temporaire.
4. L’application valide les formes d’entrée/sortie avant l’inférence.

Le checkpoint TP4 est disponible dans `models/unet_breast_batch16_epochs50.hdf5` et TensorFlow 2.20.0 est installé dans l'environnement local. Pour une démonstration scientifiquement valide, choisir la modalité BUSI et importer une échographie mammaire; ajouter son masque correctement aligné uniquement pour calculer Dice/IoU. L'application affiche l'original, la carte de probabilité, le masque strict `> 0.5` et la superposition. Ne pas utiliser l'exemple BAGLS pour évaluer ce modèle BUSI.

## Messages importants

- **Méthode non exécutée** : les autres méthodes continuent et l’erreur reste visible.
- **Masque non aligné** : Dice/IoU sont désactivés.
- **U-Net incompatible** : vérifiez la modalité, TensorFlow et le checkpoint.
- **Image refusée** : vérifiez format réel, taille en octets, dimensions et intégrité du fichier.
