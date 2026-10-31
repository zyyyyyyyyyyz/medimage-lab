# Poids du modèle (NON inclus)

Le notebook exécuté TP4 contient des appels de sauvegarde vers Google Drive :
- `unet_breast_batch16_epochs10.hdf5`
- `unet_breast_batch16_epochs50.hdf5`

Les poids **ne sont pas intégrés dans les fichiers `.ipynb`**. Aucun de ces `.hdf5` n'a été retrouvé dans la bibliothèque ChatGPT parcourue. Téléchargez votre véritable checkpoint `unet_breast_batch16_epochs50.hdf5` depuis Google Drive et placez-le **dans ce dossier**.

**Ne prétendez pas que le modèle est prêt à inférer sans ce fichier.** L'application affiche un avertissement, sans substituer de poids aléatoires.

Le modèle s'applique uniquement à des échographies mammaires de type BUSI, avec le prétraitement enregistré dans `core/unet_tp4.py` et une entrée `256 × 256 × 1`.
