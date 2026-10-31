# Notebook source: 02-Pretraitement_images_medicales.ipynb
# Original notebook code cells, copied without modifications.
# These are notebook fragments; may require input data, Colab, or earlier cells.
# Not guaranteed to run as a standalone .py file.


# -------- ORIGINAL NOTEBOOK CELL 2 --------
# Cellule de configuration - à exécuter en premier (Colab)
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# Modules skimage utilisés dans les exercices (filtrage, morphologie)
from skimage.filters import median, gaussian
from skimage.filters.rank import mean as mean_rank
from skimage.morphology import disk

# --- Détection robuste du dossier de données ---
DATA_DIR = 'Data_At2/Data_At2'
if not os.path.isfile(os.path.join(DATA_DIR, 'frame442.png')):
    for cand in ['Data_At2', 'data', '.', '/content/Data_At2/Data_At2']:
        if os.path.isfile(os.path.join(cand, 'frame442.png')):
            DATA_DIR = cand
            break

print(f'Dossier de données utilisé : {DATA_DIR}')
print('Images disponibles :', sorted(os.listdir(DATA_DIR)))

# -------- ORIGINAL NOTEBOOK CELL 5 --------
# 1) Ajout d'un bruit gaussien (moyenne 0, écart-type 25) avec numpy
img = cv2.imread(os.path.join(DATA_DIR, 'frame442.png'), cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f'frame442.png introuvable dans {DATA_DIR}')

# Bruit gaussien : moyenne 0, écart-type 25
noise = np.random.normal(loc=0.0, scale=25.0, size=img.shape)
img_noisy = img.astype(np.float32) + noise
img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Image originale frame442')
axes[0].axis('off')
axes[1].imshow(img_noisy, cmap='gray')
axes[1].set_title('Image bruitée (moyenne=0, écart-type=25)')
axes[1].axis('off')
plt.tight_layout()
plt.show()

print('Écart-type du bruit réellement ajouté :',
      np.std(img_noisy.astype(np.float32) - img.astype(np.float32)))

# -------- ORIGINAL NOTEBOOK CELL 7 --------
# 2) Filtrage de l'image bruitée avec skimage
# Modules utilisés : skimage.filters (median, gaussian) et skimage.filters.rank (mean)

footprint = disk(3)   # élément structurant de rayon 3

# Filtre moyenneur (moyenne locale, skimage.filters.rank.mean)
img_mean = mean_rank(img_noisy, footprint)

# Filtre médian (skimage.filters.median)
img_median = median(img_noisy, footprint)

# Filtre gaussien (skimage.filters.gaussian) : renvoie des flottants -> reconversion uint8
img_gauss = gaussian(img_noisy, sigma=1.0, preserve_range=True)
img_gauss = np.clip(img_gauss, 0, 255).astype(np.uint8)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
images = [img, img_noisy, img_mean, img_median, img_gauss]
titles = ['Image originale', 'Image bruitée', 'Filtre moyenneur',
          'Filtre médian', 'Filtre gaussien']
for ax, im, t in zip(axes.ravel(), images, titles):
    ax.imshow(im, cmap='gray')
    ax.set_title(t)
    ax.axis('off')
axes[1, 2].axis('off')   # cellule vide
plt.tight_layout()
plt.show()

# Interprétation :
# - Filtre moyenneur : lisse fortement (moyenne locale), atténue bien le bruit mais
#   adoucit/estompe les contours (effet de flou).
# - Filtre médian : très efficace pour réduire le bruit tout en préservant mieux les
#   contours nets que le filtre moyenneur.
# - Filtre gaussien : pondère les voisins selon une gaussienne, bon compromis
#   débruitage/préservation des contours ; son intensité dépend de sigma.

# -------- ORIGINAL NOTEBOOK CELL 10 --------
# 1) Histogramme de l'image et choix manuel du seuil de binarisation
img = cv2.imread(os.path.join(DATA_DIR, 'frame442.png'), cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f'frame442.png introuvable dans {DATA_DIR}')

# Histogramme des niveaux de gris
plt.figure(figsize=(10, 5))
plt.hist(img.ravel(), bins=256, range=(0, 256), color='gray', edgecolor='black')
plt.title('Histogramme de frame442 (niveaux de gris)')
plt.xlabel('Intensité')
plt.ylabel('Nombre de pixels')
plt.grid(alpha=0.3)
plt.show()

# Discussion : l'histogramme est globalement bimodal -
#   * un pic sombre (faibles intensités) : le glottis (ouverture des cordes vocales)
#   * un pic clair (fortes intensités) : les tissus environnants
# Le seuil idéal se situe dans la vallée entre ces deux pics.

seuil_manuel = 50   # valeur de la vallée observée sur l'histogramme
_, img_bin = cv2.threshold(img, seuil_manuel, 255, cv2.THRESH_BINARY)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Image originale')
axes[0].axis('off')
axes[1].imshow(img_bin, cmap='gray')
axes[1].set_title(f'Binarisation manuelle (seuil = {seuil_manuel})')
axes[1].axis('off')
plt.tight_layout()
plt.show()
print('Seuil choisi manuellement à partir de l\'histogramme :', seuil_manuel)

# -------- ORIGINAL NOTEBOOK CELL 12 --------
# 2) Seuillage optimal d'Otsu avec opencv
seuil_otsu, img_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Image originale')
axes[0].axis('off')
axes[1].imshow(img_otsu, cmap='gray')
axes[1].set_title(f'Seuillage d\'Otsu (seuil = {seuil_otsu:.0f})')
axes[1].axis('off')
plt.tight_layout()
plt.show()

# 3) Comparaison des deux méthodes
print('Seuil manuel (lecture de l\'histogramme) :', seuil_manuel)
print('Seuil optimal d\'Otsu                    :', seuil_otsu)
print('=> Les deux seuils sont proches : Otsu confirme le choix de la vallée de l\'histogramme.')

# -------- ORIGINAL NOTEBOOK CELL 16 --------
# 1) Lecture et affichage de l'image binaire seg1468.png
seg = cv2.imread(os.path.join(DATA_DIR, 'seg1468.png'), cv2.IMREAD_GRAYSCALE)
if seg is None:
    raise FileNotFoundError(f'seg1468.png introuvable dans {DATA_DIR}')

# Normalisation : toute valeur non nulle devient 255 (masque binaire)
seg_bin = np.where(seg > 0, 255, 0).astype(np.uint8)

print('Forme :', seg_bin.shape, '| Type :', seg_bin.dtype)
print('Valeurs uniques :', np.unique(seg_bin))

plt.figure(figsize=(7, 7))
plt.imshow(seg_bin, cmap='gray')
plt.title('Image binaire seg1468 (segmentation automatique)')
plt.axis('off')
plt.show()

# -------- ORIGINAL NOTEBOOK CELL 18 --------
# 2) Opérations morphologiques : érosion, dilatation, ouverture, fermeture
kernel = np.ones((5, 5), np.uint8)

eroded  = cv2.erode(seg_bin, kernel, iterations=1)
dilated = cv2.dilate(seg_bin, kernel, iterations=1)
opened  = cv2.morphologyEx(seg_bin, cv2.MORPH_OPEN, kernel)
closed  = cv2.morphologyEx(seg_bin, cv2.MORPH_CLOSE, kernel)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
results = [('Image binaire', seg_bin), ('Érosion', eroded), ('Dilatation', dilated),
           ('Ouverture', opened), ('Fermeture', closed)]
for ax, (t, im) in zip(axes.ravel(), results):
    ax.imshow(im, cmap='gray')
    ax.set_title(t)
    ax.axis('off')
axes[1, 2].axis('off')   # cellule vide
plt.tight_layout()
plt.show()

# Nombre de composantes connexes pour objectiver l'élimination des outliers
def n_components(mask):
    n, _ = cv2.connectedComponents(mask)
    return n - 1

print('Nb composantes - binaire :', n_components(seg_bin),
      '| ouverture :', n_components(opened),
      '| fermeture :', n_components(closed))

# Conclusion : l'OUVERTURE (érosion puis dilatation) supprime les petites régions
# (outliers) tout en conservant la taille de la région principale (le glottis).

# -------- ORIGINAL NOTEBOOK CELL 20 --------
# 3) Détection du contour du glottis par morphologie + arithmétique
# a) Nettoyage : ouverture puis conservation de la plus grande composante connexe
opened = cv2.morphologyEx(seg_bin, cv2.MORPH_OPEN, kernel)
n, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
if n > 1:
    big_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    glottis_mask = np.where(labels == big_idx, 255, 0).astype(np.uint8)
else:
    glottis_mask = opened

# b) Contour = soustraction arithmétique : masque - érosion(masque)
eroded_glottis = cv2.erode(glottis_mask, kernel, iterations=1)
contour_glottis = cv2.subtract(glottis_mask, eroded_glottis)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(opened, cmap='gray')
axes[0].set_title('Après ouverture')
axes[0].axis('off')
axes[1].imshow(glottis_mask, cmap='gray')
axes[1].set_title('Plus grande composante (glottis)')
axes[1].axis('off')
axes[2].imshow(contour_glottis, cmap='gray')
axes[2].set_title('Contour du glottis (soustraction)')
axes[2].axis('off')
plt.tight_layout()
plt.show()

# -------- ORIGINAL NOTEBOOK CELL 22 --------
# 4) Affichage du contour du glottis sur l'image originale
orig = cv2.imread(os.path.join(DATA_DIR, '01468_rgb.png'), cv2.IMREAD_UNCHANGED)
if orig is None:
    raise FileNotFoundError(f'01468_rgb.png introuvable dans {DATA_DIR}')

if orig.ndim == 2:
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)
else:
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)   # cv2 lit en BGR

# Le masque de contour peut avoir une taille différente -> redimensionnement
h, w = orig_rgb.shape[:2]
contour_resized = cv2.resize(contour_glottis, (w, h), interpolation=cv2.INTER_NEAREST)

overlay = orig_rgb.copy()
overlay[contour_resized > 0] = [255, 0, 0]   # contour en rouge

plt.figure(figsize=(8, 8))
plt.imshow(overlay)
plt.title('Contour du glottis superposé à l\'image originale')
plt.axis('off')
plt.show()

# -------- ORIGINAL NOTEBOOK CELL 25 --------
# 1) Détection de contours : Sobel et Laplacien (opencv)
img = cv2.imread(os.path.join(DATA_DIR, 'frame442.png'), cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f'frame442.png introuvable dans {DATA_DIR}')

# Sobel : gradient selon x et y puis magnitude
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel_mag = cv2.magnitude(sobelx, sobely)
sobel_mag = np.clip(sobel_mag, 0, 255).astype(np.uint8)

# Laplacien : dérivée seconde
lap = cv2.Laplacian(img, cv2.CV_64F)
lap = np.clip(np.abs(lap), 0, 255).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Image originale')
axes[0].axis('off')
axes[1].imshow(sobel_mag, cmap='gray')
axes[1].set_title('Sobel (magnitude du gradient)')
axes[1].axis('off')
axes[2].imshow(lap, cmap='gray')
axes[2].set_title('Laplacien')
axes[2].axis('off')
plt.tight_layout()
plt.show()

# Comparaison : Sobel (gradient du 1er ordre) met en évidence les contours orientés avec
# des bords épais ; le Laplacien (dérivée 2nd ordre) détecte les contours quelle que soit
# leur orientation mais est plus sensible au bruit.

# -------- ORIGINAL NOTEBOOK CELL 27 --------
# 2) Gradient morphologique = dilatation - érosion
kernel3 = np.ones((3, 3), np.uint8)

dil = cv2.dilate(img, kernel3, iterations=1)
ero = cv2.erode(img, kernel3, iterations=1)
grad_morph = cv2.subtract(dil, ero)                         # version arithmétique
grad_morph_cv = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel3)

print('Arithmétique == cv2.MORPH_GRADIENT :',
      np.array_equal(grad_morph, grad_morph_cv))

plt.figure(figsize=(8, 8))
plt.imshow(grad_morph, cmap='gray')
plt.title('Gradient morphologique (dilatation - érosion)')
plt.axis('off')
plt.show()

# -------- ORIGINAL NOTEBOOK CELL 29 --------
# 3) Suppression des contours qui ne correspondent pas au contour du glottis
# Binarisation du gradient morphologique pour garder les contours les plus marqués
_, grad_bin = cv2.threshold(grad_morph, 25, 255, cv2.THRESH_BINARY)

# Conservation de la plus grande composante connexe : le contour du glottis
n, labels, stats, _ = cv2.connectedComponentsWithStats(grad_bin)
if n > 1:
    big_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    glottis_contour = np.where(labels == big_idx, 255, 0).astype(np.uint8)
else:
    glottis_contour = grad_bin

# Fermeture légère pour obtenir un contour continu
glottis_contour = cv2.morphologyEx(glottis_contour, cv2.MORPH_CLOSE, kernel3)

overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
overlay[glottis_contour > 0] = [255, 0, 0]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(grad_morph, cmap='gray')
axes[0].set_title('Gradient morphologique')
axes[0].axis('off')
axes[1].imshow(grad_bin, cmap='gray')
axes[1].set_title('Contours binarisés')
axes[1].axis('off')
axes[2].imshow(overlay)
axes[2].set_title('Plus grande composante = contour du glottis')
axes[2].axis('off')
plt.tight_layout()
plt.show()

# -------- ORIGINAL NOTEBOOK CELL 32 --------
# 1) Watershed avec le contour du gradient morphologique comme marqueur
img_rgb = imageio.imread(os.path.join(DATA_DIR, 'frame442.png'))
if img_rgb.ndim == 2:
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
img_rgb = np.ascontiguousarray(img_rgb)

# Marqueur : contour du glottis (issu du gradient morphologique nettoyé)
marker = cv2.dilate(glottis_contour, kernel3, iterations=1)   # referme les petites ouvertures

# Région intérieure du glottis (remplissage du contour)
contours, _ = cv2.findContours(marker, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
interior = np.zeros_like(marker)
if contours:
    cv2.drawContours(interior, contours, -1, 255, thickness=cv2.FILLED)

# Construction des marqueurs : intérieur = 1, fond = 2, contour (inconnu) = 0
markers = np.zeros(img_rgb.shape[:2], dtype=np.int32)
markers[interior > 0] = 1
markers[interior == 0] = 2
markers[marker > 0] = 0

watershed_markers = cv2.watershed(img_rgb, markers)
boundary_morph = (watershed_markers == -1).astype(np.uint8) * 255

overlay = img_rgb.copy()
overlay[boundary_morph > 0] = [255, 0, 0]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img_rgb)
axes[0].set_title('Image originale')
axes[0].axis('off')
axes[1].imshow(marker, cmap='gray')
axes[1].set_title('Marqueur : contour du gradient morphologique')
axes[1].axis('off')
axes[2].imshow(overlay)
axes[2].set_title('Watershed - frontière en rouge')
axes[2].axis('off')
plt.tight_layout()
plt.show()

# -------- ORIGINAL NOTEBOOK CELL 34 --------
# 2) Watershed avec le contour ground truth (image segmentée automatiquement)
gt = cv2.imread(os.path.join(DATA_DIR, 'frame442_glottis.png'), cv2.IMREAD_GRAYSCALE)
if gt is None:
    raise FileNotFoundError(f'frame442_glottis.png introuvable dans {DATA_DIR}')

gt_bin = np.where(gt > 0, 255, 0).astype(np.uint8)
# Redimensionnement si les tailles diffèrent
h, w = img_rgb.shape[:2]
if gt_bin.shape[:2] != (h, w):
    gt_bin = cv2.resize(gt_bin, (w, h), interpolation=cv2.INTER_NEAREST)

# Contour ground truth : gradient morphologique du masque
gt_contour = cv2.morphologyEx(gt_bin, cv2.MORPH_GRADIENT, kernel3)

markers = np.zeros(img_rgb.shape[:2], dtype=np.int32)
markers[gt_bin > 0] = 1          # glottis (ground truth)
markers[gt_bin == 0] = 2         # fond
markers[gt_contour > 0] = 0      # contour = zone inconnue

watershed_markers_gt = cv2.watershed(img_rgb, markers)
boundary_gt = (watershed_markers_gt == -1).astype(np.uint8) * 255

overlay = img_rgb.copy()
overlay[boundary_gt > 0] = [0, 255, 0]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(gt_bin, cmap='gray')
axes[0].set_title('Masque ground truth')
axes[0].axis('off')
axes[1].imshow(gt_contour, cmap='gray')
axes[1].set_title('Contour ground truth (marqueur)')
axes[1].axis('off')
axes[2].imshow(overlay)
axes[2].set_title('Watershed - frontière en vert')
axes[2].axis('off')
plt.tight_layout()
plt.show()

# -------- ORIGINAL NOTEBOOK CELL 36 --------
# 3) Comparaison des deux résultats de watershed dans la même figure
overlay = img_rgb.copy()
overlay[boundary_morph > 0] = [255, 0, 0]    # rouge : marqueur gradient morphologique
overlay[boundary_gt > 0]     = [0, 255, 0]   # vert  : marqueur ground truth

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(boundary_morph, cmap='gray')
axes[0].set_title('Watershed (gradient morphologique)')
axes[0].axis('off')
axes[1].imshow(boundary_gt, cmap='gray')
axes[1].set_title('Watershed (ground truth)')
axes[1].axis('off')
axes[2].imshow(overlay)
axes[2].set_title('Comparaison : rouge = gradient morphologique, vert = ground truth')
axes[2].axis('off')
plt.tight_layout()
plt.show()

# Interprétation : les deux frontières sont très proches. Le marqueur ground truth
# (image segmentée automatiquement) sert de référence pour valider la méthode
# conventionnelle basée sur le gradient morphologique.
