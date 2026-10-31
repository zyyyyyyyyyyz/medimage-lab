# Notebook source: atelier2_part1_multi-modalites.ipynb
# Original notebook code cells, copied without modifications.
# These are notebook fragments; may require input data, Colab, or earlier cells.
# Not guaranteed to run as a standalone .py file.


# -------- ORIGINAL NOTEBOOK CELL 2 --------
# ============================================================
# CELLULE DE MISE EN PLACE : imports, chemins, chargement des images
# ============================================================
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio

try:
    from skimage import filters, morphology
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("AVERTISSEMENT : skimage indisponible, les filtres skimage seront remplacés par OpenCV.")

np.random.seed(42)
plt.rcParams["figure.dpi"] = 90

# ------------------------------------------------------------
# 1) Jeu de données : 3 fichiers connus (dossier data2.1)
#    ATTENTION : la modalité "radiographie" est ABSENTE de ce TP.
#    Aucune des 3 images n'est une radiographie (pas de rayons X) :
#    on ne doit JAMAIS qualifier l'une d'elles d'image "radio"/"X-ray".
#    Les étiquettes ci-dessous sont FIXES : on ne déduit JAMAIS la
#    modalité à partir du nom du fichier.
# ------------------------------------------------------------
KNOWN_FILES = {
    "endoscopie": "frame442.png",   # endoscopie des cordes vocales (couleur, texturée, reflets/ombres)
    "irm_t1": "Tr-pi_1.jpg",        # IRM cérébrale : coupe coronale pondérée T1
    "irm_t2": "Te-no_264.jpg",      # IRM cérébrale : coupe axiale pondérée T2
}

# ------------------------------------------------------------
# 2) Détection robuste du dossier de données
#    On privilégie le dossier contenant les 3 fichiers connus (data2.1).
# ------------------------------------------------------------
DATA_DIR_CANDIDATES = [
    "data2.1",
    "Data_At2/Data_At2",
    "Data_At2",
    "data",
    ".",
    "/content/data2.1",
    "/content/Data_At2/Data_At2",
]

DATA_DIR = None
# Préférence stricte : le dossier doit contenir les 3 fichiers connus.
for candidate in DATA_DIR_CANDIDATES:
    if os.path.isdir(candidate) and all(
        os.path.isfile(os.path.join(candidate, fname)) for fname in KNOWN_FILES.values()
    ):
        DATA_DIR = candidate
        break

# Repli 1 : dossier contenant au moins un des 3 fichiers connus.
if DATA_DIR is None:
    for candidate in DATA_DIR_CANDIDATES:
        if os.path.isdir(candidate) and any(
            os.path.isfile(os.path.join(candidate, fname)) for fname in KNOWN_FILES.values()
        ):
            DATA_DIR = candidate
            break

# Repli 2 : n'importe quel dossier de données existant.
if DATA_DIR is None:
    for candidate in DATA_DIR_CANDIDATES:
        if os.path.isdir(candidate):
            DATA_DIR = candidate
            break

print(f"DATA_DIR = {DATA_DIR!r}")

# ------------------------------------------------------------
# 3) Chemins complets des 3 images (toujours via os.path.join)
# ------------------------------------------------------------
MODALITY_PATHS = {
    "endoscopie": os.path.join(DATA_DIR, "frame442.png"),
    "irm_t1": os.path.join(DATA_DIR, "Tr-pi_1.jpg"),
    "irm_t2": os.path.join(DATA_DIR, "Te-no_264.jpg"),
}

# ------------------------------------------------------------
# 4) Avertissement clair si un fichier est manquant
# ------------------------------------------------------------
MISSING_FILES = [p for p in MODALITY_PATHS.values() if not os.path.isfile(p)]
if MISSING_FILES:
    print("AVERTISSEMENT : fichier(s) introuvable(s) :")
    for p in MISSING_FILES:
        print(f"   - {p}")
    print("  -> Vérifiez que le dossier data2.1 est présent à côté du notebook")
    print("     (ou uploadez les fichiers puis relancez cette cellule).")
else:
    print("Les 3 fichiers du jeu de données sont présents.")

if DATA_DIR and os.path.isdir(DATA_DIR):
    image_files = sorted(
        f for f in os.listdir(DATA_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    )
    if image_files:
        print("Fichiers image présents dans DATA_DIR (pour information) :")
        for fname in image_files:
            print(f"   - {fname}")

# ------------------------------------------------------------
# 5) Chargement robuste (niveaux de gris + couleur, dtype quelconque)
# ------------------------------------------------------------
def _to_uint8(img):
    # Convertit proprement une image quelconque en uint8 [0, 255].
    if img.dtype == np.uint8:
        return img
    if img.max() > 255:
        return (img.astype(np.float32) / img.max() * 255.0).astype(np.uint8)
    return img.astype(np.uint8)


def _read_raw(path):
    # Lecture brute via imageio avec gestion d'erreur.
    try:
        raw = imageio.imread(path)
    except Exception as exc:
        print(f"AVERTISSEMENT : lecture impossible de {path} -> {exc}")
        return None
    if raw is None or raw.size == 0:
        return None
    return raw


def load_grayscale(path):
    # Charge une image et la convertit proprement en niveaux de gris uint8.
    if path is None or not os.path.isfile(path):
        return None
    raw = _read_raw(path)
    if raw is None:
        return None
    if raw.ndim == 2:
        return _to_uint8(raw)
    if raw.ndim >= 3:
        return cv2.cvtColor(_to_uint8(raw[:, :, :3]), cv2.COLOR_RGB2GRAY)
    print(f"AVERTISSEMENT : dimensions inattendues {raw.shape} pour {path}")
    return None


def load_rgb(path):
    # Charge une image en couleur (3 canaux) pour l'affichage.
    if path is None or not os.path.isfile(path):
        return None
    raw = _read_raw(path)
    if raw is None:
        return None
    raw_u8 = _to_uint8(raw)
    if raw.ndim == 2:
        return cv2.cvtColor(raw_u8, cv2.COLOR_GRAY2RGB)
    return raw_u8[:, :, :3]


def is_color_image(rgb):
    # True si l'image contient réellement des canaux différents
    # (une image grise lue en RGB a des canaux identiques -> False).
    if rgb is None:
        return False
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    return bool(np.any((r != g) | (r != b)))

# ------------------------------------------------------------
# 6) Chargement des 3 modalités (étiquettes fixes, pas d'inférence par nom)
# ------------------------------------------------------------
MODALITIES = {}
for label, path in MODALITY_PATHS.items():
    gray = load_grayscale(path)
    if gray is None:
        continue  # déjà averti ci-dessus
    rgb = load_rgb(path)
    MODALITIES[label] = {
        "path": path,
        "filename": os.path.basename(path),
        "gray": gray,
        "rgb": rgb,
        "color": is_color_image(rgb),
    }

print(f"\nModalités disponibles : {len(MODALITIES)}")
for label, info in MODALITIES.items():
    color_txt = "couleur" if info["color"] else "niveaux de gris"
    print(f"  - {label}: {info['path']} | forme {info['gray'].shape} | {color_txt}")

# ------------------------------------------------------------
# 7) Dossier des figures (pour le rapport LaTeX)
# ------------------------------------------------------------
FIGURES_DIR = os.path.join(os.getcwd(), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
print(f"\nLes figures seront enregistrées dans : {FIGURES_DIR}")


def save_fig(fig, filename):
    # Enregistre la figure courante dans le dossier figures/.
    out_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Figure enregistrée : {out_path}")
    return out_path

# -------- ORIGINAL NOTEBOOK CELL 5 --------
# ============================================================
# Exercice 1 — 1) Bruit gaussien : moyenne 0, écart-type 25
# ============================================================
NOISE_MEAN = 0.0
NOISE_STD = 25.0

gaussian_noisy = {}
for label, info in MODALITIES.items():
    base = info["gray"].astype(np.float32)
    noise = np.random.normal(loc=NOISE_MEAN, scale=NOISE_STD, size=base.shape)
    gaussian_noisy[label] = np.clip(base + noise, 0, 255).astype(np.uint8)

print(f"Bruit gaussien appliqué (moyenne {NOISE_MEAN}, écart-type {NOISE_STD}) sur {len(gaussian_noisy)} modalité(s).")

for label, info in MODALITIES.items():
    display = info["rgb"] if info["color"] else info["gray"]
    cmap = None if info["color"] else "gray"
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].imshow(display, cmap=cmap)
    axes[0].set_title(f"{label} originale")
    axes[0].axis("off")
    axes[1].imshow(gaussian_noisy[label], cmap="gray")
    axes[1].set_title(f"{label} + bruit gaussien (std={NOISE_STD})")
    axes[1].axis("off")
    plt.tight_layout()
    save_fig(fig, f"ex1_bruit_gaussien_{label}.png")
    plt.show()

# -------- ORIGINAL NOTEBOOK CELL 6 --------
# ============================================================
# Exercice 1 — 2) Bruit poivre et sel
# ============================================================
def add_salt_and_pepper(image, salt_prob=0.005, pepper_prob=0.005):
    # Pixels noirs (poivre) et pixels blancs (sel) placés aléatoirement.
    out = image.copy()
    pepper_mask = np.random.random(out.shape) < pepper_prob
    out[pepper_mask] = 0
    salt_mask = np.random.random(out.shape) < salt_prob
    out[salt_mask] = 255
    return out


SALT_PROB = 0.005
PEPPER_PROB = 0.005

sp_noisy = {}
for label, info in MODALITIES.items():
    sp_noisy[label] = add_salt_and_pepper(info["gray"], SALT_PROB, PEPPER_PROB)

print(f"Bruit poivre et sel appliqué (probabilité {SALT_PROB} par pixel) sur {len(sp_noisy)} modalité(s).")

for label, info in MODALITIES.items():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].imshow(info["gray"], cmap="gray")
    axes[0].set_title(f"{label} originale")
    axes[0].axis("off")
    axes[1].imshow(sp_noisy[label], cmap="gray")
    axes[1].set_title(f"{label} + poivre et sel")
    axes[1].axis("off")
    plt.tight_layout()
    save_fig(fig, f"ex1_poivre_sel_{label}.png")
    plt.show()

# -------- ORIGINAL NOTEBOOK CELL 8 --------
# ============================================================
# Exercice 1 — 3) Filtres moyenneur, médian et gaussien
# ============================================================
def filtre_moyenneur(image, ksize=5):
    # Filtre moyenneur (moyenne locale ksize x ksize) via OpenCV.
    return cv2.blur(image, (ksize, ksize))


def filtre_median(image, ksize=5):
    # Filtre médian : skimage si disponible, sinon OpenCV.
    if HAS_SKIMAGE:
        try:
            return filters.median(image, footprint=np.ones((ksize, ksize)))
        except TypeError:
            print("AVERTISSEMENT : skimage.median incompatible, repli sur cv2.medianBlur.")
    return cv2.medianBlur(image, ksize)


def filtre_gaussien(image, ksize=5, sigma=1.0):
    # Filtre gaussien : skimage si disponible, sinon OpenCV.
    if HAS_SKIMAGE:
        try:
            # skimage conserve l'échelle de l'entrée (0-255 pour uint8)
            smoothed = filters.gaussian(image, sigma=sigma)
            return np.clip(smoothed, 0, 255).astype(np.uint8)
        except TypeError:
            print("AVERTISSEMENT : skimage.gaussian incompatible, repli sur cv2.GaussianBlur.")
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def psnr(original, restored):
    # Rapport signal sur bruit de crête (dB) entre deux images.
    mse = np.mean((original.astype(np.float32) - restored.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


# Application des trois filtres sur les deux versions bruitées (gaussien / poivre et sel)
filtered_results = {}
for label, info in MODALITIES.items():
    filtered_results[label] = {}
    for noise_label in ["gaussien", "poivre_sel"]:
        noisy_img = gaussian_noisy[label] if noise_label == "gaussien" else sp_noisy[label]
        filtered_results[label][noise_label] = {
            "moyenneur": filtre_moyenneur(noisy_img),
            "median": filtre_median(noisy_img),
            "gaussien": filtre_gaussien(noisy_img),
        }

# Figures de comparaison : 2 lignes (types de bruit) x 4 colonnes (bruité + 3 filtres)
row_titles = ["Bruit gaussien", "Bruit poivre et sel"]
col_titles = ["Image bruitée", "Filtre moyenneur (5x5)", "Filtre médian (5x5)", "Filtre gaussien (sigma=1)"]

for label, info in MODALITIES.items():
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for row, noise_label in enumerate(["gaussien", "poivre_sel"]):
        base_noisy = gaussian_noisy[label] if noise_label == "gaussien" else sp_noisy[label]
        images = [
            base_noisy,
            filtered_results[label][noise_label]["moyenneur"],
            filtered_results[label][noise_label]["median"],
            filtered_results[label][noise_label]["gaussien"],
        ]
        for col, (img, title) in enumerate(zip(images, col_titles)):
            axes[row, col].imshow(img, cmap="gray")
            axes[row, col].set_title(title, fontsize=9)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(row_titles[row], fontsize=9)
    fig.suptitle(f"{label} : débruitage comparé", fontsize=12)
    plt.tight_layout()
    save_fig(fig, f"ex1_filtres_{label}.png")
    plt.show()

# Tableau PSNR (image filtrée vs originale) : plus élevé = meilleure restauration
print("PSNR (dB) — image filtrée vs image originale :")
header = "Modalité".ljust(14) + "Bruit".ljust(12) + "Moyenneur".ljust(10) + "Médian".ljust(10) + "Gaussien"
print(header)
print("-" * len(header))
for label in MODALITIES:
    for noise_label in ["gaussien", "poivre_sel"]:
        original = MODALITIES[label]["gray"]
        res = filtered_results[label][noise_label]
        print(
            label.ljust(14)
            + noise_label.ljust(12)
            + f"{psnr(original, res['moyenneur']):7.2f}  "
            + f"{psnr(original, res['median']):7.2f}  "
            + f"{psnr(original, res['gaussien']):7.2f}"
        )

print(
    "Observation pédagogique :\n"
    "- Bruit poivre et sel : le filtre médian est nettement le plus efficace (PSNR\n"
    "  le plus élevé) : il élimine les pixels isolés noirs/blancs sans lisser les\n"
    "  contours, car la médiane du voisinage ignore les valeurs extrêmes isolées.\n"
    "- Bruit gaussien : le filtre gaussien (ou le moyenneur) est généralement\n"
    "  meilleur que le médian, car le bruit gaussien est diffus sur tout le voisinage.\n"
    "- Le moyenneur et le gaussien lissent les textures et peuvent estomper les\n"
    "  contours des tissus — critique en IRM où les frontières substance grise /\n"
    "  substance blanche doivent rester nettes.\n"
    "- Efficacité du médian selon la modalité : en endoscopie (tissu très texturé,\n"
    "  reflets) il préserve mieux les détails fins ; en IRM il préserve mieux les\n"
    "  contours que le moyenneur, ce qui est essentiel pour la segmentation.\n"
    "- Conclusion : le filtre médian est le plus adapté au bruit impulsionnel\n"
    "  (poivre et sel), quelle que soit la modalité ; pour un bruit gaussien pur,\n"
    "  un filtre gaussien bien réglé est plus approprié."
)

# -------- ORIGINAL NOTEBOOK CELL 11 --------
# ============================================================
# Exercice 2 — 1) Histogrammes des niveaux de gris + seuil manuel
# ============================================================
def compute_histogram(gray_img):
    # Histogramme sur 256 niveaux de gris (valeurs 0 à 255).
    hist, _ = np.histogram(gray_img.ravel(), bins=256, range=(0, 256))
    return hist


histograms = {}
for label, info in MODALITIES.items():
    histograms[label] = compute_histogram(info["gray"])

# Seuils manuels choisis en observant les histogrammes ci-dessous (ajustables).
# NB : pas de radiographie dans ce TP — les 3 modalités sont endoscopie + 2 IRM
# (irm_t1 et irm_t2) ; aucun seuil ne concerne une image de rayons X.
MANUAL_THRESHOLDS = {
    "endoscopie": 110,   # sépare les tissus clairs des zones sombres/ombres et reflets
    "irm_t1": 105,       # sépare la substance blanche (claire en T1) du reste / du fond
    "irm_t2": 150,       # sépare le LCR (très clair en T2) des tissus cérébraux
}

# Affichage : image + histogramme (avec le seuil manuel) + binarisation manuelle
for label, info in MODALITIES.items():
    display = info["rgb"] if info["color"] else info["gray"]
    cmap = None if info["color"] else "gray"
    threshold = MANUAL_THRESHOLDS.get(label)
    binary = (info["gray"] > threshold).astype(np.uint8) * 255 if threshold is not None else None

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    axes[0].imshow(display, cmap=cmap)
    axes[0].set_title(f"{label} originale")
    axes[0].axis("off")
    axes[1].bar(np.arange(256), histograms[label], width=1.0, color="steelblue")
    axes[1].set_title(f"Histogramme — {label}")
    axes[1].set_xlabel("Intensité")
    axes[1].set_ylabel("Nombre de pixels")
    if threshold is not None:
        axes[1].axvline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"T manuel = {threshold}")
        axes[1].legend(fontsize=8)
    if binary is not None:
        axes[2].imshow(binary, cmap="gray")
        axes[2].set_title(f"{label} binaire (T manuel = {threshold})")
        axes[2].axis("off")
    plt.tight_layout()
    save_fig(fig, f"ex2_histogramme_{label}.png")
    plt.show()

print("Seuil manuel retenu par modalité disponible (ajustable) :")
for label, threshold in MANUAL_THRESHOLDS.items():
    if label in MODALITIES:
        print(f"  - {label}: T = {threshold}")

# -------- ORIGINAL NOTEBOOK CELL 13 --------
# ============================================================
# Exercice 2 — 2) Seuillage automatique d'Otsu
# ============================================================
otsu_results = {}
for label, info in MODALITIES.items():
    if info["gray"] is None:
        continue
    otsu_threshold, binary = cv2.threshold(info["gray"], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_results[label] = {"seuil": float(otsu_threshold), "binaire": binary}

print("Seuils : manuel vs Otsu")
print("-" * 40)
for label in MODALITIES:
    manual = MANUAL_THRESHOLDS.get(label)
    otsu_value = otsu_results[label]["seuil"]
    manual_str = "—" if manual is None else str(manual)
    print(f"{label.ljust(14)} manuel = {manual_str.ljust(4)} | Otsu = {otsu_value:.1f}")

for label, info in MODALITIES.items():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].imshow(info["gray"], cmap="gray")
    axes[0].set_title(f"{label} originale")
    axes[0].axis("off")
    axes[1].imshow(otsu_results[label]["binaire"], cmap="gray")
    axes[1].set_title(f"{label} binaire (Otsu, T={otsu_results[label]['seuil']:.1f})")
    axes[1].axis("off")
    plt.tight_layout()
    save_fig(fig, f"ex2_otsu_{label}.png")
    plt.show()

# -------- ORIGINAL NOTEBOOK CELL 15 --------
# ============================================================
# Exercice 2 — 3) Comparaison multi-modalités (explication)
# ============================================================
# Figure de synthèse : histogrammes normalisés des 3 modalités superposés.
fig, ax = plt.subplots(figsize=(9, 4))
for label in MODALITIES:
    hist = histograms[label].astype(np.float32)
    hist = hist / hist.sum()  # normalisation pour comparer des tailles d'image différentes
    ax.plot(np.arange(256), hist, label=label)
    manual = MANUAL_THRESHOLDS.get(label)
    if manual is not None:
        ax.axvline(manual, linestyle="--", linewidth=1.0, color="gray", alpha=0.6)
ax.set_xlabel("Intensité")
ax.set_ylabel("Proportion de pixels (normalisée)")
ax.set_title("Histogrammes normalisés — comparaison des 3 modalités (seuils manuels en tiretés)")
ax.legend()
plt.tight_layout()
save_fig(fig, "ex2_comparaison_histogrammes.png")
plt.show()

# NB : la radiographie est ABSENTE de ce TP (aucun fichier de rayons X fourni).
# L'analyse porte donc uniquement sur l'endoscopie et les deux pondérations d'IRM.
print(
    "Comparaison multi-modalités : pourquoi le seuillage d'Otsu se comporte-t-il différemment ?\n"
    "\n"
    "Rappel important : dans ce TP, la modalité 'radiographie' est ABSENTE du jeu de\n"
    "données (aucun fichier de rayons X n'est fourni). Les 3 images traitées sont :\n"
    "  - endoscopie : frame442.png (cordes vocales, couleur, texturée)\n"
    "  - irm_t1 : Tr-pi_1.jpg (IRM cérébrale, coupe coronale pondérée T1)\n"
    "  - irm_t2 : Te-no_264.jpg (IRM cérébrale, coupe axiale pondérée T2)\n"
    "Aucune de ces images n'est une radiographie ; l'analyse porte donc uniquement\n"
    "sur l'endoscopie et sur les deux pondérations d'IRM.\n"
    "\n"
    "1) IRM cérébrale (bimodalité relativement nette) :\n"
    "   En pondération T1, la substance blanche apparaît claire, la substance grise\n"
    "   intermédiaire et le LCR sombre ; en pondération T2, le LCR devient très clair\n"
    "   et la substance blanche plus sombre que la grise. Dans les deux cas, on peut\n"
    "   distinguer une structure bimodale (tissus vs fond/LCR, ou LCR vs tissus) avec\n"
    "   une vallée plus ou moins marquée. Otsu maximise la variance inter-classe et\n"
    "   place le seuil dans cette vallée : la séparation anatomique (substance\n"
    "   blanche / substance grise / LCR) est stable et reproductible.\n"
    "\n"
    "2) Endoscopie (reflets spéculaires et ombres) :\n"
    "   L'image des cordes vocales contient des reflets spéculaires (zones très\n"
    "   brillantes, saturées), des ombres portées (zones très sombres) et des textures\n"
    "   de tissu. L'histogramme devient multimodal et étalé (pas de bimodalité claire) :\n"
    "   Otsu force une séparation en deux classes sur une distribution qui n'est pas\n"
    "   bimodale. Le seuil tombe alors dans une zone intermédiaire arbitraire, sensible\n"
    "   aux variations d'éclairage, et la binarisation produit des artefacts (trous,\n"
    "   régions parasites, reflets conservés).\n"
    "\n"
    "3) IRM T1 vs IRM T2 :\n"
    "   En T1 (irm_t1), l'image est dominée par un fond noir : la bimodalité la plus\n"
    "   forte oppose le fond au cerveau, et le seuil Otsu (~55) sépare donc surtout\n"
    "   cerveau et fond ; la vallée substance grise / substance blanche est plus\n"
    "   subtile et le seuil manuel (~105) permet de la cibler. En T2 (irm_t2), le LCR\n"
    "   est très clair : Otsu sépare les zones riches en LCR (ventricules, espaces\n"
    "   sous-arachnoïdiens) des tissus cérébraux ; le seuil manuel (~150) isole plus\n"
    "   strictement les zones de LCR pur. L'écart entre seuils manuel et Otsu n'est\n"
    "   donc pas un défaut : Otsu maximise globalement la variance inter-classe, alors\n"
    "   qu'un seuil manuel peut cibler une vallée anatomique précise — le bon choix\n"
    "   dépend de l'objectif de segmentation visé.\n"
    "\n"
    "Conclusion : Otsu est très efficace sur une IRM (bimodalité tissus/fond ou\n"
    "LCR/tissus), mais sur une endoscopie avec reflets/ombres (distribution non\n"
    "bimodale) il faut le combiner avec un prétraitement (filtrage, égalisation\n"
    "d'histogramme, morphologie) ou choisir un seuil manuel guidé par l'histogramme."
)
