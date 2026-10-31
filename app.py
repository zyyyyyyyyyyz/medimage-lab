"""MedImage Lab — interface Streamlit des TP1 à TP5 validés."""
from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
import hashlib
import importlib.util
import json
import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from core.algorithms import clean_mask, ensure_gray, overlay, preprocess, scores, timed_segmentation
from core.io_utils import load_rgb, mask_png_bytes, results_csv_bytes
from core.tp2 import manual_threshold_tp2, otsu_tp2
from core.tp3 import TP3_METHODS, dice_tp3, iou_tp3, prepare_tp3_reference, segment_tp3


ROOT = Path(__file__).resolve().parent
EXPECTED_MODEL = ROOT / "models" / "unet_breast_batch16_epochs50.hdf5"
TENSORFLOW_AVAILABLE = importlib.util.find_spec("tensorflow") is not None
IMAGE_TYPES = ["png", "jpg", "jpeg", "tif", "tiff"]
FILTERS = ["Aucun", "Médian", "Gaussien", "Bilatéral", "CLAHE", "Égalisation histogramme"]
FILTER_SOURCE = {
    "Aucun": "Identité",
    "Médian": "TP2 · cellule 7 · disque de rayon 3",
    "Gaussien": "TP2 · cellule 7 · sigma 1",
    "Bilatéral": "Extension web sans équivalent TP direct",
    "CLAHE": "Réutilisation adaptée du prétraitement FIRE",
    "Égalisation histogramme": "Extension web sans équivalent TP direct",
}
METHOD_SOURCE = {
    "Otsu": "TP2 · cellule 12 · exact",
    "Seuil manuel": "TP2 · cellule 10 · seuil original 50",
    "K-means": "TP3 · cellule 14 · exact",
    "Mean Shift": "TP3 · cellule 9 · exact",
    "Chan–Vese": "TP3 · cellule 24 · exact",
    "Watershed": "Extension web · non équivalente au Watershed guidé du TP2",
    "U-Net BUSI": "TP4 · architecture cellule 7 · checkpoint validé",
}
METHOD_HELP = {
    "Otsu": "Seuil global automatique adapté aux histogrammes séparant deux populations.",
    "Seuil manuel": "Binarisation contrôlée par un seuil d'intensité.",
    "K-means": "Trois classes RGB; conservation de la classe moyenne la plus sombre.",
    "Mean Shift": "Clustering non paramétrique sur une image limitée à 256 px de côté.",
    "Chan–Vese": "Contour actif fondé sur les régions, sans gradient de bord explicite.",
    "Watershed": "Extension pédagogique utilisant des marqueurs automatiques de percentiles.",
    "U-Net BUSI": "U-Net TensorFlow du TP4 réservé aux échographies mammaires BUSI.",
}
NAVIGATION = [
    "TP1 · Manipulation d'image",
    "TP2 · Prétraitement",
    "TP3 · Segmentation conventionnelle",
    "TP4 · U-Net BUSI",
    "TP5 · Recalage FIRE",
    "Guide scientifique",
]


st.set_page_config(page_title="MedImage Lab", page_icon="🩻", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
<style>
  .block-container {max-width: 1440px; padding-top: 1.4rem; padding-bottom: 3rem;}
  .app-header {background:#ffffff; border:1px solid #d7e3ea; border-radius:14px;
    padding:1.25rem 1.4rem; margin-bottom:1.2rem; box-shadow:0 4px 16px rgba(23,43,58,.05);}
  .app-header h1 {color:#123247; font-size:1.75rem; line-height:1.2; margin:.12rem 0 .35rem;}
  .app-header p {color:#496575; margin:0; max-width:900px; line-height:1.55;}
  .eyebrow {color:#0b6e8e; font-size:.76rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase;}
  .step {display:flex; gap:.75rem; align-items:flex-start; margin:.35rem 0 .8rem;}
  .step-number {display:inline-flex; align-items:center; justify-content:center; min-width:1.8rem;
    height:1.8rem; border-radius:50%; background:#dff1f6; color:#075c78; font-weight:750;}
  .step-title {color:#17384c; font-size:1.08rem; font-weight:750; line-height:1.3;}
  .step-copy {color:#5a7180; font-size:.9rem; margin-top:.12rem; line-height:1.45;}
  .result-title {color:#17384c; font-weight:750; font-size:1.02rem; margin-bottom:.12rem;}
  .source-note {color:#5a7180; font-size:.82rem; margin-bottom:.65rem;}
  .sidebar-brand {font-weight:800; font-size:1.35rem; color:#123247; margin-bottom:.2rem;}
  .sidebar-copy {color:#526c7b; font-size:.86rem; line-height:1.45; margin-bottom:.65rem;}
  .footer {text-align:center; color:#607987; font-size:.8rem; padding-top:2.2rem;}
  @media (max-width: 900px) {.block-container {padding-left:1rem; padding-right:1rem;}}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def _load_local_model(path: str, modified_ns: int):
    del modified_ns
    from core.unet_tp4 import load_busi_model
    return load_busi_model(path)


@st.cache_resource(show_spinner=False)
def _load_uploaded_model(payload: bytes, suffix: str):
    from core.unet_tp4 import load_busi_model
    temp_path = None
    try:
        with NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(payload)
            temp_path = handle.name
        return load_busi_model(temp_path)
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


def _configured_checkpoint():
    configured = os.environ.get("MEDIMAGE_UNET_CHECKPOINT", "").strip()
    candidates = [EXPECTED_MODEL]
    if configured:
        candidates.insert(0, Path(configured).expanduser())
    return next((path.resolve() for path in candidates if path.is_file()), None)


def _signature(*items) -> str:
    digest = hashlib.sha256()
    for item in items:
        if item is None:
            digest.update(b"<none>")
        elif isinstance(item, (bytes, bytearray)):
            digest.update(bytes(item))
        elif hasattr(item, "getvalue"):
            digest.update(item.getvalue())
        else:
            digest.update(str(item).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _save_result(key: str, signature: str, value):
    st.session_state[key] = {"signature": signature, "value": value}


def _saved_result(key: str, signature: str):
    entry = st.session_state.get(key)
    return entry["value"] if entry and entry.get("signature") == signature else None


def _rgb_png_bytes(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not success:
        raise ValueError("Encodage PNG impossible")
    return encoded.tobytes()


def _gray_png_bytes(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", np.asarray(image, dtype=np.uint8))
    if not success:
        raise ValueError("Encodage PNG impossible")
    return encoded.tobytes()


def _header(kicker: str, title: str, description: str):
    st.markdown(
        f'<div class="app-header"><div class="eyebrow">{kicker}</div><h1>{title}</h1><p>{description}</p></div>',
        unsafe_allow_html=True,
    )


def _step(number: int, title: str, copy: str):
    st.markdown(
        f'<div class="step"><span class="step-number">{number}</span><div><div class="step-title">{title}</div>'
        f'<div class="step-copy">{copy}</div></div></div>',
        unsafe_allow_html=True,
    )


def _load_input(upload, demo: bool = False):
    source = ROOT / "assets" / "bagls_test_2.png" if demo else upload
    return None if source is None else load_rgb(source)


def _load_reference(upload, demo: bool, expected_shape):
    source = upload if upload is not None else (ROOT / "assets" / "bagls_test_2_mask.png" if demo else None)
    if source is None:
        return None, None
    reference = ensure_gray(load_rgb(source))
    if reference.shape != expected_shape:
        return None, f"Masque non aligné: {reference.shape[::-1]} au lieu de {expected_shape[::-1]}."
    return reference, None


def _image_summary(image: np.ndarray, reference=None):
    columns = st.columns(3)
    columns[0].metric("Dimensions", f"{image.shape[1]} × {image.shape[0]}")
    columns[1].metric("Format interne", str(image.dtype))
    columns[2].metric("Vérité terrain", "Alignée" if reference is not None else "Absente")


def _render_histogram(gray: np.ndarray, label="Pixels"):
    frequencies, edges = np.histogram(gray, bins=64, range=(0, 256))
    table = pd.DataFrame({label: frequencies}, index=edges[:-1].astype(int))
    table.index.name = "Intensité"
    st.bar_chart(table, color="#0B6E8E", height=260)


def _result_card(name, record, metrics, probability=None):
    mask = record["mask"]
    with st.container(border=True):
        st.markdown(f'<div class="result-title">{name}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="source-note">{record["source"]}</div>', unsafe_allow_html=True)
        if probability is not None:
            views = st.columns(2)
            views[0].image(probability, caption="Carte de probabilité", clamp=True, use_container_width=True)
            views[1].image(mask, caption="Masque binaire · > 0,5", clamp=True, use_container_width=True)
        else:
            st.image(mask, caption="Masque binaire", clamp=True, use_container_width=True)
        st.image(overlay(record["image"], mask), caption="Superposition", use_container_width=True)
        metric_columns = st.columns(3 if metrics else 1)
        metric_columns[0].metric("Temps", f'{record["seconds"]:.3f} s')
        if metrics:
            metric_columns[1].metric("Dice", f'{metrics["Dice"]:.3f}')
            metric_columns[2].metric("IoU", f'{metrics["IoU"]:.3f}')
        st.download_button(
            "Télécharger le masque PNG", mask_png_bytes(mask),
            file_name=name.lower().replace(" ", "_").replace("–", "-") + "_mask.png",
            mime="image/png", key=f"download-{name}-{_signature(mask)[:10]}", use_container_width=True,
        )


def render_tp1():
    _header("TP1 · Introduction", "Manipulation et lecture d'une image médicale",
            "Importez une image, vérifiez ses propriétés, comparez les représentations et explorez son histogramme.")
    _step(1, "Importer une image", "Utilisez l'exemple BAGLS ou déposez une image PNG, JPEG ou TIFF.")
    with st.container(border=True):
        demo = st.toggle("Utiliser l'exemple BAGLS", value=True, key="tp1-demo")
        upload = None if demo else st.file_uploader("Image médicale", type=IMAGE_TYPES, key="tp1-image")
        st.caption("Limites: 25 Mo et 12 mégapixels.")
    try:
        image = _load_input(upload, demo)
    except ValueError as exc:
        st.error(f"Image non chargée: {exc}")
        return
    if image is None:
        st.info("Ajoutez une image pour afficher les outils du TP1.")
        return
    gray = ensure_gray(image)
    _image_summary(image)
    _step(2, "Comparer les représentations", "L'image originale reste visible à côté de sa représentation en niveaux de gris.")
    with st.container(border=True):
        columns = st.columns(2)
        columns[0].image(image, caption="Image originale RGB", use_container_width=True)
        columns[1].image(gray, caption="Niveaux de gris", clamp=True, use_container_width=True)
    _step(3, "Lire l'histogramme", "L'axe horizontal représente les intensités; l'axe vertical, le nombre de pixels.")
    with st.container(border=True):
        _render_histogram(gray)
    _step(4, "Exporter", "Téléchargez la représentation utile pour votre compte rendu.")
    actions = st.columns(2)
    actions[0].download_button("Télécharger l'image originale", _rgb_png_bytes(image),
                               file_name="tp1_image_originale.png", mime="image/png", use_container_width=True)
    actions[1].download_button("Télécharger les niveaux de gris", _gray_png_bytes(gray),
                               file_name="tp1_niveaux_de_gris.png", mime="image/png", use_container_width=True)


def render_tp2():
    _header("TP2 · Prétraitement", "Filtrage et seuillage d'images médicales",
            "Choisissez un prétraitement, appliquez le seuillage manuel ou Otsu, puis comparez les résultats.")
    _step(1, "Charger les données", "Le masque est facultatif; il active Dice et IoU lorsqu'il est parfaitement aligné.")
    with st.container(border=True):
        demo = st.toggle("Utiliser l'exemple BAGLS avec son masque", value=True, key="tp2-demo")
        upload = None if demo else st.file_uploader("Image médicale", type=IMAGE_TYPES, key="tp2-image")
        truth = st.file_uploader("Masque de référence · optionnel", type=IMAGE_TYPES, key="tp2-truth")
    try:
        image = _load_input(upload, demo)
    except ValueError as exc:
        st.error(f"Image non chargée: {exc}")
        return
    if image is None:
        st.info("Ajoutez une image pour configurer le TP2.")
        return
    gray = ensure_gray(image)
    try:
        reference, reference_error = _load_reference(truth, demo, gray.shape)
    except ValueError as exc:
        reference, reference_error = None, str(exc)
    if reference_error:
        st.warning(reference_error + " Les métriques sont désactivées.")
    _image_summary(image, reference)
    _step(2, "Choisir le prétraitement", "Les paramètres d'origine sont conservés; les extensions sont explicitement signalées.")
    with st.container(border=True):
        filtering = st.selectbox("Filtre ou contraste", FILTERS, key="tp2-filter")
        st.caption(FILTER_SOURCE[filtering])
        threshold_columns = st.columns(2)
        use_manual = threshold_columns[0].checkbox("Seuil manuel", value=True, key="tp2-manual")
        use_otsu = threshold_columns[1].checkbox("Otsu", value=True, key="tp2-otsu")
        with st.expander("Paramètres avancés", expanded=False):
            manual_threshold = st.slider("Seuil manuel", 0, 255, 50, key="tp2-threshold")
            st.caption("Le TP2 utilise 50. Toute autre valeur est une adaptation utilisateur explicite.")
    signature = _signature("tp2", "demo" if demo else upload, truth, filtering, use_manual, use_otsu, manual_threshold)
    _step(3, "Lancer l'analyse", "Le traitement ne s'exécute qu'avec ce bouton; les autres interactions conservent le résultat.")
    if st.button("Lancer le TP2", type="primary", use_container_width=True, key="tp2-run"):
        if not (use_manual or use_otsu):
            st.warning("Sélectionnez au moins une méthode de seuillage.")
        else:
            try:
                processed = preprocess(gray, filtering, 3)
                result = {"processed": processed, "masks": {}}
                if use_manual:
                    mask, seconds = manual_threshold_tp2(processed, manual_threshold)
                    result["masks"]["Seuil manuel"] = (mask, seconds)
                if use_otsu:
                    mask, seconds, otsu_value = otsu_tp2(processed)
                    result["masks"]["Otsu"] = (mask, seconds)
                    result["otsu_value"] = otsu_value
                _save_result("tp2-result", signature, result)
            except (ValueError, cv2.error) as exc:
                st.error(f"Traitement impossible: {exc}")
    result = _saved_result("tp2-result", signature)
    if result is None:
        columns = st.columns(2)
        columns[0].image(image, caption="Image originale", use_container_width=True)
        if reference is not None:
            columns[1].image(reference, caption="Masque de référence", clamp=True, use_container_width=True)
        return
    _step(4, "Inspecter les résultats", "Comparez l'original, le prétraitement et les segmentations binaires.")
    views = st.columns(3)
    views[0].image(image, caption="Image originale", use_container_width=True)
    views[1].image(result["processed"], caption=f"Prétraitement · {filtering}", clamp=True, use_container_width=True)
    if reference is not None:
        views[2].image(reference, caption="Référence alignée", clamp=True, use_container_width=True)
    else:
        views[2].info("Aucun masque aligné: Dice et IoU ne seront pas affichés.")
    _render_histogram(result["processed"], "Pixels prétraités")
    records = []
    columns = st.columns(2)
    for column, (name, (mask, seconds)) in zip(columns, result["masks"].items()):
        metrics = scores(mask, reference) if reference is not None else {}
        records.append({"Méthode": name, "Durée (s)": round(seconds, 4), **metrics})
        with column:
            _result_card(name, {"mask": mask, "seconds": seconds, "image": image, "source": METHOD_SOURCE[name]}, metrics)
    if "otsu_value" in result:
        st.caption(f'Seuil calculé par Otsu: {result["otsu_value"]:.1f}')
    downloads = st.columns(2)
    downloads[0].download_button("Télécharger l'image prétraitée", _gray_png_bytes(result["processed"]),
                                 file_name="tp2_image_pretraitee.png", mime="image/png", use_container_width=True)
    downloads[1].download_button("Télécharger les métriques CSV", results_csv_bytes(records),
                                 file_name="tp2_resultats.csv", mime="text/csv", use_container_width=True)
    with st.expander("Traçabilité de l'analyse", expanded=False):
        st.json({
            "tp": "TP2",
            "image": {"largeur": image.shape[1], "hauteur": image.shape[0], "dtype": str(image.dtype)},
            "pretraitement": {"methode": filtering, "source": FILTER_SOURCE[filtering]},
            "seuillages": list(result["masks"]),
            "seuil_manuel": manual_threshold if use_manual else None,
            "verite_terrain_alignee": reference is not None,
        })


def _run_tp3(image, processed, reference, methods, foreground, threshold, min_area, fill_holes):
    results, failures = {}, []
    for name in methods:
        try:
            if name in TP3_METHODS:
                result_image, mask, seconds = segment_tp3(image, name)
                result_reference = (prepare_tp3_reference(reference, (image.shape[1], image.shape[0]), mask.shape)
                                    if reference is not None else None)
            elif name == "Seuil manuel":
                mask, seconds = manual_threshold_tp2(processed, threshold)
                result_image, result_reference = image, reference
            elif name == "Otsu":
                mask, seconds, _ = otsu_tp2(processed)
                result_image, result_reference = image, reference
            else:
                mask, seconds = timed_segmentation(processed, name, foreground=foreground, clusters=3, threshold=threshold)
                mask = clean_mask(mask, min_area=min_area, fill=fill_holes)
                result_image, result_reference = image, reference
            results[name] = {"mask": mask, "seconds": seconds, "image": result_image,
                             "reference": result_reference, "source": METHOD_SOURCE[name]}
        except (ValueError, cv2.error, OSError) as exc:
            failures.append((name, str(exc)))
        except Exception as exc:
            failures.append((name, f"{type(exc).__name__}: {exc}"))
    return results, failures


def render_tp3():
    _header("TP3 · BAGLS", "Segmentation conventionnelle comparative",
            "Sélectionnez plusieurs méthodes, lancez-les ensemble et comparez masques, superpositions, temps et métriques.")
    _step(1, "Charger les données", "L'exemple BAGLS inclut une vérité terrain alignée pour la démonstration.")
    with st.container(border=True):
        demo = st.toggle("Utiliser l'exemple BAGLS avec son masque", value=True, key="tp3-demo")
        upload = None if demo else st.file_uploader("Image médicale", type=IMAGE_TYPES, key="tp3-image")
        truth = st.file_uploader("Masque de référence · optionnel", type=IMAGE_TYPES, key="tp3-truth")
    try:
        image = _load_input(upload, demo)
    except ValueError as exc:
        st.error(f"Image non chargée: {exc}")
        return
    if image is None:
        st.info("Ajoutez une image pour configurer la segmentation.")
        return
    gray = ensure_gray(image)
    try:
        reference, reference_error = _load_reference(truth, demo, gray.shape)
    except ValueError as exc:
        reference, reference_error = None, str(exc)
    if reference_error:
        st.warning(reference_error + " Les métriques sont désactivées.")
    _image_summary(image, reference)
    _step(2, "Sélectionner les méthodes", "Les méthodes sont présentées sans multiselect compact; choisissez celles à comparer.")
    methods = []
    choices = ["Otsu", "Seuil manuel", "K-means", "Mean Shift", "Chan–Vese", "Watershed"]
    defaults = {"Otsu", "K-means", "Chan–Vese"}
    with st.container(border=True):
        for row_start in range(0, len(choices), 3):
            columns = st.columns(3)
            for column, name in zip(columns, choices[row_start:row_start + 3]):
                if column.checkbox(name, value=name in defaults,
                                   key="tp3-method-" + name.lower().replace(" ", "-").replace("–", "-"),
                                   help=METHOD_HELP[name]):
                    methods.append(name)
        with st.expander("Paramètres avancés et extensions", expanded=False):
            filtering = st.selectbox("Prétraitement exploratoire", FILTERS, key="tp3-filter")
            st.caption("K-means, Mean Shift et Chan–Vese utilisent toujours l'image RGB originale selon le TP3. "
                       "Ce prétraitement concerne uniquement les méthodes de seuil et Watershed.")
            foreground = st.radio("Région Watershed", ["Sombre", "Claire"], horizontal=True, key="tp3-foreground")
            threshold = st.slider("Seuil manuel", 0, 255, 50, key="tp3-threshold")
            min_area = st.slider("Surface minimale · extensions", 0, 1000, 0, step=10, key="tp3-area")
            fill_holes = st.checkbox("Remplir les trous · extensions", value=False, key="tp3-fill")
            st.warning("Watershed est une extension web et non le Watershed guidé du TP2.")
    signature = _signature("tp3", "demo" if demo else upload, truth, methods, filtering,
                           foreground, threshold, min_area, fill_holes)
    _step(3, "Lancer la comparaison", "Les méthodes coûteuses ne sont exécutées qu'avec ce bouton.")
    if st.button("Lancer le TP3", type="primary", use_container_width=True, key="tp3-run"):
        if not methods:
            st.warning("Sélectionnez au moins une méthode.")
        else:
            try:
                processed = preprocess(gray, filtering, 3)
                with st.spinner("Segmentation en cours…"):
                    results, failures = _run_tp3(image, processed, reference, methods, foreground,
                                                 threshold, min_area, fill_holes)
                _save_result("tp3-result", signature,
                             {"results": results, "failures": failures, "processed": processed})
            except (ValueError, cv2.error) as exc:
                st.error(f"Analyse impossible: {exc}")
    analysis = _saved_result("tp3-result", signature)
    if analysis is None:
        preview = st.columns(2)
        preview[0].image(image, caption="Image originale", use_container_width=True)
        if reference is not None:
            preview[1].image(reference, caption="Masque de référence", clamp=True, use_container_width=True)
        return
    _step(4, "Comparer les résultats", "Chaque carte conserve la forme de sortie et le protocole de sa méthode.")
    for name, message in analysis["failures"]:
        st.warning(f"{name} non exécuté: {message}")
    records, names = [], list(analysis["results"])
    for index in range(0, len(names), 2):
        columns = st.columns(2)
        for column, name in zip(columns, names[index:index + 2]):
            record = analysis["results"][name]
            result_reference = record["reference"]
            metrics = {}
            if result_reference is not None:
                metrics = ({"Dice": dice_tp3(record["mask"], result_reference),
                            "IoU": iou_tp3(record["mask"], result_reference)}
                           if name in TP3_METHODS else scores(record["mask"], result_reference))
            records.append({"Méthode": name, "Durée (s)": round(record["seconds"], 4), **metrics})
            with column:
                _result_card(name, record, metrics)
    st.dataframe(pd.DataFrame.from_records(records), use_container_width=True, hide_index=True)
    st.download_button("Télécharger la comparaison CSV", results_csv_bytes(records),
                       file_name="tp3_comparaison.csv", mime="text/csv", use_container_width=True)
    with st.expander("Traçabilité de l'analyse", expanded=False):
        st.json({
            "tp": "TP3",
            "image": {"largeur": image.shape[1], "hauteur": image.shape[0], "dtype": str(image.dtype)},
            "methodes_demandees": methods,
            "methodes_reussies": names,
            "sources": {name: METHOD_SOURCE[name] for name in names},
            "pretraitement_extensions": filtering,
            "verite_terrain_alignee": reference is not None,
        })


def render_tp4():
    _header("TP4 · BUSI", "Segmentation par le U-Net entraîné",
            "Chargez une échographie mammaire BUSI. Le modèle conserve le redimensionnement, la normalisation axis=1 et le seuil strict du TP4.")
    checkpoint = _configured_checkpoint()
    _step(1, "Vérifier le modèle", "L'inférence n'est disponible qu'avec TensorFlow et un checkpoint compatible.")
    status_columns = st.columns(3)
    status_columns[0].metric("TensorFlow", "Disponible" if TENSORFLOW_AVAILABLE else "Indisponible")
    status_columns[1].metric("Checkpoint", "Détecté" if checkpoint else "Absent")
    status_columns[2].metric("Entrée", "256 × 256 × 1")
    if checkpoint:
        st.success(f"Checkpoint local prêt: {checkpoint.name}")
    else:
        st.warning("Aucun checkpoint local détecté. Vous pouvez fournir un fichier compatible ci-dessous.")
    _step(2, "Charger une échographie BUSI", "Le masque de vérité terrain est facultatif et doit avoir exactement les mêmes dimensions.")
    with st.container(border=True):
        upload = st.file_uploader("Échographie mammaire BUSI", type=IMAGE_TYPES, key="tp4-image")
        truth = st.file_uploader("Masque BUSI correspondant · optionnel", type=IMAGE_TYPES, key="tp4-truth")
        with st.expander("Configuration alternative du checkpoint", expanded=False):
            checkpoint_upload = st.file_uploader("Checkpoint Keras/HDF5", type=["h5", "hdf5", "keras"], key="tp4-checkpoint")
            st.caption("Le fichier local validé reste prioritaire sauf si vous chargez explicitement un autre checkpoint.")
    if upload is None:
        st.info("Ajoutez une image BUSI pour activer l'inférence.")
        return
    try:
        image = load_rgb(upload)
        gray = ensure_gray(image)
    except ValueError as exc:
        st.error(f"Échographie non chargée: {exc}")
        return
    try:
        reference, reference_error = _load_reference(truth, False, gray.shape)
    except ValueError as exc:
        reference, reference_error = None, str(exc)
    if reference_error:
        st.warning(reference_error + " Dice et IoU sont désactivés.")
    _image_summary(image, reference)
    st.image(image, caption="Échographie importée", width=520)
    checkpoint_signature = checkpoint_upload if checkpoint_upload is not None else (
        f"{checkpoint}:{checkpoint.stat().st_mtime_ns}" if checkpoint else None)
    signature = _signature("tp4", upload, truth, checkpoint_signature)
    can_run = TENSORFLOW_AVAILABLE and (checkpoint_upload is not None or checkpoint is not None)
    _step(3, "Lancer l'inférence", "Le modèle est chargé en cache; changer un contrôle d'affichage ne relance pas la prédiction.")
    if st.button("Lancer le U-Net TP4", type="primary", use_container_width=True,
                 disabled=not can_run, key="tp4-run"):
        try:
            start = perf_counter()
            with st.spinner("Inférence U-Net en cours…"):
                if checkpoint_upload is not None:
                    model = _load_uploaded_model(checkpoint_upload.getvalue(), Path(checkpoint_upload.name).suffix.lower())
                else:
                    model = _load_local_model(str(checkpoint), checkpoint.stat().st_mtime_ns)
                from core.unet_tp4 import predict_busi_outputs
                probability, mask = predict_busi_outputs(model, gray)
            _save_result("tp4-result", signature,
                         {"probability": probability, "mask": mask, "seconds": perf_counter() - start})
        except (ValueError, FileNotFoundError, ImportError, OSError, cv2.error) as exc:
            st.error(f"Inférence impossible: {exc}")
        except Exception as exc:
            st.error(f"Le checkpoint n'a pas pu être exécuté ({type(exc).__name__}): {exc}")
    result = _saved_result("tp4-result", signature)
    if result is None:
        if not can_run:
            st.warning("Installez TensorFlow et fournissez le checkpoint TP4 pour activer le bouton.")
        return
    _step(4, "Interpréter la sortie", "La probabilité, le masque strict >0,5 et la superposition sont affichés séparément.")
    views = st.columns(2)
    views[0].image(image, caption="Image originale", use_container_width=True)
    views[1].image(result["probability"], caption="Carte de probabilité", clamp=True, use_container_width=True)
    views = st.columns(2)
    views[0].image(result["mask"], caption="Segmentation binaire · > 0,5", clamp=True, use_container_width=True)
    views[1].image(overlay(image, result["mask"]), caption="Superposition", use_container_width=True)
    metrics = scores(result["mask"], reference) if reference is not None else {}
    metric_columns = st.columns(3 if metrics else 1)
    metric_columns[0].metric("Temps d'inférence", f'{result["seconds"]:.3f} s')
    if metrics:
        metric_columns[1].metric("Dice", f'{metrics["Dice"]:.3f}')
        metric_columns[2].metric("IoU", f'{metrics["IoU"]:.3f}')
    else:
        st.caption("Dice et IoU sont volontairement absents sans masque BUSI correspondant et aligné.")
    records = [{"Méthode": "U-Net BUSI", "Durée (s)": result["seconds"], **metrics}]
    actions = st.columns(2)
    actions[0].download_button("Télécharger le masque PNG", mask_png_bytes(result["mask"]),
                               file_name="tp4_unet_mask.png", mime="image/png", use_container_width=True)
    actions[1].download_button("Télécharger les métriques CSV", results_csv_bytes(records),
                               file_name="tp4_unet_resultats.csv", mime="text/csv", use_container_width=True)
    with st.expander("Traçabilité de l'inférence", expanded=False):
        st.json({
            "tp": "TP4",
            "modele": "simple_unet_model(256, 256, 1)",
            "checkpoint": checkpoint_upload.name if checkpoint_upload is not None else str(checkpoint),
            "normalisation": "keras.utils.normalize(axis=1)",
            "seuil": "strictement > 0.5",
            "image": {"largeur": image.shape[1], "hauteur": image.shape[0], "dtype": str(image.dtype)},
            "verite_terrain_alignee": reference is not None,
        })


def render_tp5():
    from core.tp5 import RegistrationError, parse_landmarks, register_pair, residual_optical_flow
    _header("TP5 · FIRE", "Recalage d'images rétiniennes",
            "Alignez une image mobile sur une référence avec SIFT/ORB, RANSAC et une transformation globale du notebook FIRE.")
    _step(1, "Charger la paire", "Les zones fixe et mobile sont séparées; les landmarks FIRE restent facultatifs et servent uniquement à la TRE.")
    with st.container(border=True):
        upload_columns = st.columns(2)
        fixed_upload = upload_columns[0].file_uploader("Image fixe / référence", type=IMAGE_TYPES, key="tp5-fixed")
        moving_upload = upload_columns[1].file_uploader("Image mobile / test", type=IMAGE_TYPES, key="tp5-moving")
        landmark_upload = st.file_uploader("Landmarks FIRE · optionnel (.txt)", type=["txt"], key="tp5-landmarks")
        st.caption("Format landmarks: x_fixe y_fixe x_mobile y_mobile. Ils ne participent jamais à RANSAC.")
    if fixed_upload is None or moving_upload is None:
        st.info("Ajoutez les deux images pour afficher les paramètres de recalage.")
        return
    try:
        fixed_full, moving_full = load_rgb(fixed_upload), load_rgb(moving_upload)
    except ValueError as exc:
        st.error(f"Paire non chargée: {exc}")
        return
    input_columns = st.columns(2)
    input_columns[0].image(fixed_full, caption="Image fixe", use_container_width=True)
    input_columns[1].image(moving_full, caption="Image mobile", use_container_width=True)
    _step(2, "Choisir la transformation", "Les paramètres SIFT/ORB, Lowe et RANSAC restent ceux du notebook.")
    with st.container(border=True):
        model_label = st.radio("Transformation globale", ["Similarité", "Affine", "Homographie"],
                               index=1, horizontal=True, key="tp5-model")
        model = {"Similarité": "similarity", "Affine": "affine", "Homographie": "homography"}[model_label]
        with st.expander("Paramètres scientifiques conservés", expanded=False):
            st.markdown("- côté maximal: **850 px**\n- canal vert + CLAHE **2.0 / 8×8**\n"
                        "- SIFT (repli ORB), Lowe **0,78**\n- RANSAC: paramètres exacts du TP5")
    signature = _signature("tp5", fixed_upload, moving_upload, landmark_upload, model)
    _step(3, "Lancer le recalage", "L'analyse coûteuse ne se relance pas lors d'une interaction d'affichage.")
    if st.button("Lancer le TP5", type="primary", use_container_width=True, key="tp5-run"):
        try:
            landmarks = parse_landmarks(landmark_upload.getvalue()) if landmark_upload is not None else None
            start = perf_counter()
            with st.spinner(f"Recalage {model_label.lower()} en cours…"):
                result = register_pair(fixed_full, moving_full, model=model, max_side=850, landmarks=landmarks)
                flow = residual_optical_flow(result.fixed, result.registered, optical_side=440)
            _save_result("tp5-result", signature,
                         {"result": result, "flow": flow, "seconds": perf_counter() - start})
        except (RegistrationError, ValueError, cv2.error) as exc:
            st.error(f"Recalage non estimé: {exc}")
            st.info("Vérifiez le recouvrement et la qualité des images, ou essayez une autre transformation.")
    analysis = _saved_result("tp5-result", signature)
    if analysis is None:
        return
    result, (flow_rgb, refined, mean_flow, registered_small) = analysis["result"], analysis["flow"]
    _step(4, "Inspecter l'estimation", "Commencez par les correspondances et les indicateurs RANSAC.")
    metrics_columns = st.columns(5)
    metrics_columns[0].metric("Détecteur", result.detector)
    metrics_columns[1].metric("Correspondances", result.n_matches)
    metrics_columns[2].metric("Inliers", result.n_inliers)
    metrics_columns[3].metric("Recouvrement", f"{result.overlap_fraction:.1%}")
    metrics_columns[4].metric("Durée", f'{analysis["seconds"]:.2f} s')
    st.image(result.matches_image, caption="Correspondances mobile → fixe", use_container_width=True)
    _step(5, "Comparer avant et après", "Les images conservent leur ratio; les métriques utilisent les mêmes pixels valides.")
    views = st.columns(3)
    views[0].image(result.fixed, caption="Fixe", use_container_width=True)
    views[1].image(result.moving, caption="Mobile", use_container_width=True)
    views[2].image(result.registered, caption=f"Recalée · {model_label}", use_container_width=True)
    views = st.columns(3)
    views[0].image(result.before_overlay, caption="Superposition avant", use_container_width=True)
    views[1].image(result.after_overlay, caption="Superposition après", use_container_width=True)
    views[2].image(result.absolute_difference, caption="Différence absolue", use_container_width=True)
    metric_table = pd.DataFrame([{"Étape": "Avant", **result.before_metrics},
                                 {"Étape": "Après", **result.after_metrics}]).rename(
                                     columns={"mse": "MSE", "ncc": "NCC", "nmi": "NMI"})
    st.dataframe(metric_table, use_container_width=True, hide_index=True)
    if metric_table[["MSE", "NCC", "NMI"]].isna().any().any():
        st.warning("Métriques indisponibles: moins de 500 pixels valides ou variance insuffisante.")
    if result.landmark_metrics is not None:
        st.markdown("#### TRE · évaluation indépendante")
        landmark_columns = st.columns(4)
        landmark_columns[0].metric("TRE avant", f'{result.landmark_metrics["tre_before_px"]:.2f} px')
        landmark_columns[1].metric("TRE après", f'{result.landmark_metrics["tre_after_px"]:.2f} px')
        landmark_columns[2].metric("TRE médiane", f'{result.landmark_metrics["tre_median_px"]:.2f} px')
        landmark_columns[3].metric("Landmarks", result.landmark_metrics["n_landmarks"])
        st.image(result.landmark_image, caption="Vert: référence · rouge: projeté · jaune: erreur", use_container_width=True)
    st.markdown("#### Flot optique résiduel de Farnebäck")
    flow_columns = st.columns(3)
    flow_columns[0].image(registered_small, caption="Recalage global", use_container_width=True)
    flow_columns[1].image(flow_rgb, caption="Champ résiduel HSV", use_container_width=True)
    flow_columns[2].image(refined, caption="Correction locale indicative", use_container_width=True)
    st.caption(f"Norme moyenne: {mean_flow:.2f} px. Le flot peut aussi refléter l'éclairage, le bruit ou une erreur de recalage.")
    export_record = {"model": model, "detector": result.detector, "n_matches": result.n_matches,
                     "n_inliers": result.n_inliers, "overlap_fraction": result.overlap_fraction,
                     "elapsed_seconds": analysis["seconds"],
                     **{f"before_{key}": value for key, value in result.before_metrics.items()},
                     **{f"after_{key}": value for key, value in result.after_metrics.items()},
                     **(result.landmark_metrics or {})}
    _step(6, "Télécharger", "Conservez l'image, les métriques et les matrices dans le sens mobile → fixe.")
    actions = st.columns(3)
    actions[0].download_button("Image recalée PNG", _rgb_png_bytes(result.registered),
                               file_name=f"tp5_registered_{model}.png", mime="image/png", use_container_width=True)
    actions[1].download_button("Résultats CSV", pd.DataFrame([export_record]).to_csv(index=False).encode("utf-8-sig"),
                               file_name="tp5_registration_results.csv", mime="text/csv", use_container_width=True)
    actions[2].download_button("Transformation JSON", json.dumps({"model": model, "direction": "moving_to_fixed",
                               "processed_matrix": result.transform_processed.tolist(),
                               "original_coordinate_matrix": result.transform_full.tolist()}, indent=2).encode("utf-8"),
                               file_name="tp5_transformation.json", mime="application/json", use_container_width=True)
    with st.expander("Transformation et traçabilité", expanded=False):
        st.caption("Matrice 3×3 dans le repère des images importées · direction mobile → fixe")
        st.dataframe(pd.DataFrame(result.transform_full), use_container_width=True, hide_index=True)
        st.json({
            "tp": "TP5",
            "modele": model,
            "detecteur": result.detector,
            "correspondances": result.n_matches,
            "inliers_ransac": result.n_inliers,
            "landmarks_utilises_pour_estimation": False,
            "landmarks_evaluation_disponibles": result.landmark_metrics is not None,
        })


def render_guide():
    _header("Référence", "Guide scientifique et limites",
            "Retrouvez la provenance des méthodes, les adaptations explicites et les règles d'interprétation.")
    for name, explanation in METHOD_HELP.items():
        with st.container(border=True):
            st.markdown(f"#### {name}")
            st.caption(METHOD_SOURCE[name])
            st.write(explanation)
    st.warning("Les sorties sont pédagogiques et ne doivent pas guider un diagnostic ou une décision clinique.")
    st.info("BUSI, BAGLS et FIRE sont des contextes distincts. Ne comparez pas leurs métriques comme un benchmark commun. "
            "Une vérité terrain doit correspondre à l'image et être correctement alignée.")
    st.markdown("**Normalisation TP4:** redimensionnement 256×256, `keras.utils.normalize(axis=1)`, ajout du canal, seuil strict `>0.5`.\n\n"
                "**Landmarks TP5:** utilisés exclusivement après l'estimation SIFT/ORB + RANSAC pour calculer la TRE.")


with st.sidebar:
    st.markdown('<div class="sidebar-brand">🩻 MedImage Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-copy">Plateforme universitaire de traitement et d’analyse d’images médicales.</div>',
                unsafe_allow_html=True)
    page = st.radio("Choisir un TP", NAVIGATION, key="navigation")
    st.divider()
    checkpoint = _configured_checkpoint()
    st.caption("ÉTAT DU SYSTÈME")
    st.write("● Traitement conventionnel prêt")
    st.write("● TP4 prêt" if TENSORFLOW_AVAILABLE and checkpoint else "○ TP4 incomplet")
    st.write("● TP5 prêt")
    with st.expander("Aide rapide", expanded=False):
        st.write("1. Choisissez un TP.\n2. Chargez les données.\n3. Configurez.\n4. Lancez.\n5. Téléchargez.")
    st.caption("Usage pédagogique uniquement · Aucun diagnostic")


RENDERERS = {NAVIGATION[0]: render_tp1, NAVIGATION[1]: render_tp2, NAVIGATION[2]: render_tp3,
             NAVIGATION[3]: render_tp4, NAVIGATION[4]: render_tp5, NAVIGATION[5]: render_guide}
RENDERERS[page]()
st.markdown('<div class="footer">MedImage Lab · SUP’COM · TP1–TP5 · Résultats non destinés au diagnostic</div>',
            unsafe_allow_html=True)
