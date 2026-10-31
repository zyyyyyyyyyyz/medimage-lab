from streamlit.testing.v1 import AppTest


PAGES = [
    "TP1 · Manipulation d'image",
    "TP2 · Prétraitement",
    "TP3 · Segmentation conventionnelle",
    "TP4 · U-Net BUSI",
    "TP5 · Recalage FIRE",
    "Guide scientifique",
]


def test_all_five_tp_pages_render_without_application_errors():
    app = AppTest.from_file("app.py", default_timeout=120).run()
    assert list(app.sidebar.radio[0].options) == PAGES
    for page in PAGES:
        app.sidebar.radio[0].set_value(page)
        app.run(timeout=120)
        assert not app.exception, page
        assert not app.error, page


def test_tp2_explicit_run_exposes_visual_results_and_downloads():
    app = AppTest.from_file("app.py", default_timeout=120).run()
    app.sidebar.radio[0].set_value("TP2 · Prétraitement")
    app.run()
    next(button for button in app.button if button.label == "Lancer le TP2").click()
    app.run(timeout=120)
    assert not app.exception
    assert not app.error
    assert len(app.get("imgs")) >= 7
    labels = [button.label for button in app.get("download_button")]
    assert labels.count("Télécharger le masque PNG") == 2
    assert "Télécharger l'image prétraitée" in labels
    assert "Télécharger les métriques CSV" in labels


def test_sidebar_is_navigation_and_status_only():
    app = AppTest.from_file("app.py", default_timeout=120).run()
    assert len(app.sidebar.radio) == 1
    assert app.sidebar.radio[0].label == "Choisir un TP"
    assert not app.sidebar.get("file_uploader")
    assert not app.sidebar.multiselect
    assert not app.sidebar.slider
