from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _all_text(at):
    parts = []
    for group in (at.title, at.header, at.subheader, at.markdown, at.caption, at.text):
        parts.extend(str(el.value) for el in group)
    return "\n".join(parts)


def _new_app():
    return AppTest.from_file(APP, default_timeout=60)


def test_app_runs_with_default_image():
    at = _new_app().run()
    assert not at.exception
    assert "Vitor Saddi" in _all_text(at)


def test_app_reruns_after_changing_params():
    at = _new_app().run()
    at.sidebar.slider(key="theta").set_value(90).run()
    assert not at.exception
    at.sidebar.slider(key="ksize").set_value(11).run()
    assert not at.exception


def test_app_magnitude_mode_and_opencv_check():
    at = _new_app().run()
    at.sidebar.radio(key="mode").set_value("Magnitude (real + imaginária)")
    at.sidebar.checkbox(key="compare_opencv").set_value(True).run()
    assert not at.exception
    text = _all_text(at)
    # Mostra a diferença máxima ou avisa que o OpenCV não está disponível.
    assert "Diferença máxima" in text or "OpenCV indisponível" in text


def test_app_has_a_download_button_under_each_image():
    at = _new_app().run()
    assert not at.exception
    buttons = at.get("download_button")
    assert len(buttons) == 4  # original, cinza, kernel, filtrada
    labels = [b.proto.label for b in buttons]
    assert all("Baixar" in label for label in labels)
