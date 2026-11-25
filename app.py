"""Filtro de Gabor — Vitor Saddi · PDI.

Rodar:  streamlit run app.py
"""

import numpy as np
import streamlit as st

from gabor_core import (
    apply_gabor,
    gabor_kernel,
    kernel_to_image,
    load_rgb,
    max_diff_vs_opencv,
    normalize_for_display,
    resize_max_side,
    synthetic_test_image,
    to_grayscale,
    to_png_bytes,
)

MODE_REAL = "Parte real"
MODE_MAGNITUDE = "Magnitude (real + imaginária)"

st.set_page_config(page_title="Filtro de Gabor", layout="wide")


def show(img: np.ndarray, filename: str, key: str, caption: str | None = None) -> None:
    """Mostra a imagem na largura da coluna, com um botão de download (PNG) embaixo."""
    try:
        st.image(img, caption=caption, width="stretch")
    except TypeError:  # versões antigas do Streamlit
        st.image(img, caption=caption, use_container_width=True)
    options = dict(
        label="Baixar PNG",
        data=to_png_bytes(img),
        file_name=filename,
        mime="image/png",
        icon=":material/download:",
        key=key,
    )
    try:
        st.download_button(on_click="ignore", **options)  # não recarrega a página
    except TypeError:
        st.download_button(**options)


# ---------------------------------------------------------------- cabeçalho
st.title("Filtro de Gabor")
st.caption("Vitor Saddi · Processamento Digital de Imagens")

# ------------------------------------------------------------ barra lateral
with st.sidebar:
    st.header("Imagem")
    upload = st.file_uploader(
        "Escolha uma imagem",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
        key="upload",
    )

    st.header("Parâmetros do filtro de Gabor")
    ksize = st.slider("Tamanho do kernel (ímpar)", 3, 101, 31, step=2, key="ksize")
    lambd = st.slider("Comprimento de onda λ (pixels)", 2.0, 50.0, 10.0, step=0.5, key="lambd")
    theta = st.slider("Orientação θ (graus)", 0, 180, 45, step=1, key="theta")
    psi = st.slider("Fase ψ (graus)", 0, 360, 0, step=5, key="psi")
    sigma = st.slider("Desvio padrão σ (envelope)", 1.0, 20.0, 5.0, step=0.5, key="sigma")
    gamma = st.slider("Razão de aspecto γ", 0.1, 2.0, 0.5, step=0.05, key="gamma")
    mode_label = st.radio("Resposta exibida", [MODE_REAL, MODE_MAGNITUDE], key="mode")
    compare_opencv = st.checkbox("Conferir kernel com o OpenCV", key="compare_opencv")

# ------------------------------------------------------------------ imagem
if upload is not None:
    try:
        image = resize_max_side(load_rgb(upload))
    except Exception:
        st.error("Não consegui abrir esse arquivo como imagem. Tente outro.")
        st.stop()
else:
    image = synthetic_test_image()
    st.caption("Nenhuma imagem enviada: usando a imagem de teste (listras em 0°, 45°, 90° e 135°).")

# ------------------------------------------------------ 1. imagem original
st.header("1. Imagem original")
show(image, "1_original.png", "dl_original", f"{image.shape[1]} × {image.shape[0]} pixels")

# ----------------------------------------------------- 2. tons de cinza
st.header("2. Tons de cinza")
st.latex(r"Y = 0{,}299\,R + 0{,}587\,G + 0{,}114\,B")
gray = to_grayscale(image)
show(np.clip(gray, 0, 255).astype(np.uint8), "2_tons_de_cinza.png", "dl_cinza")

# ------------------------------------------------------ 3. kernel de Gabor
st.header("3. Kernel de Gabor")
st.latex(
    r"g(x,y)=\exp\!\left(-\frac{x'^2+\gamma^2 y'^2}{2\sigma^2}\right)"
    r"\cos\!\left(2\pi\,\frac{x'}{\lambda}+\psi\right)"
)
st.latex(r"x' = x\cos\theta + y\sin\theta \qquad y' = -x\sin\theta + y\cos\theta")
st.caption(
    f"λ = {lambd}  ·  θ = {theta}°  ·  ψ = {psi}°  ·  σ = {sigma}  ·  γ = {gamma}  ·  "
    f"kernel {ksize} × {ksize}  (vermelho = positivo, azul = negativo)"
)
kernel_real = gabor_kernel(ksize, sigma, theta, lambd, gamma, psi)
kernel_imag = gabor_kernel(ksize, sigma, theta, lambd, gamma, psi, imag=True)
params_tag = f"theta{theta}_lambda{lambd:g}_sigma{sigma:g}_gamma{gamma:g}_psi{psi}_k{ksize}"
show(kernel_to_image(kernel_real), f"3_kernel_gabor_{params_tag}.png", "dl_kernel")

# --------------------------------------------------- 4. imagem filtrada
st.header("4. Imagem filtrada")
if mode_label == MODE_MAGNITUDE:
    st.caption("Magnitude = √(real² + imaginária²): resposta da textura independente da fase.")
    response = apply_gabor(gray, kernel_real, kernel_imag, mode="magnitude")
else:
    st.caption("Parte real: convolução da imagem em cinza com o kernel (normalizada para exibição).")
    response = apply_gabor(gray, kernel_real, mode="real")
mode_tag = "magnitude" if mode_label == MODE_MAGNITUDE else "real"
show(normalize_for_display(response), f"4_filtrada_{mode_tag}_{params_tag}.png", "dl_filtrada")

# ------------------------------------------------ 5. conferência (opcional)
if compare_opencv:
    st.header("5. Conferência com o OpenCV")
    diff = max_diff_vs_opencv(ksize, sigma, theta, lambd, gamma, psi)
    if diff is None:
        st.markdown("OpenCV indisponível nesta máquina (não instalado ou bloqueado pelo sistema).")
    else:
        st.markdown(f"Diferença máxima entre o nosso kernel e `cv2.getGaborKernel`: **{diff:.2e}**")
