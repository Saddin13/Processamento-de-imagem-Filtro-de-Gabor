"""Núcleo do filtro de Gabor: funções puras, sem interface gráfica."""

import io
import math

import numpy as np
from PIL import Image, ImageOps


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Converte para tons de cinza: Y = 0,299 R + 0,587 G + 0,114 B.

    Aceita HxW (já cinza), HxWx3 (RGB) ou HxWx4 (RGBA, alpha ignorado).
    Retorna float32 HxW na faixa 0..255.
    """
    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    rgb = arr[..., :3].astype(np.float32)
    return (
        0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    ).astype(np.float32)


def gabor_kernel(
    ksize: int,
    sigma: float,
    theta_deg: float,
    lambd: float,
    gamma: float,
    psi_deg: float,
    imag: bool = False,
) -> np.ndarray:
    """Kernel de Gabor pela fórmula matemática.

    g(x,y) = exp(-(x'^2 + gamma^2 y'^2) / (2 sigma^2)) * cos(2 pi x'/lambda + psi)
    x' =  x cos(theta) + y sin(theta)
    y' = -x sin(theta) + y cos(theta)

    Com imag=True usa seno no lugar do cosseno (parte imaginária).
    O eixo y cresce para cima (linha 0 <-> y = +ksize//2), como no OpenCV.
    """
    if ksize < 1 or ksize % 2 == 0:
        raise ValueError("ksize deve ser um inteiro ímpar >= 1")
    if sigma <= 0 or lambd <= 0 or gamma <= 0:
        raise ValueError("sigma, lambd e gamma devem ser positivos")

    half = ksize // 2
    ys, xs = np.mgrid[half : -half - 1 : -1, -half : half + 1]
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)

    theta = math.radians(theta_deg)
    psi = math.radians(psi_deg)
    x_r = xs * math.cos(theta) + ys * math.sin(theta)
    y_r = -xs * math.sin(theta) + ys * math.cos(theta)

    envelope = np.exp(-(x_r**2 + (gamma**2) * y_r**2) / (2.0 * sigma**2))
    phase = 2.0 * math.pi * x_r / lambd + psi
    wave = np.sin(phase) if imag else np.cos(phase)
    return (envelope * wave).astype(np.float32)


def max_diff_vs_opencv(
    ksize: int,
    sigma: float,
    theta_deg: float,
    lambd: float,
    gamma: float,
    psi_deg: float,
) -> float | None:
    """Maior diferença absoluta entre nosso kernel e cv2.getGaborKernel.

    O OpenCV é opcional: retorna None se não puder ser importado
    (não instalado ou bloqueado pelo sistema).
    """
    try:
        import cv2
    except ImportError:
        return None

    ours = gabor_kernel(ksize, sigma, theta_deg, lambd, gamma, psi_deg)
    ref = cv2.getGaborKernel(
        (ksize, ksize),
        sigma,
        math.radians(theta_deg),
        lambd,
        gamma,
        math.radians(psi_deg),
        ktype=cv2.CV_32F,
    )
    return float(np.max(np.abs(ours - ref)))


# --------------------------------------------------------------------------
# Imagens: carregar, reduzir, imagem de teste
# --------------------------------------------------------------------------


def load_rgb(source) -> np.ndarray:
    """Abre uma imagem (caminho ou arquivo aberto) como uint8 HxWx3 RGB.

    Respeita a orientação EXIF (fotos de celular) e descarta alpha/paleta.
    """
    with Image.open(source) as im:
        im = ImageOps.exif_transpose(im)
        return np.array(im.convert("RGB"), dtype=np.uint8)


def resize_max_side(img: np.ndarray, max_side: int = 1600) -> np.ndarray:
    """Reduz a imagem (mantendo proporção) se o lado maior passar de max_side."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    scale = max_side / longest
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = Image.fromarray(img).resize((new_w, new_h), Image.LANCZOS)
    return np.array(resized)


def synthetic_test_image(size: int = 512) -> np.ndarray:
    """Imagem de teste: quatro quadrantes coloridos com listras em 0, 45, 90 e 135 graus."""
    half = size // 2
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    tints = [(1.0, 0.6, 0.5), (0.5, 1.0, 0.6), (0.5, 0.7, 1.0), (1.0, 0.9, 0.4)]
    angles = [0, 45, 90, 135]
    quadrants = [(slice(0, half), slice(0, half)), (slice(0, half), slice(half, size)),
                 (slice(half, size), slice(0, half)), (slice(half, size), slice(half, size))]
    out = np.zeros((size, size, 3), dtype=np.float32)
    for (rows, cols), angle, tint in zip(quadrants, angles, tints):
        a = math.radians(angle)
        proj = xs[rows, cols] * math.cos(a) + ys[rows, cols] * math.sin(a)
        wave = 127.5 + 110.0 * np.cos(2 * math.pi * proj / 12.0)
        for ch in range(3):
            out[rows, cols, ch] = wave * tint[ch]
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Filtragem e exibição
# --------------------------------------------------------------------------


def _next_fast_len(n: int) -> int:
    """Menor inteiro >= n cujos fatores primos são só 2, 3 e 5 (FFT rápida)."""
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def _convolve_same(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolução 2D verdadeira (mesmo tamanho da imagem), via FFT, borda espelhada."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    h, w = gray.shape
    padded = np.pad(gray.astype(np.float32), ((ph, ph), (pw, pw)), mode="symmetric")
    shape = (_next_fast_len(padded.shape[0] + kh - 1), _next_fast_len(padded.shape[1] + kw - 1))
    spectrum = np.fft.rfft2(padded, shape) * np.fft.rfft2(kernel.astype(np.float32), shape)
    full = np.fft.irfft2(spectrum, shape)
    return full[kh - 1 : kh - 1 + h, kw - 1 : kw - 1 + w].astype(np.float32)


def apply_gabor(
    gray: np.ndarray,
    kernel_real: np.ndarray,
    kernel_imag: np.ndarray | None = None,
    mode: str = "real",
) -> np.ndarray:
    """Filtra a imagem em tons de cinza com o kernel de Gabor.

    mode="real": resposta do kernel real.
    mode="magnitude": sqrt(real^2 + imag^2), exige kernel_imag.
    """
    if mode not in ("real", "magnitude"):
        raise ValueError(f"mode inválido: {mode!r}")
    if mode == "magnitude" and kernel_imag is None:
        raise ValueError("mode='magnitude' exige kernel_imag")
    real = _convolve_same(gray, kernel_real)
    if mode == "real":
        return real
    imag = _convolve_same(gray, kernel_imag)
    return np.sqrt(real**2 + imag**2).astype(np.float32)


def normalize_for_display(arr: np.ndarray) -> np.ndarray:
    """Min-max para uint8 0..255; array constante ou inválido vira tudo zero."""
    a = np.nan_to_num(np.asarray(arr, dtype=np.float64))
    lo, hi = float(a.min()), float(a.max())
    if hi == lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return np.round((a - lo) / (hi - lo) * 255.0).astype(np.uint8)


def kernel_to_image(kernel: np.ndarray, size: int = 512) -> np.ndarray:
    """Kernel como imagem RGB (coolwarm, zero no meio), ampliada sem suavização."""
    from matplotlib import colormaps

    scale = max(float(np.abs(kernel).max()), 1e-12)
    values = (kernel / scale + 1.0) / 2.0
    rgb = (colormaps["coolwarm"](values)[..., :3] * 255).astype(np.uint8)
    return np.array(Image.fromarray(rgb).resize((size, size), Image.NEAREST))


def to_png_bytes(img: np.ndarray) -> bytes:
    """Codifica a imagem (HxW ou HxWx3) como PNG, para o botão de download.

    Arrays que não são uint8 são arredondados e limitados a 0..255.
    """
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(np.rint(np.nan_to_num(arr.astype(np.float64))), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()
