import math
import sys

import numpy as np
import pytest

from gabor_core import gabor_kernel, max_diff_vs_opencv, to_grayscale


def test_grayscale_white_is_255():
    white = np.full((2, 2, 3), 255, dtype=np.uint8)
    assert np.allclose(to_grayscale(white), 255.0, atol=1e-3)


def test_grayscale_pure_red_uses_luma_weight():
    red = np.zeros((1, 1, 3), dtype=np.uint8)
    red[..., 0] = 255
    assert to_grayscale(red)[0, 0] == pytest.approx(76.245, abs=1e-3)


def test_grayscale_accepts_2d_and_rgba():
    gray = np.array([[0, 10], [200, 255]], dtype=np.uint8)
    out2d = to_grayscale(gray)
    assert out2d.dtype == np.float32
    assert np.array_equal(out2d, gray.astype(np.float32))

    rgb = np.random.default_rng(0).integers(0, 256, (4, 5, 3), dtype=np.uint8)
    alpha = np.full((4, 5, 1), 17, dtype=np.uint8)
    rgba = np.concatenate([rgb, alpha], axis=2)
    assert np.allclose(to_grayscale(rgba), to_grayscale(rgb))


def test_kernel_shape_dtype_and_center():
    k = gabor_kernel(31, 5, 0, 10, 0.5, 0)
    assert k.shape == (31, 31)
    assert k.dtype == np.float32
    assert k[15, 15] == pytest.approx(1.0)


def test_kernel_real_is_point_symmetric():
    k = gabor_kernel(31, 5, 30, 10, 0.5, 0)
    assert np.allclose(k, k[::-1, ::-1])


def test_kernel_theta_zero_varies_along_columns():
    k = gabor_kernel(31, 20, 0, 10, 1.0, 0)
    center_row = k[15, :]
    assert (center_row > 0).any() and (center_row < 0).any()
    # Along the center column x'=0, so cos(0)=1: positive near the center.
    assert (k[12:19, 15] > 0).all()


def test_kernel_imag_is_zero_at_center():
    k = gabor_kernel(31, 5, 0, 10, 0.5, 0, imag=True)
    assert k[15, 15] == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize(
    "args",
    [
        (31, 5, 0, 10, 0.5, 0),
        (31, 5, 30, 10, 0.5, 0),
        (21, 3, 135, 8, 1.0, 90),
        (15, 2, 60, 6, 0.3, 45),
    ],
)
def test_kernel_matches_opencv(args):
    diff = max_diff_vs_opencv(*args)
    if diff is None:  # OpenCV ausente ou bloqueado por política do Windows
        pytest.skip("OpenCV indisponível nesta máquina")
    assert diff < 1e-5


def test_max_diff_vs_opencv_returns_none_when_cv2_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)  # força ImportError no import
    assert max_diff_vs_opencv(31, 5, 0, 10, 0.5, 0) is None


def _reference_kernel(ksize, sigma, theta_deg, lambd, gamma, psi_deg):
    """Implementação de referência em laços puros, direto da fórmula (y para cima)."""
    half = ksize // 2
    th, psi = math.radians(theta_deg), math.radians(psi_deg)
    out = np.zeros((ksize, ksize))
    for row in range(ksize):
        for col in range(ksize):
            x, y = col - half, half - row
            xr = x * math.cos(th) + y * math.sin(th)
            yr = -x * math.sin(th) + y * math.cos(th)
            out[row, col] = math.exp(-(xr**2 + gamma**2 * yr**2) / (2 * sigma**2)) * math.cos(
                2 * math.pi * xr / lambd + psi
            )
    return out


@pytest.mark.parametrize(
    "args",
    [(31, 5, 0, 10, 0.5, 0), (31, 5, 30, 10, 0.5, 0), (21, 3, 135, 8, 1.0, 90), (15, 2, 60, 6, 0.3, 45)],
)
def test_kernel_matches_reference_formula(args):
    assert np.allclose(gabor_kernel(*args), _reference_kernel(*args), atol=1e-6)


def test_kernel_ksize_one():
    k = gabor_kernel(1, 5, 0, 10, 0.5, 60)
    assert k.shape == (1, 1)
    assert k[0, 0] == pytest.approx(math.cos(math.radians(60)), abs=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(ksize=30, sigma=5, lambd=10, gamma=0.5),
        dict(ksize=0, sigma=5, lambd=10, gamma=0.5),
        dict(ksize=31, sigma=0, lambd=10, gamma=0.5),
        dict(ksize=31, sigma=5, lambd=0, gamma=0.5),
        dict(ksize=31, sigma=5, lambd=10, gamma=0),
    ],
)
def test_kernel_invalid_args_raise(kwargs):
    with pytest.raises(ValueError):
        gabor_kernel(theta_deg=0, psi_deg=0, **kwargs)
