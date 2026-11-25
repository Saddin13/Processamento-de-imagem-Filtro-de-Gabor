import io
import warnings

import numpy as np
import pytest
from PIL import Image

from gabor_core import (
    apply_gabor,
    gabor_kernel,
    kernel_to_image,
    load_rgb,
    normalize_for_display,
    resize_max_side,
    synthetic_test_image,
    to_png_bytes,
)


def test_to_png_bytes_roundtrip_rgb_and_gray():
    rgb = synthetic_test_image(64)
    data = to_png_bytes(rgb)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert np.array_equal(np.array(Image.open(io.BytesIO(data))), rgb)

    gray = np.arange(16, dtype=np.uint8).reshape(4, 4)
    back = np.array(Image.open(io.BytesIO(to_png_bytes(gray))))
    assert np.array_equal(back, gray)


def test_to_png_bytes_clips_float_arrays_to_uint8():
    data = to_png_bytes(np.array([[-5.0, 100.4], [255.0, 300.0]]))
    back = np.array(Image.open(io.BytesIO(data)))
    assert back.dtype == np.uint8
    assert back.tolist() == [[0, 100], [255, 255]]


def _stripes(size=128, lambd=10.0, vertical=True):
    coords = np.arange(size, dtype=np.float32)
    wave = 127.5 + 100.0 * np.cos(2 * np.pi * coords / lambd)
    img = np.tile(wave, (size, 1)) if vertical else np.tile(wave[:, None], (1, size))
    return img.astype(np.float32)


def test_load_rgb_applies_exif_orientation():
    img = Image.new("RGB", (40, 20), (200, 10, 10))
    exif = Image.Exif()
    exif[0x0112] = 6  # gira 90 graus
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    buf.seek(0)
    out = load_rgb(buf)
    assert out.shape == (40, 20, 3)


def test_load_rgb_converts_rgba_and_gray_to_rgb():
    for mode in ("RGBA", "L"):
        buf = io.BytesIO()
        Image.new(mode, (6, 4)).save(buf, format="PNG")
        buf.seek(0)
        out = load_rgb(buf)
        assert out.shape == (4, 6, 3)
        assert out.dtype == np.uint8


def test_resize_max_side_caps_large_image():
    big = np.zeros((2000, 4000, 3), dtype=np.uint8)
    out = resize_max_side(big, 1600)
    assert out.shape == (800, 1600, 3)
    small = np.zeros((50, 100, 3), dtype=np.uint8)
    assert resize_max_side(small, 1600).shape == (50, 100, 3)


def test_apply_gabor_shape_and_dtype():
    gray = _stripes(64)
    out = apply_gabor(gray, gabor_kernel(15, 3, 0, 10, 0.5, 0))
    assert out.shape == gray.shape
    assert out.dtype == np.float32


def test_apply_gabor_prefers_matching_orientation():
    gray = _stripes(128, lambd=10.0, vertical=True)
    matched = apply_gabor(gray - gray.mean(), gabor_kernel(31, 5, 0, 10, 0.5, 0))
    orthogonal = apply_gabor(gray - gray.mean(), gabor_kernel(31, 5, 90, 10, 0.5, 0))
    assert np.abs(matched).mean() > 3 * np.abs(orthogonal).mean()


def test_apply_gabor_magnitude_is_nonnegative_and_phase_invariant():
    gray = _stripes(128, lambd=10.0, vertical=True) - 127.5
    kr = gabor_kernel(31, 5, 0, 10, 0.5, 0)
    ki = gabor_kernel(31, 5, 0, 10, 0.5, 0, imag=True)
    mag = apply_gabor(gray, kr, ki, mode="magnitude")
    assert (mag >= 0).all()
    core = mag[40:88, 40:88]
    assert core.std() / core.mean() < 0.2


def test_apply_gabor_invalid_mode_raises():
    gray = _stripes(32)
    kr = gabor_kernel(7, 2, 0, 6, 0.5, 0)
    with pytest.raises(ValueError):
        apply_gabor(gray, kr, mode="foo")
    with pytest.raises(ValueError):
        apply_gabor(gray, kr, None, mode="magnitude")


def test_apply_gabor_image_smaller_than_kernel():
    tiny = np.random.default_rng(1).random((8, 8)).astype(np.float32)
    out = apply_gabor(tiny, gabor_kernel(31, 5, 0, 10, 0.5, 0))
    assert out.shape == (8, 8)
    assert np.isfinite(out).all()


def test_normalize_constant_image_is_zeros():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for arr in (np.full((5, 5), 7.0), np.zeros((5, 5))):
            out = normalize_for_display(arr)
            assert out.dtype == np.uint8
            assert (out == 0).all()


def test_normalize_range():
    out = normalize_for_display(np.array([-1.0, 0.0, 3.0]))
    assert out.min() == 0 and out.max() == 255


def test_kernel_to_image_shape_dtype():
    img = kernel_to_image(gabor_kernel(31, 5, 0, 10, 0.5, 0))
    assert img.shape == (512, 512, 3)
    assert img.dtype == np.uint8


def test_synthetic_test_image_shape_dtype():
    img = synthetic_test_image()
    assert img.shape == (512, 512, 3)
    assert img.dtype == np.uint8
    assert img.std() > 10
