import cv2
import numpy as np

from LiveFT import FrameProcessor
from liveft_processing import ProcessingConfig, process_image


def _config(**overrides) -> ProcessingConfig:
    values = {
        "window": False,
        "representation": "power",
        "display_scaling": "linear",
    }
    values.update(overrides)
    return ProcessingConfig(**values)


def test_process_image_is_deterministic() -> None:
    random = np.random.default_rng(1234)
    image = random.integers(0, 256, size=(65, 65), dtype=np.uint8)

    first = process_image(image, ProcessingConfig())
    second = process_image(image, ProcessingConfig())

    np.testing.assert_array_equal(first.processed_image, second.processed_image)
    np.testing.assert_array_equal(first.spectrum, second.spectrum)
    np.testing.assert_array_equal(first.display_spectrum, second.display_spectrum)


def test_shared_api_matches_live_frame_processor_defaults() -> None:
    random = np.random.default_rng(7)
    bgr_image = random.integers(0, 256, size=(65, 81, 3), dtype=np.uint8).astype(np.float32)

    live_image, live_fft = FrameProcessor()(bgr_image.copy())
    shared = process_image(bgr_image.copy(), ProcessingConfig())

    np.testing.assert_array_equal(shared.processed_image, live_image)
    np.testing.assert_array_equal(shared.display_spectrum, live_fft)


def test_periodic_pattern_has_expected_fft_peak_positions() -> None:
    size = 128
    cycles = 7
    x = np.arange(size)
    image = np.tile(1.0 + np.cos(2 * np.pi * cycles * x / size), (size, 1)).astype(np.float32)

    result = process_image(image, _config())
    spectrum = result.spectrum.copy()
    center = size // 2
    spectrum[center, center] = 0
    peak_positions = np.argpartition(spectrum.ravel(), -2)[-2:]
    peaks = {tuple(position) for position in np.column_stack(np.unravel_index(peak_positions, spectrum.shape))}

    assert peaks == {(center, center - cycles), (center, center + cycles)}


def test_circle_size_has_reciprocal_fourier_scaling() -> None:
    size = 257
    y, x = np.ogrid[:size, :size]
    center = size // 2

    def first_minimum(radius: int) -> int:
        circle = ((x - center) ** 2 + (y - center) ** 2 <= radius**2).astype(np.float32)
        spectrum = process_image(circle, _config(representation="magnitude")).spectrum
        line = spectrum[center, center:] / spectrum[center, center]
        candidates = np.flatnonzero(line[1:] < 0.01)
        assert candidates.size
        return int(candidates[0] + 1)

    small_minimum = first_minimum(12)
    large_minimum = first_minimum(24)

    assert 1.7 < small_minimum / large_minimum < 2.3


def test_rotating_anisotropic_pattern_rotates_spectrum() -> None:
    size = 129
    image = np.zeros((size, size), dtype=np.float32)
    cv2.ellipse(image, (size // 2, size // 2), (28, 9), 0, 0, 360, 1.0, -1)

    original = process_image(image, _config()).spectrum
    rotated = process_image(np.rot90(image), _config()).spectrum

    np.testing.assert_allclose(rotated, np.rot90(original), rtol=2e-5, atol=2e-2)


def test_complementary_patterns_match_away_from_dc() -> None:
    random = np.random.default_rng(42)
    image = random.integers(0, 2, size=(65, 65), dtype=np.uint8).astype(np.float32)

    original = process_image(image, _config()).spectrum
    complement = process_image(1.0 - image, _config()).spectrum
    center = tuple(dimension // 2 for dimension in image.shape)
    original[center] = 0
    complement[center] = 0

    np.testing.assert_allclose(original, complement, rtol=2e-5, atol=0.25)


def test_display_controls_do_not_change_numerical_spectrum() -> None:
    image = np.arange(81, dtype=np.float32).reshape(9, 9)
    baseline = process_image(image, _config(gamma=1.0, suppress_center_lines=False))
    enhanced = process_image(image, _config(gamma=0.5, suppress_center_lines=True))

    np.testing.assert_array_equal(baseline.spectrum, enhanced.spectrum)
    assert not np.array_equal(baseline.display_spectrum, enhanced.display_spectrum)
