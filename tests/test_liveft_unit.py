import sys

import cv2
import numpy as np
import pytest

from LiveFT import FrameProcessor, LiveFT, limitFPS, parse_args


def reference_fft(frame: np.ndarray, kill_center_lines: bool = False) -> np.ndarray:
    dft = cv2.dft(frame, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft = dft[:, :, 0] ** 2 + dft[:, :, 1] ** 2
    fft_log = np.log1p(np.fft.fftshift(dft))

    if kill_center_lines:
        h, w = fft_log.shape[:2]
        fft_log[h // 2 - 1 : h // 2 + 1, :] = fft_log[h // 2 + 1 : h // 2 + 3, :]
        fft_log[:, w // 2 - 1 : w // 2 + 1] = fft_log[:, w // 2 + 1 : w // 2 + 3]

    max_value = fft_log.max()
    if not np.isfinite(max_value) or max_value <= 0:
        return np.zeros_like(fft_log, dtype=np.float32)
    return fft_log / max_value


def test_prepare_frame_applies_crop_scale_and_grayscale() -> None:
    frame = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    processor = FrameProcessor(
        cropVertLo=1,
        cropVertUp=4,
        cropHorzLo=1,
        cropHorzUp=5,
        scaleVert=0.5,
        scaleHorz=0.5,
    )

    prepared = processor.prepareFrame(frame)
    expected = cv2.cvtColor(
        cv2.resize(frame[1:4, 1:5], None, fx=0.5, fy=0.5),
        cv2.COLOR_BGR2GRAY,
    )

    np.testing.assert_array_equal(prepared, expected)


def test_apply_window_normalizes_and_reuses_cached_window() -> None:
    processor = FrameProcessor(taperWidth=0.3)

    first = processor.applyWindow(np.ones((8, 8), dtype=np.float32))
    window_id = id(processor.window)
    second = processor.applyWindow(np.full((8, 8), 2.0, dtype=np.float32))

    assert first.min() == pytest.approx(0.0)
    assert first.max() == pytest.approx(1.0)
    assert second.min() == pytest.approx(0.0)
    assert second.max() == pytest.approx(1.0)
    assert id(processor.window) == window_id


def test_apply_window_handles_zero_signal_without_nan() -> None:
    processor = FrameProcessor()

    windowed = processor.applyWindow(np.zeros((8, 8), dtype=np.float32))

    assert np.isfinite(windowed).all()
    np.testing.assert_array_equal(windowed, np.zeros((8, 8), dtype=np.float32))


def test_compute_fft_matches_reference_implementation() -> None:
    frame = np.arange(64, dtype=np.float32).reshape(8, 8)
    processor = FrameProcessor()

    actual = processor.computeFFT(frame)
    expected = reference_fft(frame)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_compute_fft_applies_center_line_removal() -> None:
    frame = np.arange(64, dtype=np.float32).reshape(8, 8)
    processor = FrameProcessor(killCenterLines=True)

    actual = processor.computeFFT(frame)
    expected = reference_fft(frame, kill_center_lines=True)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_compute_fft_handles_flat_signal_after_center_line_removal() -> None:
    processor = FrameProcessor(killCenterLines=True)

    fft = processor.computeFFT(np.ones((8, 8), dtype=np.float32))

    assert np.isfinite(fft).all()
    np.testing.assert_array_equal(fft, np.zeros((8, 8), dtype=np.float32))


def test_limit_fps_sleeps_until_the_next_frame_budget() -> None:
    current_time = {"value": 1.0}
    sleep_calls = []

    def fake_time() -> float:
        return current_time["value"]

    def fake_sleep(duration: float) -> None:
        sleep_calls.append(duration)
        current_time["value"] += duration

    next_frame_time = limitFPS(1.04, 25.0, time_fn=fake_time, sleep_fn=fake_sleep)

    assert sleep_calls == [pytest.approx(0.04)]
    assert next_frame_time == pytest.approx(1.08)


def test_limit_fps_does_not_sleep_when_disabled_or_running_late() -> None:
    sleep_calls = []

    def fake_sleep(duration: float) -> None:
        sleep_calls.append(duration)

    no_limit_next_frame = limitFPS(5.0, 0.0, time_fn=lambda: 2.0, sleep_fn=fake_sleep)
    late_next_frame = limitFPS(1.0, 25.0, time_fn=lambda: 1.2, sleep_fn=fake_sleep)

    assert sleep_calls == []
    assert no_limit_next_frame == pytest.approx(2.0)
    assert late_next_frame == pytest.approx(1.2)


def test_parse_args_uses_liveft_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["liveft"])

    args = parse_args(LiveFT)

    assert args.numShots == int(1e5)
    assert args.camDevice == 0
    assert args.imAvgs == 1
    assert args.killCenterLines is False
    assert args.showInfo is False
    assert args.noGPU is True
    assert args.maxFPS == pytest.approx(0.0)


def test_parse_args_toggles_boolean_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["liveft", "-k", "-i", "-p", "-g", "-m", "25"])

    args = parse_args(LiveFT)

    assert args.killCenterLines is True
    assert args.showInfo is True
    assert args.downScale is True
    assert args.noGPU is False
    assert args.maxFPS == pytest.approx(25.0)
