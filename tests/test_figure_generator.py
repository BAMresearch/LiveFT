from pathlib import Path

import cv2
import numpy as np
import pytest

from liveft_figure import (
    compose_figure,
    export_figure,
    fit_to_canvas,
    generate_from_config,
    load_source_image,
    select_field_of_view,
)


def test_field_of_view_and_canvas_preserve_geometry() -> None:
    image = np.zeros((100, 200), dtype=np.uint8)
    image[25:75, 75:125] = 255

    selected = select_field_of_view(image, "center-square")
    fitted = fit_to_canvas(selected, (80, 120), fill=0)

    assert selected.shape == (100, 100)
    assert fitted.shape == (120, 80)
    # The original square stays square after aspect-preserving fitting.
    foreground = np.argwhere(fitted == 255)
    assert np.ptp(foreground[:, 0]) == np.ptp(foreground[:, 1])


def test_export_writes_valid_png_and_pdf(tmp_path: Path) -> None:
    pytest.importorskip("pymupdf")
    real = np.full((64, 64), 255, dtype=np.uint8)
    fourier = np.zeros((64, 64), dtype=np.uint8)
    fourier[32, 32] = 255
    figure = compose_figure([("Test", real, fourier)], figure_dpi=100, width_inches=4, row_height_inches=2)
    pdf = tmp_path / "figure.pdf"
    second_pdf = tmp_path / "figure_again.pdf"
    png = tmp_path / "figure.png"

    export_figure(figure, pdf, figure_dpi=100, png_preview=png)
    export_figure(figure, second_pdf, figure_dpi=100)

    assert pdf.read_bytes().startswith(b"%PDF")
    assert pdf.read_bytes() == second_pdf.read_bytes()
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert cv2.imread(str(png)).shape == figure.shape


def test_raster_source_loading(tmp_path: Path) -> None:
    path = tmp_path / "pattern.png"
    expected = np.full((12, 17, 3), (10, 20, 30), dtype=np.uint8)
    assert cv2.imwrite(str(path), expected)

    loaded = load_source_image(path)

    np.testing.assert_array_equal(loaded, expected)


def test_config_workflow_exports_figure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = np.full((100, 160, 3), 255, dtype=np.uint8)
    cv2.circle(source, (80, 50), 15, (0, 0, 0), -1)
    monkeypatch.setattr("liveft_figure.render_pdf_page", lambda path, page_number, dpi: source.copy())
    config = {
        "patterns": [{"source": "pattern.pdf", "label": "Circle"}],
        "render": {"image_size": [64, 64]},
        "layout": {"dpi": 100, "width_inches": 4, "row_height_inches": 2},
        "output": {"path": "figure.png"},
    }

    output, preview = generate_from_config(config, tmp_path)

    assert output == tmp_path / "figure.png"
    assert preview is None
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
