"""Headless, reproducible manuscript figure generation for LiveFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from liveft_processing import ProcessingConfig, process_image

DEFAULT_RENDER_DPI = 180
DEFAULT_IMAGE_SIZE = (768, 768)
DEFAULT_FIGURE_DPI = 300


def _pymupdf():
    try:
        import pymupdf
    except ImportError as error:
        raise RuntimeError("PDF support requires the manuscript extra; run `uv sync --extra manuscript`.") from error
    return pymupdf


def parse_pair(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.split(","))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("expected WIDTH,HEIGHT") from error
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("dimensions must be positive")
    return width, height


def parse_crop(value: str) -> tuple[float, float, float, float]:
    try:
        crop = tuple(float(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected X,Y,WIDTH,HEIGHT as fractions") from error
    if len(crop) != 4:
        raise argparse.ArgumentTypeError("expected X,Y,WIDTH,HEIGHT as fractions")
    x, y, width, height = crop
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
        raise argparse.ArgumentTypeError("crop must lie within normalized page coordinates [0, 1]")
    return x, y, width, height


def render_pdf_page(path: Path, page_number: int = 0, dpi: int = DEFAULT_RENDER_DPI) -> np.ndarray:
    """Render one PDF page as a BGR uint8 array."""
    if dpi <= 0:
        raise ValueError("render DPI must be positive")
    pymupdf = _pymupdf()
    with pymupdf.open(path) as document:
        if not 0 <= page_number < len(document):
            raise ValueError(f"page {page_number} is outside {path} ({len(document)} pages)")
        pixmap = document[page_number].get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
        rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def load_source_image(path: Path, page_number: int = 0, dpi: int = DEFAULT_RENDER_DPI) -> np.ndarray:
    """Load a PDF page or a raster image as a BGR uint8 array."""
    if path.suffix.lower() == ".pdf":
        return render_pdf_page(path, page_number, dpi)
    if page_number != 0:
        raise ValueError("page selection is only valid for PDF sources")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read image source {path}")
    return image


def select_field_of_view(
    image: np.ndarray,
    mode: str = "center-square",
    crop: Sequence[float] | None = None,
) -> np.ndarray:
    """Select a page region without changing its geometry."""
    height, width = image.shape[:2]
    if crop is not None:
        if len(crop) != 4:
            raise ValueError("normalized crop must contain x, y, width and height")
        x, y, crop_width, crop_height = crop
        left = round(float(x) * width)
        top = round(float(y) * height)
        right = round(float(x + crop_width) * width)
        bottom = round(float(y + crop_height) * height)
        if left < 0 or top < 0 or right > width or bottom > height or right <= left or bottom <= top:
            raise ValueError("normalized crop lies outside the rendered page")
        return image[top:bottom, left:right]
    if mode == "full":
        return image
    if mode == "center-square":
        side = min(height, width)
        left = (width - side) // 2
        top = (height - side) // 2
        return image[top : top + side, left : left + side]
    raise ValueError("field_of_view must be 'full' or 'center-square'")


def fit_to_canvas(image: np.ndarray, size: tuple[int, int], fill: int = 255) -> np.ndarray:
    """Resize into a fixed canvas while preserving aspect ratio."""
    target_width, target_height = size
    height, width = image.shape[:2]
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    shape = (target_height, target_width) if image.ndim == 2 else (target_height, target_width, image.shape[2])
    canvas = np.full(shape, fill, dtype=image.dtype)
    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def _display_u8(image: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def process_pattern(
    path: Path,
    page: int,
    render_dpi: int,
    field_of_view: str,
    crop: Sequence[float] | None,
    image_size: tuple[int, int],
    processing: ProcessingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    rendered = load_source_image(path, page, render_dpi)
    selected = select_field_of_view(rendered, field_of_view, crop)
    source = fit_to_canvas(selected, image_size)
    result = process_image(source, processing)
    return _display_u8(result.processed_image), _display_u8(result.display_spectrum)


def compose_figure(
    rows: Sequence[tuple[str, np.ndarray, np.ndarray]],
    figure_dpi: int = DEFAULT_FIGURE_DPI,
    width_inches: float = 7.2,
    row_height_inches: float = 2.0,
) -> np.ndarray:
    """Compose labeled real/Fourier-space pairs into a journal-style raster."""
    if not rows:
        raise ValueError("at least one pattern is required")
    if figure_dpi <= 0 or width_inches <= 0 or row_height_inches <= 0:
        raise ValueError("figure dimensions and DPI must be positive")

    width = round(width_inches * figure_dpi)
    margin = round(0.22 * figure_dpi)
    column_gap = round(0.18 * figure_dpi)
    header_height = round(0.34 * figure_dpi)
    row_height = round(row_height_inches * figure_dpi)
    height = margin * 2 + header_height + len(rows) * row_height
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    cell_width = (width - 2 * margin - column_gap) // 2
    title_scale = figure_dpi / 300 * 0.75
    label_scale = figure_dpi / 300 * 0.60
    thickness = max(1, round(figure_dpi / 180))

    headings = ("Real space", "Fourier space")
    for column, heading in enumerate(headings):
        x0 = margin + column * (cell_width + column_gap)
        text_size = cv2.getTextSize(heading, cv2.FONT_HERSHEY_SIMPLEX, title_scale, thickness)[0]
        x = x0 + (cell_width - text_size[0]) // 2
        cv2.putText(
            canvas,
            heading,
            (x, margin + text_size[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            title_scale,
            (30, 30, 30),
            thickness,
            cv2.LINE_AA,
        )

    panel_letters = iter("abcdefghijklmnopqrstuvwxyz")
    for row_index, (label, real_image, fourier_image) in enumerate(rows):
        row_top = margin + header_height + row_index * row_height
        label_height = round(0.25 * figure_dpi)
        available_height = row_height - label_height - round(0.08 * figure_dpi)
        for column, image in enumerate((real_image, fourier_image)):
            x0 = margin + column * (cell_width + column_gap)
            panel = fit_to_canvas(image, (cell_width, available_height), fill=255)
            y0 = row_top + label_height
            if panel.ndim == 2:
                panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
            canvas[y0 : y0 + available_height, x0 : x0 + cell_width] = panel
            cv2.rectangle(canvas, (x0, y0), (x0 + cell_width - 1, y0 + available_height - 1), (170, 170, 170), 1)
            panel_label = f"({next(panel_letters)}) {label if column == 0 else 'Fourier space'}"
            cv2.putText(
                canvas,
                panel_label,
                (x0, row_top + round(0.18 * figure_dpi)),
                cv2.FONT_HERSHEY_SIMPLEX,
                label_scale,
                (30, 30, 30),
                thickness,
                cv2.LINE_AA,
            )
    return canvas


def export_figure(figure: np.ndarray, output: Path, figure_dpi: int, png_preview: Path | None = None) -> None:
    """Export a composed figure to PDF or PNG, plus an optional PNG preview."""
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".png":
        if not cv2.imwrite(str(output), figure):
            raise OSError(f"could not write {output}")
    elif suffix == ".pdf":
        pymupdf = _pymupdf()
        success, encoded = cv2.imencode(".png", figure)
        if not success:
            raise OSError("could not encode figure for PDF export")
        document = pymupdf.open()
        width_points = figure.shape[1] / figure_dpi * 72
        height_points = figure.shape[0] / figure_dpi * 72
        page = document.new_page(width=width_points, height=height_points)
        page.insert_image(page.rect, stream=encoded.tobytes())
        document.save(output, garbage=4, deflate=True, no_new_id=True, reproducible=True)
        document.close()
    else:
        raise ValueError("output filename must end in .pdf or .png")

    if png_preview is not None and png_preview != output:
        png_preview.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(png_preview), figure):
            raise OSError(f"could not write {png_preview}")


def _resolved_path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config.get("patterns"), list) or not config["patterns"]:
        raise ValueError("configuration must contain a non-empty 'patterns' list")
    return config


def generate_from_config(
    config: dict[str, Any],
    base: Path,
    output_override: Path | None = None,
    png_override: Path | None = None,
) -> tuple[Path, Path | None]:
    render = config.get("render", {})
    processing_values = config.get("processing", {})
    layout = config.get("layout", {})
    output_values = config.get("output", {})
    image_size = tuple(render.get("image_size", DEFAULT_IMAGE_SIZE))
    processing = ProcessingConfig(
        window=processing_values.get("window", True),
        taper_width=processing_values.get("taper_width", 0.2),
        representation=processing_values.get("representation", "power"),
        display_scaling=processing_values.get("display_scaling", "log"),
        gamma=processing_values.get("gamma", 1.0),
        suppress_center_lines=processing_values.get("suppress_center_lines", False),
    )
    rows = []
    for pattern in config["patterns"]:
        source = _resolved_path(pattern["source"], base)
        label = pattern.get("label", source.stem)
        real, fourier = process_pattern(
            source,
            page=pattern.get("page", render.get("page", 0)),
            render_dpi=render.get("dpi", DEFAULT_RENDER_DPI),
            field_of_view=pattern.get("field_of_view", render.get("field_of_view", "center-square")),
            crop=pattern.get("crop", render.get("crop")),
            image_size=(int(image_size[0]), int(image_size[1])),
            processing=processing,
        )
        rows.append((label, real, fourier))

    figure_dpi = int(layout.get("dpi", DEFAULT_FIGURE_DPI))
    figure = compose_figure(
        rows,
        figure_dpi=figure_dpi,
        width_inches=float(layout.get("width_inches", 7.2)),
        row_height_inches=float(layout.get("row_height_inches", 2.0)),
    )
    output = output_override or _resolved_path(output_values.get("path", "liveft_figure.pdf"), base)
    png_value = png_override if png_override is not None else output_values.get("png_preview")
    png = _resolved_path(png_value, base) if png_value else None
    export_figure(figure, output, figure_dpi, png)
    return output, png


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate paired real/Fourier-space manuscript panels.")
    parser.add_argument("sources", nargs="*", type=Path, help="Source PDF or raster-image files (direct mode)")
    parser.add_argument("--config", type=Path, help="JSON configuration for a reproducible multi-pattern figure")
    parser.add_argument("--output", type=Path, help="Output .pdf or .png filename")
    parser.add_argument("--png-preview", type=Path, help="Optional additional PNG filename")
    parser.add_argument("--labels", nargs="*", help="Labels corresponding to direct-mode sources")
    parser.add_argument("--page", type=int, default=0, help="Zero-based PDF page number")
    parser.add_argument("--render-dpi", type=int, default=DEFAULT_RENDER_DPI)
    parser.add_argument("--field-of-view", choices=("full", "center-square"), default="center-square")
    parser.add_argument("--crop", type=parse_crop, help="Normalized X,Y,WIDTH,HEIGHT page crop")
    parser.add_argument("--image-size", type=parse_pair, default=DEFAULT_IMAGE_SIZE, help="WIDTH,HEIGHT pixels")
    parser.add_argument("--window", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--taper-width", type=float, default=0.2)
    parser.add_argument("--representation", choices=("magnitude", "power"), default="power")
    parser.add_argument("--display-scaling", choices=("linear", "log"), default="log")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--suppress-center-lines", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--figure-dpi", type=int, default=DEFAULT_FIGURE_DPI)
    parser.add_argument("--figure-width", type=float, default=7.2, help="Figure width in inches")
    parser.add_argument("--row-height", type=float, default=2.0, help="Height per pair in inches")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.config:
        if args.sources:
            raise SystemExit("source arguments cannot be combined with --config")
        config_path = args.config.resolve()
        output, png = generate_from_config(
            load_config(config_path),
            config_path.parent,
            output_override=args.output,
            png_override=args.png_preview,
        )
    else:
        if not args.sources:
            raise SystemExit("provide PDF sources or --config")
        if args.output is None:
            raise SystemExit("direct mode requires --output")
        if args.labels is not None and len(args.labels) != len(args.sources):
            raise SystemExit("--labels must provide one label per source")
        labels = args.labels or [source.stem for source in args.sources]
        processing = ProcessingConfig(
            window=args.window,
            taper_width=args.taper_width,
            representation=args.representation,
            display_scaling=args.display_scaling,
            gamma=args.gamma,
            suppress_center_lines=args.suppress_center_lines,
        )
        rows = []
        for source, label in zip(args.sources, labels):
            real, fourier = process_pattern(
                source,
                args.page,
                args.render_dpi,
                args.field_of_view,
                args.crop,
                args.image_size,
                processing,
            )
            rows.append((label, real, fourier))
        figure = compose_figure(rows, args.figure_dpi, args.figure_width, args.row_height)
        export_figure(figure, args.output, args.figure_dpi, args.png_preview)
        output, png = args.output, args.png_preview

    print(f"Wrote {output}")
    if png is not None:
        print(f"Wrote {png}")


if __name__ == "__main__":
    main()
