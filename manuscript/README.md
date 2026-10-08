# Reproducible manuscript figures

`liveft-figure` renders the original demonstration PDFs and passes the result
through the numerical processing shared with the live camera application.  It
does not contain a second FFT implementation.

## Installation and prototype

Install the optional PDF support and generate the four-row prototype from the
repository root:

```bash
uv sync --extra manuscript
uv run --extra manuscript liveft-figure --config manuscript/prototype.json
```

This exact command writes:

- `manuscript/generated/liveft_prototype.pdf`
- `manuscript/generated/liveft_prototype.png`

Generated files are ignored because they are reproducible; the JSON
configuration and source patterns are the publication inputs.

## Selecting patterns and fields of view

For a reproducible sequence, copy `prototype.json` and edit its `patterns`
list. PDF and common OpenCV-readable raster images are accepted. Each entry
accepts `source`, `label`, zero-based PDF `page`, and an optional
normalized `crop` (`[x, y, width, height]`, with the page spanning 0 to 1).
Shared render settings select PDF DPI, output pixel dimensions, and either the
full page or a centered square field of view.  A crop on a pattern overrides
the shared field of view.  Images are fitted without changing aspect ratio.

The same settings are available directly on the command line:

```bash
uv run --extra manuscript liveft-figure \
  images/singleObjects/smallSphere.pdf \
  images/singleObjects/largeSphere.pdf \
  --labels "Small circle" "Large circle" \
  --render-dpi 180 --field-of-view center-square --image-size 768,768 \
  --window --taper-width 0.2 \
  --representation power --display-scaling log --gamma 1.0 \
  --no-suppress-center-lines \
  --output manuscript/generated/circles.pdf \
  --png-preview manuscript/generated/circles.png
```

Use `--crop X,Y,WIDTH,HEIGHT` to set one normalized crop in direct mode.  Run
`uv run --extra manuscript liveft-figure --help` for all layout and processing
options.

## Numerical method and interpretation

The shared `process_image(image, config)` interface separates two products:

1. `spectrum` is the centered numerical magnitude or power spectrum.
2. `display_spectrum` applies linear or `log1p` scaling, optional center-line
   replacement, max normalization, and gamma for display.

LiveFT's established default is the **power spectrum** (`real² + imag²`), not
FFT magnitude.  It is centered with `fftshift`, transformed with `log1p`, and
max-normalized.  The prototype names these choices explicitly.  Gamma and
center-line suppression change only `display_spectrum`; they do not alter the
returned numerical spectrum.  Center-line suppression is disabled in the
prototype so the zero-frequency neighborhood is visible.

Before the transform, LiveFT converts OpenCV BGR input to grayscale, applies a
separable error-function vignette with taper width 0.2, subtracts the minimum,
and max-normalizes.  The prototype uses the same operations and displays that
processed real-space input.  The vignette reduces hard field-of-view edges but
also broadens reciprocal-space features slightly.  Disable it explicitly with
`"window": false` or `--no-window` when investigating boundary effects.

For an actual magnitude representation, select `magnitude`; for an actual
power spectrum select `power`.  Display scaling remains a separate choice:
`linear` preserves proportional display intensity after normalization, while
`log` reveals weak features.  A finite sampled transform, page crop, white
background (DC signal), rasterization, and window all leave recognizable
signatures.  Keep these settings fixed when comparing size, rotation,
complement, or periodicity.

## Suggested manuscript description

> Demonstration-pattern PDFs were rasterized at 180 dpi and cropped to a
> consistent centered square field of view. The images were processed with the
> same grayscale conversion, error-function edge window (relative taper width
> 0.2), and OpenCV discrete Fourier-transform implementation used by LiveFT.
> The displayed reciprocal-space panels show the centered power spectrum after
> `log(1 + I)` scaling and max normalization; display gamma was 1.0 and central
> line suppression was disabled.

Adapt the statement if the checked-in configuration changes.  In particular,
do not describe `power` as magnitude or `log` as a physical intensity change.
