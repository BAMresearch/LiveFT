# LiveFT Upgrade Plan

## Goal

Capture the highest-value improvements for this repository, then execute them in a safe order.
This plan focuses on:

- a small, fast test suite
- code structure and maintainability
- packaging and repository hygiene
- macOS build, signing, and release readiness

## Current Baseline

### What is already in place

- Main app entry point: `LiveFT.py`
- Existing image-regression test file: `test_LiveFT.py`
- Existing PyInstaller spec: `LiveFT.spec`
- Existing GitHub Actions build pipeline: `.github/workflows/build.yml`
- Existing macOS entitlements file: `CodeSigning/entitlements.xml`
- Existing built macOS app bundle in `binaries/`

### What I verified locally

- `python -m py_compile LiveFT.py test_LiveFT.py` passes.
- `pytest` does not currently collect safely with the checked-in `pytest.ini`.
- Running only `test_LiveFT.py` shows the baseline image tests are partially failing on this machine.
- The checked-in macOS app bundle is ad-hoc signed, not release signed.

### Concrete issues found

1. Test collection is fragile.
   `pytest.ini` enabled `--doctest-modules`, which caused pytest to import every Python module in the repo. That pulled in an experimental PyQt prototype, which imports `torch` at module import time and aborted collection on this machine.

2. The existing test suite is slow and brittle.
   `test_LiveFT.py` is a PDF-to-image regression suite. It is useful, but it is not a good "small fast suite" because:
   - it depends on PyMuPDF rendering
   - it compares generated FFT images against exact-ish baselines
   - 11 of 19 cases currently fail here due to image diffs

3. The app is hard to test because construction has side effects.
   `LiveFT.__attrs_post_init__()` opens the camera, creates a window, and immediately starts the main loop. That makes unit testing and reuse harder than necessary.

4. Some flags and docs no longer match the implementation.
   - `README.md` still needs to match the current app behavior as features evolve.

5. The release metadata is not ready for public macOS distribution.
   In `LiveFT.spec`:
   - `codesign_identity=None`
   - `entitlements_file=None`
   - `bundle_identifier=None`
   - `NSCameraUsageDescription` is placeholder text

6. The current GitHub Actions workflow builds artifacts but does not protect release quality.
   It packages and publishes, but it does not run tests before release, does not sign, and does not notarize.

7. Repository hygiene can be improved.
   The repo currently includes built artifacts and local clutter such as `.DS_Store`, plus prototype code under `experiments/` that should not be treated as release targets.

## Proposed Upgrade Order

### Phase 1: Establish a real baseline

- [ ] Fix `pytest` collection so the default test command runs reliably.
- [ ] Split tests into:
  - a small fast unit suite that runs on every change
  - a slower image-regression suite that can be optional or separately marked
- [ ] Update README and developer docs so setup, testing, and runtime behavior match the actual code.

Why first:
This gives us a safe place to refactor from. Right now the repository can build artifacts without proving the code is healthy.

### Phase 2: Add the small fast test suite

- [ ] Add unit tests for `FrameProcessor.prepareFrame()`.
  Cover crop, scale, and grayscale conversion.
- [ ] Add unit tests for `FrameProcessor.applyWindow()`.
  Cover shape, normalization bounds, and window caching behavior.
- [ ] Add unit tests for `FrameProcessor.computeFFT()`.
  Cover output shape, normalization, and `killCenterLines`.
- [ ] Add unit tests for argument parsing.
  Cover boolean toggles and default values from `parse_args()`.
- [ ] Mark the PDF/image comparisons as `slow` or `regression`.

Expected result:
`pytest` should finish quickly for day-to-day work and catch obvious breakage without requiring a camera or PDF rendering.

### Phase 3: Refactor for testability and clarity

- [ ] Move side-effectful startup out of `LiveFT.__attrs_post_init__()`.
- [ ] Add a `main()` entry point that wires together:
  - argument parsing
  - camera/window initialization
  - run loop
- [ ] Keep `FrameProcessor` pure and easy to test.
- [ ] Introduce a small camera abstraction or wrapper for `cv2.VideoCapture`.
- [ ] Add guards against divide-by-zero in normalization paths.

Expected result:
Cleaner code boundaries, easier tests, fewer hidden side effects.

### Phase 4: User-facing feature improvements

- [ ] Add a frame-rate limiter so processing can be capped at 25 FPS instead of always chasing camera speed.
- [ ] Expose the frame-rate limiter as configuration.
  Prefer at least a command-line option, with runtime control if the UX stays simple.
- [ ] Add an optional radial distribution plot derived from the FFT center and render it underneath the source and FFT images.
- [ ] Add interactive gamma control for the FFT display with the `+` and `-` keys.
- [ ] Add tests for the new display-state logic where practical, especially gamma adjustment and frame-rate limit configuration.

Expected result:
The app becomes more usable during demos and lectures, and the FFT visualization becomes easier to tune live.

### Phase 5: Packaging and repo cleanup

- [ ] Keep the PyQt prototypes under `experiments/` and out of the supported app entry points.
- [ ] Exclude `experiments/` from the default test/discovery/lint path unless work is explicitly happening there.
- [ ] Stop checking built app bundles into the main source tree.
- [ ] Extend `.gitignore` for:
  - `binaries/`
  - macOS Finder files
  - build output directories
- [ ] Consolidate developer dependencies.
  Prefer either:
  - `requirements-dev.txt`, or
  - a `pyproject.toml` with optional test/build dependencies

Expected result:
A cleaner repo and fewer accidental release or collection problems.

### Phase 6: macOS release pipeline

- [ ] Update `LiveFT.spec` for proper app metadata:
  - bundle identifier
  - version metadata
  - icon
  - non-placeholder camera permission text
  - signing identity and entitlements wiring
- [ ] Replace the current `sed`-based spec mutation in CI with a dedicated build script.
- [ ] Build a signed `.app`.
- [ ] Notarize the app or the DMG.
- [ ] Staple notarization tickets.
- [ ] Verify the final artifact with Apple tooling before release.

Expected result:
A distributable macOS artifact that Gatekeeper accepts on another machine.

## Recommended First Deliverables

### Step 1

Make the default test command sane:

- remove or narrow `--doctest-modules`
- add a tiny fast unit suite for `FrameProcessor`
- keep the existing PDF baselines, but mark them separately

### Step 2

Refactor `LiveFT` startup so we can test logic without opening a real camera or GUI window.

### Step 3

Add the demo-oriented UI features:

- 25 FPS processing cap
- optional radial distribution panel
- `+`/`-` gamma control for the FFT image

### Step 4

Clean up packaging metadata and add a repeatable macOS release script.

## macOS Build and Signing Plan

### What is missing today

The current bundled app is only ad-hoc signed. That is enough for some local testing, but not for public release.

To release on macOS, we need:

- an Apple Developer account
- a `Developer ID Application` certificate in the macOS keychain
- a real bundle identifier, for example `nl.stack.liveft`
- notarization credentials
  - either Apple ID + app-specific password + team ID
  - or App Store Connect API key credentials

### Release prerequisites

- [ ] Choose bundle identifier
- [ ] Create app icon (`.icns`) if needed
- [ ] Replace placeholder camera usage string
- [ ] Confirm entitlements are minimal and correct
- [ ] Decide whether signing/notarization will happen:
  - only locally on a trusted Mac, or
  - in GitHub Actions with secrets

### Suggested local release flow

1. Create a clean environment and install runtime/build dependencies.
2. Build the app with PyInstaller.
3. Sign the app bundle and nested binaries with `Developer ID Application`.
4. Create a DMG.
5. Submit for notarization with `xcrun notarytool`.
6. Staple the notarization ticket.
7. Verify with `codesign` and `spctl`.

### Suggested verification commands

```bash
codesign --verify --deep --strict --verbose=2 dist/LiveFT.app
spctl -a -vvv dist/LiveFT.app
xcrun stapler validate dist/LiveFT.dmg
```

### Suggested automation targets

- [ ] `scripts/build_macos.sh`
- [ ] `scripts/sign_macos.sh`
- [ ] `scripts/notarize_macos.sh`
- [ ] `scripts/release_macos.sh`

Those scripts should avoid editing `LiveFT.spec` in place and should take version, signing identity, bundle ID, and output paths through environment variables or CLI args.

## Immediate Risks To Address

- The current release workflow can publish artifacts without running tests first.
- The default pytest configuration is not safe.
- The current regression suite is not stable enough to be the only test layer.
- The macOS app metadata and signing configuration are not release-ready.

## Proposed Working Sequence

1. Fix pytest collection and add the small fast unit suite.
2. Refactor startup and separate pure logic from UI/camera side effects.
3. Implement the user-facing visualization controls and the optional radial distribution panel.
4. Clean up repo structure and packaging inputs.
5. Implement the macOS build/sign/notarize flow.
6. Re-enable or tighten CI gates so releases only happen from green builds.
