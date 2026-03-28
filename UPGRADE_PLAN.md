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
- Fast default unit suite: `tests/test_liveft_unit.py`
- Explicit image-regression suite: `test_LiveFT.py`
- PyInstaller spec and helper scripts for build, DMG creation, signing, notarization, and verification
- GitHub Actions build pipeline: `.github/workflows/build.yml`
- Existing macOS entitlements file: `CodeSigning/entitlements.xml`
- Experimental prototypes isolated under `experiments/`

### What I verified locally

- `pytest -q` passes with the fast unit suite.
- `ruff check .`, `pre-commit run --all-files`, and script syntax checks pass.
- The app lifecycle is explicit and side-effect free at construction time.
- A local macOS smoke build can produce an unsigned `.app` and `.dmg`.
- The macOS workflow can optionally sign and notarize when Apple credentials are provided.

### Remaining concrete issues

1. The slow regression suite is still brittle.
   `test_LiveFT.py` remains useful as a regression backstop, but it is still not a stable default gate for day-to-day work.

2. The camera layer is still fairly direct.
   `cv2.VideoCapture` is no longer created during construction, but there is still no dedicated camera abstraction or mock-friendly wrapper.

3. macOS release metadata is still incomplete.
   The spec now supports bundle identifier, signing identity, entitlements, and camera usage text, but version metadata and a real app icon are still missing.

4. Cross-platform release polish is still open.
   Linux and Windows builds already exist in CI, but they are still unsigned and their release packaging can be improved.

5. A real signed/notarized macOS release has not been exercised yet.
   The scripts and workflow are ready, but they still need an end-to-end run with actual Apple credentials.

## Proposed Upgrade Order

### Phase 1: Establish a real baseline

- [x] Fix `pytest` collection so the default test command runs reliably.
- [x] Split tests into:
  - a small fast unit suite that runs on every change
  - a slower image-regression suite that can be optional or separately marked
- [x] Update README and developer docs so setup, testing, and runtime behavior match the actual code.

Why first:
This gives us a safe place to refactor from. Right now the repository can build artifacts without proving the code is healthy.

### Phase 2: Add the small fast test suite

- [x] Add unit tests for `FrameProcessor.prepareFrame()`.
  Cover crop, scale, and grayscale conversion.
- [x] Add unit tests for `FrameProcessor.applyWindow()`.
  Cover shape, normalization bounds, and window caching behavior.
- [x] Add unit tests for `FrameProcessor.computeFFT()`.
  Cover output shape, normalization, and `killCenterLines`.
- [x] Add unit tests for argument parsing.
  Cover boolean toggles and default values from `parse_args()`.
- [x] Mark the PDF/image comparisons as `slow` or `regression`.

Expected result:
`pytest` should finish quickly for day-to-day work and catch obvious breakage without requiring a camera or PDF rendering.

### Phase 3: Refactor for testability and clarity

- [x] Move side-effectful startup out of `LiveFT.__attrs_post_init__()`.
- [x] Add a `main()` entry point that wires together:
  - argument parsing
  - camera/window initialization
  - run loop
- [x] Keep `FrameProcessor` pure and easy to test.
- [ ] Introduce a small camera abstraction or wrapper for `cv2.VideoCapture`.
- [x] Add guards against divide-by-zero in normalization paths.

Expected result:
Cleaner code boundaries, easier tests, fewer hidden side effects.

### Phase 4: User-facing feature improvements

- [x] Add a frame-rate limiter so processing can be capped at 25 FPS instead of always chasing camera speed.
- [x] Expose the frame-rate limiter as configuration.
  Prefer at least a command-line option, with runtime control if the UX stays simple.
- [x] Add an optional radial distribution plot derived from the FFT center and render it underneath the source and FFT images.
- [x] Add interactive gamma control for the FFT display with the `+` and `-` keys.
- [x] Add tests for the new display-state logic where practical, especially gamma adjustment and frame-rate limit configuration.

Expected result:
The app becomes more usable during demos and lectures, and the FFT visualization becomes easier to tune live.

### Phase 5: Packaging and repo cleanup

- [x] Keep the PyQt prototypes under `experiments/` and out of the supported app entry points.
- [x] Exclude `experiments/` from the default test/discovery/lint path unless work is explicitly happening there.
- [x] Stop checking built app bundles into the main source tree.
- [x] Extend `.gitignore` for:
  - `binaries/`
  - macOS Finder files
  - build output directories
- [x] Consolidate developer dependencies.
  Prefer either:
  - `requirements-dev.txt`, or
  - a `pyproject.toml` with optional test/build dependencies

Expected result:
A cleaner repo and fewer accidental release or collection problems.

### Phase 6: macOS release pipeline

- [ ] Update `LiveFT.spec` for proper app metadata:
  - [x] bundle identifier
  - [ ] version metadata
  - [ ] icon
  - [x] non-placeholder camera permission text
  - [x] signing identity and entitlements wiring
- [x] Replace the current `sed`-based spec mutation in CI with a dedicated build script.
- [x] Add reusable macOS signing, notarization, verification, and release scripts.
- [x] Wire optional macOS signing and notarization secrets into the GitHub Actions workflow.
- [ ] Build a signed `.app`.
- [ ] Notarize the app or the DMG.
- [ ] Staple notarization tickets.
- [ ] Verify the final artifact with Apple tooling before release.

Expected result:
A distributable macOS artifact that Gatekeeper accepts on another machine.

### Phase 7: Cross-platform release polish

- [x] Build Linux and Windows artifacts in CI.
- [ ] Decide on the preferred Linux release format.
  Candidates: raw binary, `.tar.gz`, or AppImage.
- [ ] Decide on the preferred Windows release format.
  Candidates: raw `.exe` or zipped release bundle.
- [ ] Add post-build packaging so Linux and Windows release assets are easier to download and use.
- [ ] Decide whether Windows signing is worth doing now or can wait.
  Unsigned Windows binaries are acceptable for internal/testing use, but SmartScreen warnings are expected.

Expected result:
Cleaner Linux and Windows release assets without blocking on a Windows signing certificate.

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

- [x] Choose bundle identifier
- [ ] Create app icon (`.icns`) if needed
- [x] Replace placeholder camera usage string
- [x] Confirm entitlements are minimal and correct
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
- [x] `scripts/sign_macos.sh`
- [x] `scripts/notarize_macos.sh`
- [x] `scripts/release_macos.sh`
- [x] `scripts/verify_macos.sh`

Those scripts should avoid editing `LiveFT.spec` in place and should take version, signing identity, bundle ID, and output paths through environment variables or CLI args.

## Immediate Risks To Address

- The slow regression suite is still not stable enough to become the default gate.
- The macOS app still needs real icon/version metadata before it is release-finished.
- The macOS workflow still needs one real signed/notarized run with Apple credentials.
- Linux and Windows release assets are functional, but not yet polished or signed.

## Proposed Working Sequence

1. Finish macOS release metadata by adding version info and a real app icon.
2. Run one real signed/notarized macOS build through the workflow with Apple credentials.
3. Improve Linux and Windows release packaging in CI.
4. Decide whether Windows signing should be postponed or added later.
5. Return to remaining code-level cleanup, especially the camera abstraction and regression-suite stability.
