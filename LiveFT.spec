# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path
from platform import system

IS_DARWIN = system().startswith("Darwin")
APP_NAME = os.getenv("LIVEFT_APP_NAME", "LiveFT")
BUNDLE_IDENTIFIER = os.getenv("LIVEFT_BUNDLE_ID", "nl.stack.liveft")
CAMERA_USAGE_DESCRIPTION = os.getenv(
    "LIVEFT_CAMERA_USAGE",
    "LiveFT needs camera access to capture the live image used for Fourier-transform visualization.",
)
ICON_PATH = os.getenv("LIVEFT_ICON_PATH", "").strip() or None
CODE_SIGN_IDENTITY = os.getenv("LIVEFT_CODESIGN_IDENTITY", "").strip() or None
ENTITLEMENTS_FILE = os.getenv("LIVEFT_ENTITLEMENTS_FILE", "").strip() or None

if ICON_PATH is not None and not Path(ICON_PATH).is_file():
    raise FileNotFoundError(f"Icon file not found: {ICON_PATH}")

if ENTITLEMENTS_FILE is not None and not Path(ENTITLEMENTS_FILE).is_file():
    raise FileNotFoundError(f"Entitlements file not found: {ENTITLEMENTS_FILE}")

a = Analysis(
    ["LiveFT.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    *(([]) if IS_DARWIN else (a.binaries, a.datas, [])),
    exclude_binaries=IS_DARWIN,  # has to be False for Windows, it seems
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    # https://developer.apple.com/forums/thread/691986
    codesign_identity=CODE_SIGN_IDENTITY if IS_DARWIN else None,
    entitlements_file=ENTITLEMENTS_FILE if IS_DARWIN and CODE_SIGN_IDENTITY else None,
)
coll, app = None, None
if IS_DARWIN:
    # macOS specific config
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name=APP_NAME,
    )
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        icon=ICON_PATH,
        bundle_identifier=BUNDLE_IDENTIFIER,
        info_plist={
            "NSCameraUsageDescription": CAMERA_USAGE_DESCRIPTION,
            "NSPrincipalClass": "NSApplication",
            "NSAppleScriptEnabled": False,
        },
    )
