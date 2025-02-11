# -*- mode: python ; coding: utf-8 -*-
# macOS, create DMG with:
#   hdiutil create -fs HFS+ -srcfolder dist/LiveFT.app -volname LiveFT-xy LiveFT-xy.dmg

from platform import system

a = Analysis(
    ['LiveFT.py'],
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
    a.binaries,
    a.datas,
    [],
    name='LiveFT',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll, app = None, None
if system().startswith("Darwin"):
    # macOS specific config
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='LiveFT',
    )
    app = BUNDLE(
        coll,
        name='LiveFT.app',
        icon=None,
        bundle_identifier=None,
        info_plist={
            'NSCameraUsageDescription': 'Pleease! ^_^',
            'NSPrincipalClass': 'NSApplication',
            'NSAppleScriptEnabled': False,
        },
    )
