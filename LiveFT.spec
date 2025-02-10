# -*- mode: python ; coding: utf-8 -*-
# macOS, create DMG with:
#   hdiutil create -fs HFS+ -srcfolder dist/LiveFT.app -volname LiveFT-xy LiveFT-xy.dmg

a = Analysis(
    ['LiveFT.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'sqlite3', 'PIL', 'pandas', 'networkx', 'torch.onnx', 'sympy'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
# to avoid pink borders, the splash image needs to be filtered, as in here:
#  https://github.com/pyinstaller/pyinstaller/issues/8579#issuecomment-2226981506
splash = Splash(
    'splash.png',
    binaries=a.binaries,
    datas=a.datas,
    text_pos=(10,50),
    text_size=12,
    text_color='darkgray',
    minify_script=True,
    always_on_top=True,
)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    splash,
    splash.binaries,
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
        'NSCameraUsageDescription': 'Pleease! ^_^'
    },
)
