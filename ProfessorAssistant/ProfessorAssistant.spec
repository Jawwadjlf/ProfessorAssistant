# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['assistant.py'],
    pathex=[],
    binaries=[],
    datas=[('models', 'models')],
    hiddenimports=[
        'openwakeword',
        'paho.mqtt.client',
        'google.generativeai',
        'webrtcvad',
        '_webrtcvad'
    ],
    hookspath=[],
    hooksconfig={
        'webrtcvad': {'enabled': False}  # Disable the problematic hook
    },
    runtime_hooks=[],
    excludes=[
        '_pyinstaller_hooks_contrib.hooks.stdhooks.hook-webrtcvad'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ProfessorAssistant',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Disable UPX to avoid issues with binary extensions
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
