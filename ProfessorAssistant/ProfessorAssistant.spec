# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

# Collect all openwakeword data files (includes melspectrogram.onnx and other internal models)
openwakeword_datas = collect_data_files('openwakeword')
openwakeword_binaries = collect_dynamic_libs('openwakeword')

a = Analysis(
    ['assistant.py'],
    pathex=[],
    binaries=openwakeword_binaries,
    datas=[('models', 'models')] + openwakeword_datas,
    hiddenimports=[
        'openwakeword',
        'openwakeword.utils',
        'openwakeword.model',
        'paho.mqtt.client',
        'google.generativeai',
        'webrtcvad',
        '_webrtcvad',
        'onnxruntime',
        'onnxruntime.capi',
        'onnxruntime.capi.onnxruntime_pybind11_state'
    ],
    hookspath=['hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
