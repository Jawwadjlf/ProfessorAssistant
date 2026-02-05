# Custom hook for openwakeword to ensure all resources are collected
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

# Collect all data files from openwakeword (includes models and resources)
datas = collect_data_files('openwakeword', include_py_files=False)

# Collect any dynamic libraries
binaries = collect_dynamic_libs('openwakeword')

# Collect all submodules
hiddenimports = collect_submodules('openwakeword')

# Add onnxruntime dependencies that openwakeword needs
hiddenimports += [
    'onnxruntime',
    'onnxruntime.capi',
    'onnxruntime.capi.onnxruntime_pybind11_state',
    'onnxruntime.capi._pybind_state'
]
