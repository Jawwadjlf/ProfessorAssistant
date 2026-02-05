# Custom hook for webrtcvad to replace the broken one from pyinstaller-hooks-contrib
# This hook simply collects the necessary binary files without trying to copy metadata

from PyInstaller.utils.hooks import collect_dynamic_libs

# Collect the binary/shared library files for webrtcvad
binaries = collect_dynamic_libs('webrtcvad')

# Include the internal _webrtcvad module
hiddenimports = ['_webrtcvad']
