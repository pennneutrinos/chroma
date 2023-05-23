# Chroma in geant4_pybind

Chroma traditionally builds additional objects for geant4 that would be included using
Boost. Switching to pybind11 does not carry over everything needed to avoid circular
dependencies and collissions -- so chroma will be injected directly.

## Modifications
- Edit `geant4_pybind.cc` to add `export_Chroma` function
- Edit `CMakeLists.txt` to add chroma/*.cc

Likely need to edit setup.py to include rat-pac library and headers.
