

you must use the x64 (64-bit) Native Tools Command Prompt for Visual Studio 2022 to ensure that the C code is compiled and linked as a 64-bit executable
..\..\.venv\Scripts\activate
`..\..\.venv\Scripts\python setup.py bdist_wheel`
cd dist
pip install numextension-1.0.0-cp311-cp311-win_amd64.whl


# Clean previous build artifacts
rm -rf build/ dist/ *.egg-info/

# Build again
python -m build --wheel


pip install -e .


