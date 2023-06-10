#!/bin/bash

# Uninstall the package
pip uninstall nlu-inference-agl

# Build the wheel distribution
python setup.py bdist_wheel

# Install the generated wheel package
pip install dist/nlu_inference_agl-0.20.2-py3-none-any.whl
