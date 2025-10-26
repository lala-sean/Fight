# Installing MediaPipe and Related Libraries on Windows (Conda + PowerShell)

This guide explains how to create an isolated Python environment using Conda on Windows (PowerShell or Anaconda Prompt) and install MediaPipe, OpenCV, NumPy, pygame, and other dependencies needed for this project. It includes steps to create the environment, install compatible package versions (note: MediaPipe requires `numpy < 2`), verify the installation, and troubleshoot common issues.

---

## Quick Start (recommended)

1. Create and activate a Conda environment (Python 3.10 is recommended):

```
conda create -n mpenv python=3.10 -y
conda activate mpenv
```

3. Install a compatible NumPy release (force NumPy 1.x) and upgrade pip:

```
conda install -y numpy=1.26
python -m pip install --upgrade pip setuptools wheel
```

4. Install the remaining dependencies:

It is recommended to install commonly used binary packages with conda (more stable), then install MediaPipe via pip:

```
# Install OpenCV and pygame from conda-forge (optional but recommended)
conda install -c conda-forge opencv pygame -y

# Install MediaPipe via pip (MediaPipe is distributed via pip)
pip install mediapipe
```

If you prefer to use pip only:

```
pip install numpy==1.26 opencv-python pygame mediapipe
```

> Note: Do not mix NumPy 2.x with MediaPipe; MediaPipe >= 0.10.x requires `numpy < 2`.

5. (Optional) Export your environment's installed packages to `requirements.txt`:

```powershell
pip freeze > requirements.txt
```

---

## Verify the installation

Run the following quick test while the Conda environment is active. Save it as `test_mediapipe.py` or run it interactively:

```python
import numpy as np
import mediapipe as mp
import cv2
import pygame
print('numpy', np.__version__)
print('mediapipe', mp.__version__)
print('opencv', cv2.__version__)
print('pygame', pygame.__version__)
```

Run it from PowerShell:

```powershell
python test_mediapipe.py
```

If there are no import errors and NumPy reports a 1.x version (for example `1.26.*`), the environment is correctly configured.

---

