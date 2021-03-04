import os

from setuptools import find_packages, setup

opencv = 'opencv-python-headless' if os.path.exists('/.dockerenv') else 'opencv-python'

setup(
    name="centermask",
    version="0.0.1",
    packages=find_packages(),
    url="https://github.com/monkvision/monk",
    author="Monk",
    author_email="monk@monkvision.ai",
)
