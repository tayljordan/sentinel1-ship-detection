"""Vessel detection in Sentinel-1 radar imagery.

Downloads Sentinel-1 GRD scenes from the Copernicus Data Space Ecosystem and runs the
pretrained vessel detection and attribute models from the Allen Institute for AI
(Beukema et al., 2023, arXiv:2312.03207).
"""
import os

__version__ = "0.1.0"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
