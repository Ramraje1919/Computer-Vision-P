"""
OSNet Backbone Model Loader for Person Re-Identification.
Loads the OSNet model directly without importing heavy training frameworks.
"""

from pathlib import Path
from typing import Optional
import importlib.util
import torch
import torch.nn as nn


def _load_osnet_definition():
    """Load OSNet definition directly from osnet.py to avoid torchreid.__init__ cascade."""
    # Direct import from site-packages
    site_packages = Path(torch.__file__).parent.parent
    osnet_path = site_packages / "torchreid" / "reid" / "models" / "osnet.py"
    if osnet_path.exists():
        spec = importlib.util.spec_from_file_location("osnet_module", osnet_path)
        osnet_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(osnet_mod)
        return osnet_mod

    from torchreid.reid.models import osnet
    return osnet


def build_osnet_model(
    checkpoint_path: Path,
    device: Optional[torch.device] = None
) -> nn.Module:
    """Load pretrained OSNet (x0_25) model with MSMT17 weights."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"OSNet checkpoint not found at: {checkpoint_path}")

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=False)

    num_classes = checkpoint["classifier.weight"].shape[0] if "classifier.weight" in checkpoint else 4101
    osnet_module = _load_osnet_definition()
    model = osnet_module.osnet_x0_25(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint, strict=True)
    model.to(dev)
    model.eval()
    return model
