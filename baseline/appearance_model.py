import os
import cv2
import torch
import numpy as np
from torchreid.reid.models import osnet


class OSNetAppearance:
    def __init__(self, checkpoint_path=None, device=None):
        if checkpoint_path is None:
            project_root = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
            checkpoint_path = os.path.join(
                project_root,
                "weights",
                "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
            )

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        num_classes = checkpoint["classifier.weight"].shape[0]

        self.model = osnet.osnet_x0_25(
            num_classes=num_classes,
            pretrained=False
        )

        self.model.load_state_dict(checkpoint, strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.input_width = 128
        self.input_height = 256

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        print(f"OSNet loaded successfully on {self.device}")

    def extract(self, frame, box):
        x1, y1, x2, y2 = map(int, box)

        h, w = frame.shape[:2]

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        crop = cv2.resize(
            crop,
            (self.input_width, self.input_height)
        )

        crop = crop.astype(np.float32) / 255.0
        crop = (crop - self.mean) / self.std

        tensor = torch.from_numpy(crop)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            features = self.model(tensor)

            features = torch.nn.functional.normalize(
                features,
                p=2,
                dim=1
            )

        return features.squeeze(0).cpu().numpy()