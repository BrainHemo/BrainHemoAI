import numpy as np
import torch

from . import comm_tools
from .model_wrapper import BaseModelWrapper
from .tri_hybrid_unet import TriHybridUNet


class TriHybridNetWrapper(BaseModelWrapper):
    def __init__(self, pth: str, device: str) -> None:
        # disable the auxiliary decoder
        model = TriHybridUNet(5, 4, 32, auxiliary=False)
        model = model.to(device=device)
        model.load_state_dict(torch.load(pth, map_location=device))
        super().__init__(model)
    
    def predict(self, head: np.ndarray, m_brain: np.ndarray) -> np.ndarray:
        """
        x=> (5, 18, 352, 288)
        """
        brain = np.zeros_like(head)
        brain[m_brain] = head[m_brain]

        x1, top_left_pos = comm_tools.crop_data(brain)
        x2 = comm_tools.resize_data(x1)

        idx_start, idx_end = comm_tools.area_clip(x2)

        x = x2[idx_start: idx_end]
        # the number of slice can be any value after training, and 18 is the best.
        x = comm_tools.pad_zdims(x, 18)
        x = comm_tools.get_win_data(x)

        x = torch.tensor(x, dtype=torch.float32)

        x2d = torch.permute(x, [1, 0, 2, 3])
        x3d = torch.unsqueeze(x, dim=0)

        x2d = x2d.to(self.device())
        x3d = x3d.to(self.device())

        with torch.no_grad():
            preds_hybrid, preds_2d, preds_3d = self.model(x2d, x3d)
        
        preds = torch.permute(preds_hybrid, [0, 2, 3, 4, 1])
        preds = torch.argmax(preds, dim=4)
        preds = torch.squeeze(preds)
        preds = preds.cpu().numpy()

        result = np.zeros_like(x2, dtype=np.int8)
        nz = min(idx_end - idx_start, 18)
        result[idx_start: idx_start+nz] = preds[:nz]
        result = comm_tools.resize_each_blood_type(result, x1.shape[1:])
        top, bottom = top_left_pos[0], top_left_pos[0] + result.shape[1]
        left, right = top_left_pos[1], top_left_pos[1] + result.shape[2]
        res = np.zeros_like(brain, dtype=np.int8)
        res[:, top: bottom, left: right] = result

        return res
        