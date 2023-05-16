from abc import abstractmethod, ABCMeta
import torch.nn as nn
import numpy as np


class BaseModelWrapper(metaclass=ABCMeta):
    def __init__(self, model: nn.Module) -> None:
        self.model: nn.Module = model.eval()
    
    @abstractmethod
    def predict(self, head: np.ndarray, m_brain: np.ndarray) -> np.ndarray:
        pass

    def device(self):
        return next(self.model.parameters()).device
