from .model_wrapper import BaseModelWrapper
from .tri_hybrid_unet_wrapper import TriHybridNetWrapper


class ModelWrapperFactory:
    """
    register all the models here
    """

    TriHybridUNet = 'TriHybridUNet'

    @classmethod
    def get_model_names(cls):
        return ModelWrapperFactory.TriHybridUNet
    
    @classmethod
    def get_model_wrapper(cls, model_name: str, device='cpu') -> BaseModelWrapper:
        if model_name == cls.TriHybridUNet:
            return TriHybridNetWrapper('ml/pth/tri_hybrid_unet.pth', device)
        return None
