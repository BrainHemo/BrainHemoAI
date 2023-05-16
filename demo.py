import typing
from os.path import join

import SimpleITK
import numpy as np

import ml
import utils
from utils import DataIO, aSAH_SlicerSegMetaData


def run(head: np.ndarray, model_wrapper: ml.BaseModelWrapper,
        m_brain: np.ndarray, rotation=None, post_mode='connect') -> typing.Dict[str, np.ndarray]:

    angle, center = rotation

    head, m_brain = utils.prepare_data(head, m_brain, (angle, center))

    preds = model_wrapper.predict(head, m_brain)

    m_blood = utils.rotate_each_blood_type(preds, -angle, center)
    if post_mode == 'connect':
        m_blood = utils.postprocess.closing(m_blood)
        m_blood = utils.postprocess.remove_small_object(m_blood)
        # run twice
        m_blood = utils.postprocess.rectify_blood_type(m_blood)
        m_blood = utils.postprocess.rectify_blood_type(m_blood)

    return {
        'm_blood': m_blood
    }


def main():
    root = 'nrrd'

    model_wrapper: ml.BaseModelWrapper = ml.ModelWrapperFactory.get_model_wrapper(ml.ModelWrapperFactory.TriHybridUNet)

    head = SimpleITK.ReadImage(join(root, 'head.nrrd'))
    m_brain = SimpleITK.ReadImage(join(root, 'mask_brain.seg.nrrd'))

    result = run(
            head=SimpleITK.GetArrayFromImage(head),
            model_wrapper=model_wrapper,
            m_brain=SimpleITK.GetArrayFromImage(m_brain),
            rotation=(17, (256, 256))
        )

    ref_geometry = DataIO.get_ref_image_geometry(head)
    meta = aSAH_SlicerSegMetaData.copy()
    meta['Segmentation_ConversionParameters'] = 'Reference image geometry|' + ref_geometry + '&'

    DataIO.save_by_sitk_with_ref(head, [result['m_blood']], [join(root, 'preds_blood.seg.nrrd')], [meta])


if __name__ == '__main__':
    main()
    print('finish')

