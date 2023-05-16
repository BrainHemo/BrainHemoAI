import os
import numpy as np
import SimpleITK
import typing

aSAH_SlicerSegMetaData = {
    'Segment0_ID': 'Segment_1',
    'Segment0_Name': 'SAH',
    'Segment0_LabelValue': '1',
    'Segment0_Layer': '0',

    'Segment1_ID': 'Segment_2',
    'Segment1_Name': 'IPH',
    'Segment1_LabelValue': '2',
    'Segment1_Layer': '0',

    'Segment2_ID': 'Segment_3',
    'Segment2_Name': 'IVH',
    'Segment2_LabelValue': '3',
    'Segment2_Layer': '0',
}

extension = ('nrrd', 'nhdr', 'mhd', 'mha')
file_filter = 'NRRD(*.nrrd);; NRRD(*nhdr);; MetaImage(*.mhd);; MetaImage(*.mha)'


class DataIO:
    @classmethod
    def load_by_sitk(cls, path: str) -> SimpleITK.Image:
        image = SimpleITK.ReadImage(path)
        return image

    @classmethod
    def load_dicom_series(cls, path: str) -> SimpleITK.Image:
        dcm_name = SimpleITK.ImageSeriesReader.GetGDCMSeriesFileNames(path)
        dcm_read = SimpleITK.ImageSeriesReader()
        dcm_read.SetFileNames(dcm_name)
        dcm_series = dcm_read.Execute()
        return dcm_series

    @classmethod
    def save_by_sitk_with_ref(cls, ref: SimpleITK.Image, datas: typing.List[np.ndarray], paths: typing.List[str],
                              meta: typing.Union[typing.List[typing.Dict[str, str]], None] = None) -> None:
        assert len(datas) == len(paths)

        for i in range(len(datas)):
            data = datas[i]
            path = paths[i]

            out_dir = os.path.dirname(path)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)

            image = SimpleITK.GetImageFromArray(data)
            # image.SetOrigin(ref.GetOrigin())
            # image.SetSpacing(ref.GetSpacing())
            # image.SetDirection(ref.GetDirection())
            image.CopyInformation(ref)

            if meta is not None:
                infos = meta[i]
                for key in infos.keys():
                    image.SetMetaData(key, infos[key])

            SimpleITK.WriteImage(image, path, useCompression=True)

    @classmethod
    def save_by_sitk(cls, data: SimpleITK.Image, path: str) -> None:
        ex = path.split('.')[-1]
        if ex not in extension:
            print('not matched file\'s extension!')
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            SimpleITK.WriteImage(data, path, useCompression=True)

    @classmethod
    def get_ref_image_geometry(cls, ref_image: SimpleITK.Image) -> str:
        """
        the reference image geometry for 3d slicer
        """
        spacing = np.array(ref_image.GetSpacing())
        direction = np.array(ref_image.GetDirection()).reshape((len(spacing), -1))
        spacing_and_direction = spacing * direction
        origin = ref_image.GetOrigin()
        size = ref_image.GetSize()

        def to_str(data: np.ndarray) -> str:
            res = ''
            for v in data:
                res += str(v) + ';'
            return res[:-2]

        # the first two dimensions of spacing and origin need to use the opposite number.
        # 0;0;0;1 => I don't what it is
        ref_geometry = to_str(-spacing_and_direction[0]) + ';' + str(-origin[0]) + ';' + \
                       to_str(-spacing_and_direction[1]) + ';' + str(-origin[1]) + ';' + \
                       to_str(spacing_and_direction[2]) + ';' + str(origin[2]) + ';' + \
                       '0;0;0;1;' + \
                       '0;' + str(size[0] - 1) + ';0;' + str(size[1] - 1) + ';0;' + str(size[2] - 1) + ';'
        return ref_geometry
