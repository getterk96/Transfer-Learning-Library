import os

import numpy as np
import torch
from PIL import Image

from .segmentation_list import SegmentationList
from .cityscapes import Cityscapes
from .._util import download as download_data


class GTA5(SegmentationList):
    """`GTA5 <https://download.visinf.tu-darmstadt.de/data/from_games/>`_

    Args:
        root (str): Root directory of dataset
        split (str, optional): The dataset split, supports ``train``.
        data_folder (str, optional): Sub-directory of the image. Default: 'images'.
        label_folder (str, optional): Sub-directory of the label. Default: 'labels'.
        mean (seq[float]): mean BGR value. Normalize the image if not None. Default: None.
        transforms (callable, optional): A function/transform that  takes in  (PIL image, label) pair \
            and returns a transformed version. E.g, :class:`~dalib.vision.transforms.segmentation.Resize`.

    .. note:: You need to download GTA5 manually.
        Ensure that there exist following directories in the `root` directory before you using this class.
        ::
            images/
            labels/
    """
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/e13fbec03e5045058df1/?dl=1"),
    ]

    EVALUATE_CLASSES = Cityscapes.CLASSES

    def __init__(self, root, split='train', data_folder='images', label_folder='labels', **kwargs):
        assert split in ['train']
        # download meta information from Internet
        list(map(lambda args: download_data(root, *args), self.download_list))
        data_list_file = os.path.join(root, "image_list", "{}.txt".format(split))
        self.split = split
        super(GTA5, self).__init__(root, Cityscapes.CLASSES, data_list_file, data_list_file, data_folder, label_folder,
                                   id_to_train_id=Cityscapes.ID_TO_TRAIN_ID, train_id_to_color=Cityscapes.TRAIN_ID_TO_COLOR, **kwargs)

class GTA5_6(SegmentationList):
    """`GTA5 <https://download.visinf.tu-darmstadt.de/data/from_games/>`_

    Args:
        root (str): Root directory of dataset
        split (str, optional): The dataset split, supports ``train``.
        data_folder (str, optional): Sub-directory of the image. Default: 'images'.
        label_folder (str, optional): Sub-directory of the label. Default: 'labels'.
        mean (seq[float]): mean BGR value. Normalize the image if not None. Default: None.
        transforms (callable, optional): A function/transform that  takes in  (PIL image, label) pair \
            and returns a transformed version. E.g, :class:`~dalib.vision.transforms.segmentation.Resize`.

    .. note:: You need to download GTA5 manually.
        Ensure that there exist following directories in the `root` directory before you using this class.
        ::
            images/
            labels/
    """
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/e13fbec03e5045058df1/?dl=1"),
    ]

    EVALUATE_CLASSES = Cityscapes.CLASSES

    def __init__(self, root, split='train', data_folder='images', label_folder='labels', **kwargs):
        assert split in ['train']
        # download meta information from Internet
        list(map(lambda args: download_data(root, *args), self.download_list))
        data_list_file = os.path.join(root, "image_list", "{}.txt".format(split))
        self.split = split
        super(GTA5_6, self).__init__(root, Cityscapes.CLASSES, data_list_file, data_list_file, data_folder,
                                   label_folder,
                                   id_to_train_id=Cityscapes.ID_TO_TRAIN_ID,
                                   train_id_to_color=Cityscapes.TRAIN_ID_TO_COLOR, **kwargs)

    def __getitem__(self, index):
        image_name = self.data_list[index]
        label_name = self.label_list[index]
        ori_image = Image.open(os.path.join(self.root, self.data_folder, image_name)).convert('RGB')
        ori_label = Image.open(os.path.join(self.root, self.label_folder, label_name))

        images = []
        labels = []
        for i in range(6):
            image, label = self.transforms(ori_image, ori_label)

            # remap label
            if isinstance(label, torch.Tensor):
                label = label.numpy()
            label = np.asarray(label, np.int64)
            label_copy = self.ignore_label * np.ones(label.shape, dtype=np.int64)
            if self.id_to_train_id:
                for k, v in self.id_to_train_id.items():
                    label_copy[label == k] = v

            images.append(image)
            labels.append(label_copy.copy())

        return images, labels