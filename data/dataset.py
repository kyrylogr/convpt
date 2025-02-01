import torch
import torchvision.transforms.v2 as transforms

from data_generator import *


class SynteticTransformDataSet(torch.utils.data.Dataset):
    """Dataset generating center point tracking images with random trasnforms.
    Item returns image1, image2 (transformed), class and offset maps for ground
    truth center offset.
    """

    def __init__(
        self,
        filelist,
        size1,
        size2,
        max_shift,
        angle_sigma,
        scale_sigma,
        rotation_center_max_shift,
        result_stride=16,
        pad=8,
    ):
        self._filelist = filelist.copy()
        self.size1 = size1
        self.size2 = size2
        self.max_shift = max_shift
        self.angle_sigma = angle_sigma
        self.scale_sigme = scale_sigma
        self.rotation_center_max_shift = rotation_center_max_shift
        self.result_stride = result_stride
        self.padding = pad
        self.transforms = transforms.Compose(
            [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
        )

    def __getitem__(self, index):
        img = read_rgb(self._filelist[index])
        img1, img2, offset = sample_crops_with_transform(
            img,
            self.size1,
            self.size2,
            self.max_shift,
            self.angle_sigma,
            self.scale_sigma,
            self.rotation_center_max_shift,
            self.pad,
        )
        img_cls, img_offsets = center_offset_encoder(
            self.size2 // self.result_stride, self.result_stride, offset
        )
        gt = torch.cat(
            [
                torch.Tensor(np.expand_dims(img_cls, 0)),
                torch.Tensor(img_offsets).permute(2, 0, 1),
            ]
        )
        return self.transforms(img1), self.transforms(img2), gt

    def __len__(self):
        return len(self._filelist)
