import numpy as np
import cv2 as cv
import os
import random


def read_rgb(imgfilename):
    img_org = cv.imread(imgfilename)
    return cv.cvtColor(img_org, cv.COLOR_BGR2RGB)


def point_transform(pt, tr):
    """transform sigle point"""
    return cv.perspectiveTransform(pt.reshape(1, 1, 2), tr).reshape(2)


def crop_at_point(img, point_xy, size: int):
    """crop with center at specified point and size.
    Does not yet support padding.
    """
    hsize = size // 2
    x1 = round(point_xy[0] - hsize)
    y1 = round(point_xy[1] - hsize)
    assert x1 >= 0
    assert y1 >= 0
    return img[y1 : y1 + size, x1 : x1 + size]


def gen_random_affine(center_point, angle_sigma, scale_sigma, max_center_shift=0):
    """generates random affine"""
    if max_center_shift > 0:
        center_range = max_center_shift * 2
        center_point = (
            center_point[0] + center_range * (random.random() - 0.5),
            center_point[1] + center_range * (random.random() - 0.5),
        )
    scale = 1.0 + np.random.randn() * scale_sigma
    angle = np.random.randn() * angle_sigma
    return cv.getRotationMatrix2D(center_point, angle, scale)


def sample_crops_with_transform(
    img,
    size1,
    size2,
    max_shift,
    angle_sigma,
    scale_sigma,
    rotation_center_max_shift,
    center_crop=True,
    pad=8,
):
    """Samples two crops of specified size with second patch being random transform from original image.
    Returns:
        img1: square image of size1, randomly cropped from original image (without resize)
        img2: square image of size2, randomly cropped from randomly transformed image
        translation (x,y): translation values of center2 with respect to center1 (values <= max_shift)
    """
    h, w = img.shape[:2]
    tr = gen_random_affine(
        (w / 2, h / 2), angle_sigma, scale_sigma, rotation_center_max_shift
    )
    dst = cv.warpAffine(img, tr, (w, h))
    trex = np.vstack([tr, [0.0, 0.0, 1.0]])
    max_size = max(size1, size2)
    center_pad = pad + max_size / 2
    w_h = np.array([w, h])
    original_center_range_wh = w_h - 2 * center_pad
    if center_crop:
        original_center_range_wh = original_center_range_wh.min()
    original_center_xy = np.round(
        w_h / 2 + original_center_range_wh * (np.random.rand(2) - 0.5)
    )
    center_transformed = point_transform(original_center_xy, trex)
    translate_range = max_shift * 2
    translated_candidate_center_xy = center_transformed + translate_range * (
        np.random.rand(2) - 0.5
    )
    translated_candidate_center_xy[0] = np.clip(
        translated_candidate_center_xy[0], center_pad, w - center_pad
    )
    translated_candidate_center_xy[1] = np.clip(
        translated_candidate_center_xy[1], center_pad, h - center_pad
    )
    translated_transformed_center_xy = np.round(translated_candidate_center_xy)
    translation = center_transformed - translated_transformed_center_xy
    crop_original = crop_at_point(img, original_center_xy, size1)
    crop_transformed = crop_at_point(dst, translated_transformed_center_xy, size2)
    return crop_original, crop_transformed, translation


def to_grayscale(img):
    return cv.cvtColor(img, cv.COLOR_RGB2GRAY)


def filter_color_imgfiles_min_size(folder, minsize):
    """returns list of imagefile paths from the folder if it satisfies minsize.
    Currently only imagefiles should be in input folder.
    """

    def fits(fpath):
        img = cv.imread(fpath)
        s = img.shape
        return len(s) == 3 and s[0] >= minsize and s[1] >= minsize

    file_paths = [os.path.join(folder, fname) for fname in os.listdir(folder)]
    return list(filter(fits, file_paths))


def gaussian2D(shape, sigma_x=1, sigma_y=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-x * x / (2 * sigma_x**2) - y * y / (2 * sigma_y**2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius):
    diameter = 2 * radius + 1
    sigma = diameter / 6
    gaussian = gaussian2D((diameter, diameter), sigma, sigma)

    x, y = int(np.round(center[0])), int(np.round(center[1]))

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
    return heatmap


def center_offset_encoder(size, stride, offset_xy, sigma=2):
    cls = np.zeros(shape=(size, size), dtype=np.float32)
    offsets = np.zeros(shape=(size, size, 2), dtype=np.float32)
    hsize = size // 2
    offset_strided_xy = np.floor(offset_xy / stride)
    y_strided = int(offset_strided_xy[1] + hsize)
    x_strided = int(offset_strided_xy[0] + hsize)
    draw_gaussian(cls, np.array([x_strided, y_strided]), sigma)
    offsets[y_strided, x_strided] = offset_xy - (0.5 + offset_strided_xy) * stride
    return cls, offsets
