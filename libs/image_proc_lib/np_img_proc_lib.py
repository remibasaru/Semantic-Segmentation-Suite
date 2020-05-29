# --------------------------------------------------------
# Written by Rilwan Remilekun Basaru
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
from numpy.lib.stride_tricks import as_strided
import os


def compositeshow(im_patches, shape):
    im_patches = im_patches.astype(np.float32)/im_patches.max()
    im_patches = rotate_color_channels(im_patches)
    im_patches = np.reshape(im_patches, (shape[0],
                                         shape[1],
                                         im_patches.shape[-3],
                                         im_patches.shape[-2],
                                         im_patches.shape[-1]))

    im_patches = np.transpose(im_patches, (0, 2, 1, 3, 4))

    composite_img = np.reshape(im_patches, (im_patches.shape[0] * im_patches.shape[1],
                                            im_patches.shape[2] * im_patches.shape[3],
                                            im_patches.shape[4]))
    cv2.imshow("cv2Im scaled", composite_img)
    cv2.waitKey(0)
    return im_patches


def rotate_color_channels(img_rgb):
    assert img_rgb.shape[-1] == 3, 'color chanel should be in the last axis'
    _axis = len(img_rgb.shape) - 1
    tmp_im = np.split(img_rgb, 3, axis=_axis)
    im_bgr = np.squeeze(np.stack((tmp_im[2], tmp_im[1], tmp_im[0]), axis=_axis))

    return im_bgr


def prepare_image(file_name, resize_factor, border):
    orig_img = cv2.imread(file_name)
    if orig_img is None:
        return None
    # print(os.getcwd())
    # print(file_name)

    orig_img = cv2.resize(orig_img, (0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_NEAREST)

    rows, cols, _ = orig_img.shape
    if rows < cols:
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90, 1)
        orig_img = cv2.warpAffine(orig_img, M, (cols, rows))
        print(34)

    orig_img = orig_img[border + 90: rows - border, :]
    orig_img = np.array(orig_img)
    return orig_img


def im2patches(I, patch_size, _stride):
    assert (len(patch_size) == 2), "The patch size should be of size 2!"
    assert (patch_size[0] >= 0 or patch_size[0] >= 0), "Patch size cannot be negative or zero"
    stride_x = stride_y = _stride
    patch_height = patch_size[0]
    patch_width = patch_size[1]
    hin, win, din = I.shape

    hout = hin - ((hin - patch_height) % stride_y) + stride_y * (stride_y > 1)
    wout = win - ((win - patch_width) % stride_x) + stride_x * (stride_x > 1)

    # pad with zeros if necessary
    I = cv2.copyMakeBorder(I, top=0, bottom=hout - hin, left=0, right=wout - win, borderType=cv2.BORDER_CONSTANT,
                           value=[0, 0, 0])

    num_height = int(1 + (hout - patch_height) / stride_y)
    num_width = int(1 + (wout - patch_width) / stride_x)

    shape_out = (num_height, num_width, patch_size[0], patch_size[1], din,)
    stride_out = (I.strides[0] * stride_y, I.strides[1] * stride_x) + I.strides
    # print(stride_out)
    # print(shape_out)
    out_patches = as_strided(I, shape=shape_out, strides=stride_out, writeable=False)

    return out_patches, I


def patches2im(_patches, patch_dim, _stride, pad=np.nan):
    # if _patches.shape
    assert _patches.dtype == np.uint8, 'Invalid data type convert patches to numpy.uint8'
    # TO DO: Make this function work for data type this not np.uint8
    assert _patches.ndim == 3 or _patches.ndim == 4, 'Invalid patch shape'
    if _patches.ndim == 3:
        N, H, W = _patches.shape
        C = 1
    else:
        N, H, W, C = _patches.shape

    assert len(patch_dim) == 2, 'Only 2-D reconstruct is permitted!'
    assert patch_dim[0] * patch_dim[1] == N, 'Product of patch dimension should be equal to number ' \
                                                             'of patches'

    h_out = (patch_dim[0] - 1) * _stride + H
    w_out = (patch_dim[1] - 1) * _stride + W
    stride_x = stride_y = _stride

    composite_image = np.zeros((N, h_out, w_out, C), dtype=_patches.dtype)
    composite_denominator = np.zeros((N, h_out, w_out, C), dtype=_patches.dtype)

    composite_image[:] = pad
    stride_in = composite_image.strides
    shape_out = (patch_dim[0], patch_dim[1], H, W, C)
    stride_out = ((patch_dim[1] * h_out + stride_y) * stride_in[-3], stride_in[-4] + stride_x * C, stride_in[-3],
                  stride_in[-2], stride_in[-1])
    idx = as_strided(composite_image, shape=shape_out, strides=stride_out, writeable=True)
    idx_denom = as_strided(composite_denominator, shape=shape_out, strides=stride_out, writeable=True)

    _patches = _patches.reshape(idx.shape)

    # Broadcast to from patches to the composite image
    idx[:] = _patches[:]
    idx_denom[:] = 1
    return composite_image, composite_denominator


def load_patches_from_image(file_name=os.path.dirname(__file__) + '/Data/Raw/S1/sample_img.jpg',
                            tensorflow_compatible=True):
    if file_name is None:
        print('Please specify filename!')
        exit(0)
    else:
        Im = prepare_image(file_name, resizeFactor, border)
        if Im is None: 
            print('Invalid file name. File : ' + file_name + ' does not exist')
            exit(0)
        patches, i_out = im2patches(Im, (patch_size, patch_size), stride)
        patches = np.ascontiguousarray(patches, dtype=np.float32)
        patch_prop = dict()
        patch_prop['orig_shape'] = patches.shape[0:2]
        patch_prop['stride'] = stride
        if tensorflow_compatible:
            patches = np.reshape(patches, (-1, patches.shape[2], patches.shape[3], patches.shape[4]))
            # patches = np.transpose(patches, (0, 3, 1, 2))
            # patches.astype(np.uint32)
        return patches, i_out, patch_prop


if __name__ == "__main__":
    border = 200
    resizeFactor = 0.1500
    patch_size = 121
    stride = 100
    filename = os.path.join('sample_img.jpg')

    Im = prepare_image(filename, resizeFactor, border)
    # Im = cv2.cvtColor(Im, cv2.COLOR_BGR2GRAY)
    patches, i_out = im2patches(Im, (patch_size, patch_size), stride)

    patches = np.reshape(patches, (-1, patches.shape[2], patches.shape[3], patches.shape[4]))

    cI = patches2im(patches, (3, 5), stride)

    cI = patches2im(patches.astype(np.uint8), (3, 5), stride)

    # plt.ion()
    for j in range(len(cI)):

        y = cI[j]
        print(y.shape)
        # y = Image.fromarray(y[:, :, 0])
        y = (255/y.max() * y).astype(np.uint8)

        # plt.imshow(y[:, :, 0])
        # plt.pause(0.95)


