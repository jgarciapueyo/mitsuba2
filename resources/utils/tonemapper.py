from typing import Tuple

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def toSRGB(linear: float, gamma: float = 2.4):
    a = 0.055
    if linear <= 0.0031308:
        return 12.92 * linear
    else:
        return (1 + a) * pow(linear, 1 / gamma) - a


def tonemap_SRGB(img_hdr_RGB: np.array) -> np.array:
    vf = np.vectorize(toSRGB)
    return np.clip(vf(img_hdr_RGB), a_min=0., a_max=1.)


def to_gamma(value: float, gamma: float):
    return pow(value, 1 / gamma)


def tonemap_gamma(img_hdr_RGB: np.array, gamma: float = 2.2) -> np.array:
    vf = np.vectorize(to_gamma)
    return np.clip(vf(img_hdr_RGB, gamma), a_min=0., a_max=1.)


def tonemap_falsenegative(img_hdr_RGB: np.array) -> np.array:
    channelR = -2 * img_hdr_RGB.clip(max=0.).mean(axis=-1)
    channelG = 2 * img_hdr_RGB.clip(min=0.).mean(axis=-1)
    channelB = np.zeros(channelG.shape)
    img_mapped = np.stack((channelR, channelG, channelB), axis=-1)
    return np.clip(img_mapped, a_min=0., a_max=1.)


def show_image(img: np.array):
    number_of_channels = img.shape[2]
    fig, axs = plt.subplots(number_of_channels)
    for channel in range(number_of_channels):
        im = axs[channel].imshow(img[:, :, channel])
        fig.colorbar(im)
    plt.show()


def tonemap_custom(img_hdr_RGB: np.array, tonemap: str) -> Tuple[np.array, np.array]:
    img_ldr_RGB = None
    if tonemap == "SRGB":
        img_ldr_RGB = tonemap_SRGB(img_hdr_RGB)
    elif tonemap == "GAMMA":
        img_ldr_RGB = tonemap_gamma(img_hdr_RGB)
    elif tonemap == "PN":
        img_ldr_RGB = tonemap_falsenegative(img_hdr_RGB)
    else:
        raise "UNKNOWN TONEMAPPER"

    img_ldr_RGB = img_ldr_RGB.astype(np.float32)
    img_ldr_BGR = cv.cvtColor(img_ldr_RGB, code=cv.COLOR_RGB2BGR)
    return img_ldr_RGB, img_ldr_BGR
