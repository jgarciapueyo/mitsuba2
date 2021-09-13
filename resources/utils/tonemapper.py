import numpy as np
import matplotlib.pyplot as plt


def apply_normalization(img: np.array) -> np.array:
    maximum = img.max()
    minimum = img.min()
    factor = 1 / (maximum - minimum)
    return (img - minimum) * factor


def tonemap_SRGB(img: np.array, gamma: float = 2.2) -> np.array:
    a = 0.055
    img_tonemapped = np.where(img <= 0.0031308,
                              img * 12.92,
                              (1 + a) * np.power(img, 1 / gamma) - a)
    return np.clip(img_tonemapped, a_min=0., a_max=1.)


def tonemap_gamma(img: np.array, gamma: float = 2.2) -> np.array:
    img_tonemapped = np.power(img, 1 / gamma)
    return np.clip(img_tonemapped, a_min=0., a_max=1.)


def tonemap_falsenegative(img: np.array) -> np.array:
    channelR = -2 * img.clip(max=0.).mean(axis=-1)
    channelG = 2 * img.clip(min=0.).mean(axis=-1)
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


def tonemap(img: np.array,
            normalize: bool = False,
            exposure: float = 0.,
            offset: float = 0.,
            tonemapper: float = "SRGB",
            gamma: float = 2.2) -> np.array:
    if normalize:
        img = apply_normalization(img)

    if exposure != 0.:
        img = np.power(2, exposure) * img

    if offset != 0.:
        img = img + offset

    if tonemapper == "SRGB":
        img = tonemap_SRGB(img, gamma)
    elif tonemapper == "GAMMA":
        img = tonemap_gamma(img, gamma)
    elif tonemapper == "PN":
        img = tonemap_falsenegative(img)
    else:
        raise "UNKNOWN TONEMAPPER"

    img = img.astype(np.float32)
    return img
