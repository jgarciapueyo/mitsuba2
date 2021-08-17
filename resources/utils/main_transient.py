import argparse
import glob
from typing import Tuple

import cv2 as cv
import imageio
import numpy as np
import matplotlib.pyplot as plt

import mitsuba

mitsuba.set_variant("scalar_rgb")


def read_steadyimg_mitsuba(dir: str, extension: str = "exr") -> np.array:
    from mitsuba.core import Bitmap, Struct, float_dtype
    # other = Bitmap(f"{dir}.{extension}") \
    #     .convert(Bitmap.PixelFormat.XYZA, Struct.Type.Float32, srgb_gamma=False)
    other = Bitmap(f"{dir}.{extension}") \
        .convert(Bitmap.PixelFormat.RGBA, Struct.Type.Float32, srgb_gamma=False)
    img = np.array(other, copy=False)
    return img


def read_streakimg(dir: str, extension: str = "exr") -> np.array:
    """
    Reads all the images x-t that compose the streak image.

    :param dir: path where the images x-t are stored
    :param extension: of the images x-t
    :return: a streak image of shape [height, width, time, nchannels]
    """
    number_of_xtframes = len(glob.glob(f"{dir}/frame_*.{extension}"))
    fileList = []
    for i_xtframe in range(number_of_xtframes):
        img = imageio.imread(f"{dir}/frame_{i_xtframe}.{extension}")
        fileList.append(np.expand_dims(img, axis=0))

    streak_img = np.concatenate(fileList)
    return np.nan_to_num(streak_img, nan=0.)


def read_streakimg_mitsuba(dir: str, extension: str = "exr") -> np.array:
    """
    Reads all the images x-t that compose the streak image.

    :param dir: path where the images x-t are stored
    :param extension: of the images x-t
    :return: a streak image of shape [height, width, time, nchannels]
    """
    from mitsuba.core import Bitmap, Struct, float_dtype
    number_of_xtframes = len(glob.glob(f"{dir}/frame_*.{extension}"))
    fileList = []
    for i_xtframe in range(number_of_xtframes):
        other = Bitmap(f"{dir}/frame_{i_xtframe}.{extension}") \
            .convert(Bitmap.PixelFormat.RGBA, Struct.Type.Float32, srgb_gamma=False)
        img = np.array(other, copy=False)
        fileList.append(np.expand_dims(img, axis=0))

    streak_img = np.concatenate(fileList)
    return np.nan_to_num(streak_img, nan=0.)


def streakimg2steadyimg(streakimg: np.array) -> np.array:
    """
    Accumulates all the radiance of the streak image along time to convert it to a steady (normal) image.

    :param streakimg: streak image of shape [height, width, time, nchannels]
    :return: steady image of shape [height, width, nchannels]
    """
    return np.sum(streakimg, 2)


def diff_images(img1: np.array, img2: np.array):
    """
    Shows the difference in values between both images.
    :param img1:
    :param img2:
    :return:
    """
    diff = streakimg_mitsuba - steadyimg_mitsuba
    number_of_channels = diff.shape[2]
    fig, axs = plt.subplots(number_of_channels)
    for channel in range(number_of_channels):
        axs[channel].imshow(diff[:, :, channel])
        axs[channel].colorbar()
    plt.show()


def tonemap_wrapper(tonemap: cv.Tonemap, img_hdr_RGB: np.array) -> Tuple[np.array, np.array]:
    """
    Tonemap wrapper from HDR image with RGB format to LDR images with format RGB and BGR format. The values of the LDR
    images are in range [0, 1].

    :param tonemap:
    :param img:
    :return:
    """
    img_ldr_RGB = tonemap.process(img_hdr_RGB)
    img_ldr_BGR = cv.cvtColor(img_ldr_RGB, code=cv.COLOR_RGB2BGR)
    return img_ldr_RGB, img_ldr_BGR


def write_video(streakimg_hdr_RGB: np.array, out_file: str, tonemap: cv.Tonemap):
    """
    Creates a video from a HDR streak image (dimensions [height, width, time, channels]) in RGB format. The tonemap is
    needed to transform the HDR streak image to a LDR streak image.

    :param streakimg_hdr:
    :param tonemap:
    """
    # 1. Get the streak image (already done) and define the output
    writer = imageio.get_writer(out_file + ".mp4", fps=10)
    # 2. Iterate over the streak img frames
    for i in range(streakimg_hdr_RGB.shape[2]):
        img_temp = streakimg_hdr_RGB[:, :, i, :3]
        img_temp[img_temp < 0] = 0.
        img_hdr_RGB = img_temp
        img_ldr_RGB = tonemap.process(img_hdr_RGB)
        writer.append_data((img_ldr_RGB * 255).astype(np.uint8))
    # 3. Write the video
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool for Streak Images")
    parser.add_argument('-d', '--dir', type=str, help="Directory where the steady and streak images are stored",
                        default="cbox")
    parser.add_argument('-fs', '--file-steady-image', type=str, help="Name of the image file (for steady image)",
                        default="scene")
    parser.add_argument('-ft', '--file-streak-image', type=str, help="Name of the folder containing the single images "
                                                                     "that compose the streak image",
                        default="transient_scene")
    parser.add_argument('-e', '--extension', type=str, help="Name of the extension of the images", default="exr")
    args = parser.parse_args()

    path_steady_img = args.dir + "/" + args.file_steady_image
    path_streak_img = args.dir + "/" + args.file_streak_image

    # Check if steady image exists for this scene
    steady_img_exists = False
    if glob.glob(path_steady_img + "." + args.extension):
        steady_img_exists = True

    # Read steady image
    steadyimg_mitsuba = None
    if steady_img_exists:
        steadyimg_mitsuba = read_steadyimg_mitsuba(path_steady_img, extension=args.extension)
        steadyimg_mitsuba = steadyimg_mitsuba[:, :, :3]  # drop alpha

    # Read streak image
    streakimg_mitsuba_original = read_streakimg_mitsuba(path_streak_img, extension=args.extension)
    streakimg_mitsuba = streakimg2steadyimg(streakimg_mitsuba_original)
    streakimg_mitsuba = streakimg_mitsuba[:, :, :3]  # drop alpha

    # Comparison of steady and streak (accumulated)
    if steadyimg_mitsuba is not None:
        diff_images(steadyimg_mitsuba, streakimg_mitsuba)

    # Tonemapping of HDR images
    tonemap = cv.createTonemapReinhard(gamma=2.2, light_adapt=0., color_adapt=0.)
    steadyimg_mitsuba_ldr_RGB = None
    steadyimg_mitsuba_ldr_BGR = None
    if steadyimg_mitsuba:
        [steadyimg_mitsuba_ldr_RGB, steadyimg_mitsuba_ldr_BGR] = tonemap_wrapper(tonemap, steadyimg_mitsuba)
    [streakimg_mitsuba_ldr_RGB, streakimg_mitsuba_ldr_BGR] = tonemap_wrapper(tonemap, streakimg_mitsuba)

    # Show tonemapped images
    if steadyimg_mitsuba:
        cv.imshow("Steady Image", steadyimg_mitsuba_ldr_BGR)
    cv.imshow("Streak Image (Accumulated)", streakimg_mitsuba_ldr_BGR)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Write LDR images
    name_steadyimg_file = args.dir + "/steadyimg.png"
    name_streakimg_file = args.dir + "/streakimg_accumulated.png"
    if steadyimg_mitsuba:
        imageio.imwrite(name_steadyimg_file, (steadyimg_mitsuba_ldr_RGB * 255).astype(np.uint8))
    imageio.imwrite(name_streakimg_file, (streakimg_mitsuba_ldr_RGB * 255).astype(np.uint8))

    # Write video of streak image
    name_video_file = args.dir + "/streak_video"
    write_video(streakimg_mitsuba_original, name_video_file, tonemap)
