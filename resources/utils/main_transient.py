import argparse
import glob
from pathlib import Path

import imageio

import cv2 as cv
import mitsuba

from tonemapper import *

mitsuba.set_variant("scalar_rgb")


def read_steadyimg_mitsuba(dir: str, extension: str = "exr") -> np.array:
    from mitsuba.core import Bitmap, Struct, float_dtype
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
        other = Bitmap(f"{dir}/frame_{i_xtframe}.{extension}")
        #     .convert(Bitmap.PixelFormat.RGBA, Struct.Type.Float32, srgb_gamma=False)
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
    diff = img1 - img2
    show_image(diff)


def maxmin(img: np.array):
    if len(img.shape) == 3:
        axis = (0,1)
    else:
        axis = (0,1,2)

    maximum = np.amax(img, axis=axis)
    minimum = np.amin(img, axis=axis)
    print(f"Max: {str(maximum)} - Min: {str(minimum)}")


def write_video_custom(streakimg_ldr: np.array, filename: str):
    """
    Creates a video from a HDR streak image (dimensions [height, width, time, channels]) in RGB format. The tonemap is
    needed to transform the HDR streak image to a LDR streak image.

    :param streakimg_hdr:
    :param tonemap:
    """
    number_of_frames = streakimg_ldr.shape[2]
    # 1. Get the streak image (already done) and define the output
    writer = imageio.get_writer(filename + ".mp4", fps=10)
    # 2. Iterate over the streak img frames
    for i in range(number_of_frames):
        writer.append_data((streakimg_ldr[:, :, i, :3] * 255).astype(np.uint8))
        print(f"{i}/{number_of_frames}", end="\r")
    # 3. Write the video
    writer.close()


def write_frames(streakimg_ldr: np.array, folder: str):
    """
    Writes the frames separately of a HDR streak image.

    :param streakimg:
    :param filename:
    :return:
    """
    Path(folder).mkdir(parents=True, exist_ok=True)
    number_of_frames = streakimg.shape[2]
    # 2. Iterate over the streak img frames
    for i in range(number_of_frames):
        frame = (streakimg_ldr[:, :, i, :3] * 255).astype(np.uint8)
        imageio.imwrite(folder + f"/frame_{str(i)}.png", frame)
        print(f"{i}/{number_of_frames}", end="\r")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool for Streak Images")
    # Options for reading steady and streak image
    parser.add_argument('-d', '--dir', type=str, help="Directory where the steady and streak images are stored",
                        default="cbox")
    parser.add_argument('-fs', '--file-steady-image', type=str, help="Name of the image file (for steady image)",
                        default="scene")
    parser.add_argument('-ft', '--file-streak-image', type=str, help="Name of the folder containing the single images "
                                                                     "that compose the streak image",
                        default="transient_scene")
    parser.add_argument('-ext', '--extension', type=str, help="Name of the extension of the images", default="exr")
    # Option to show intermediate result
    parser.add_argument('-s', '--show', action="store_true", help="Show images or results visually")
    # Tonemapping options
    parser.add_argument('-n', '--normalize', action="store_true", help="Normalize values before applying tonemapper")
    parser.add_argument('-e', '--exposure', type=float, help="Exposure: 2^exposure_value", default=0)
    parser.add_argument('-o', '--offset', type=float, help="Offset: value + offset_value", default=0)
    parser.add_argument('-t', '--tonemapper', type=str, help="Tonemapper applied: SRGB, GAMMA, PN", default="SRGB")
    parser.add_argument('-g', '--gamma', type=float, help="Float value of the gamma", default=2.2)
    # Options for result
    parser.add_argument('-r', '--result', type=str, nargs='+', help="Result: video (v), frames (f)", default=["v"])
    args = parser.parse_args()

    # 1. Load steady image (if it exists)
    path_steady_img = args.dir + "/" + args.file_steady_image
    # Check if steady image exists for this scene
    steady_img_exists = False
    if glob.glob(path_steady_img + "." + args.extension):
        steady_img_exists = True
    # Read steady image
    steadyimg = None
    if steady_img_exists:
        print("Loading steady image")
        steadyimg = read_steadyimg_mitsuba(path_steady_img, extension=args.extension)
        steadyimg = steadyimg[:, :, :3]  # drop alpha
        maxmin(steadyimg)

    # 2. Load streak image
    print("Loading streak image")
    path_streak_img = args.dir + "/" + args.file_streak_image
    streakimg = read_streakimg_mitsuba(path_streak_img, extension=args.extension)
    streakimg = streakimg[:, :, :, :3]  # drop alpha
    streakimg_acc = streakimg2steadyimg(streakimg)
    streakimg_acc = streakimg_acc[:, :, :3]  # drop alpha

    # 3. Tonemapping for HDR images
    steadyimg_ldr = None
    if steadyimg is not None:
        print("Applying tonemap to steadyimg")
        steadyimg_ldr = tonemap(steadyimg,
                                normalize=args.normalize,
                                exposure=args.exposure,
                                offset=args.offset,
                                tonemapper=args.tonemapper,
                                gamma=args.gamma)
        maxmin(steadyimg_ldr)
    print("Applying tonemap to streak image accumulated")
    streakimg_acc_ldr = tonemap(streakimg_acc,
                                normalize=args.normalize,
                                exposure=args.exposure,
                                offset=args.offset,
                                tonemapper=args.tonemapper,
                                gamma=args.gamma)
    maxmin(streakimg_acc_ldr)

    # 4. Comparison of steady and streak (accumulated)
    if args.show and steadyimg is not None:
        diff_images(steadyimg, streakimg_acc)

    # 5. Show tonemapped images
    if args.show:
        if steadyimg_ldr is not None:
            steadyimg_ldr_BGR = cv.cvtColor(steadyimg_ldr, code=cv.COLOR_RGB2BGR)
            cv.imshow("Steady Image", steadyimg_ldr_BGR)
        streakimg_acc_ldr_BGR = cv.cvtColor(streakimg_acc_ldr, code=cv.COLOR_RGB2BGR)
        cv.imshow("Streak Image (Accumulated)", streakimg_acc_ldr_BGR)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # 6. Write LDR images
    if steadyimg_ldr is not None:
        print("Writing steady image")
        name_steadyimg_file = args.dir + "/steadyimg.png"
        imageio.imwrite(name_steadyimg_file, (steadyimg_ldr * 255).astype(np.uint8))
    print("Writing streak image accumulated")
    name_streakimg_file = args.dir + "/streakimg_accumulated.png"
    imageio.imwrite(name_streakimg_file, (streakimg_acc_ldr * 255).astype(np.uint8))

    # 7. Apply tonemap to streak image
    print("Applying tonemap to streak image")
    streakimg_ldr = tonemap(streakimg,
                            normalize=args.normalize,
                            exposure=args.exposure,
                            offset=args.offset,
                            tonemapper=args.tonemapper,
                            gamma=args.gamma)
    maxmin(streakimg_ldr)

    # 8. Write video of streak image
    if "v" in args.result:
        name_video_file = args.dir + "/streak_video"
        print("Writing streak image video")
        write_video_custom(streakimg_ldr, filename=name_video_file)

    if "f" in args.result:
        name_folder = args.dir + "/frames"
        print("Writing frames separately")
        write_frames(streakimg_ldr, folder=name_folder)
