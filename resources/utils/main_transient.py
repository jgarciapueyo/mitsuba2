import imageio
import numpy as np
import cv2 as cv
import glob
import mitsuba
import matplotlib.pyplot as plt
import OpenEXR
import Imath

mitsuba.set_variant("scalar_rgb")


def read_steadyimg_mitsuba(dir: str, extension: str = "exr") -> np.array:
    from mitsuba.core import Bitmap, Struct, float_dtype
    # other = Bitmap(f"{dir}.{extension}") \
    #     .convert(Bitmap.PixelFormat.XYZA, Struct.Type.Float32, srgb_gamma=False)
    other = Bitmap(f"{dir}.{extension}")\
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
        other = Bitmap(f"{dir}/frame_{i_xtframe}.{extension}")\
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

if __name__ == "__main__":
    # READ IMG MITSUBA
    print("READ STREAK IMG MITSUBA")
    steadyimg_mitsuba = read_steadyimg_mitsuba("../../resources/data/scenes/cbox/cbox", extension="exr")
    steadyimg_mitsuba = steadyimg_mitsuba[:,:,:3]
    print(steadyimg_mitsuba.shape)
    print(steadyimg_mitsuba.dtype)
    print(np.count_nonzero(np.isfinite(steadyimg_mitsuba[:, :, :3])))

    # READ STREAK IMG AND ACCUMULATE
    print("READ STREAK IMG")
    streakimg_mitsuba_original = read_streakimg_mitsuba("../../resources/data/scenes/cbox/cbox_transient", extension="exr")
    print("Original Streak: Min Value: " + str(np.amin(streakimg_mitsuba_original[:,:,:,:3], axis=(0,1,2))) + " - Max Value: " + str(np.amax(streakimg_mitsuba_original[:,:,:,:3], axis=(0,1,2))))
    streakimg_mitsuba = streakimg2steadyimg(streakimg_mitsuba_original)
    streakimg_mitsuba = streakimg_mitsuba[:, :, :3]  # drop alpha
    print(streakimg_mitsuba_original.shape)
    print(streakimg_mitsuba_original.dtype)
    print(np.count_nonzero(np.isfinite(streakimg_mitsuba_original[:, :, :, :3])))
    print(streakimg_mitsuba.shape)
    print(streakimg_mitsuba.dtype)
    print(np.count_nonzero(np.isfinite(streakimg_mitsuba[:, :, :3])))

    # Comparison of streak (accumulated) and steady
    if True:
        diff = streakimg_mitsuba - steadyimg_mitsuba
        plt.imshow(diff[:,:,0])
        plt.colorbar()
        plt.show()
        plt.imshow(diff[:,:,1])
        plt.colorbar()
        plt.show()
        plt.imshow(diff[:,:,2])
        plt.colorbar()
        # plt.show()
    print("Steady: Min Value: " + str(np.amin(steadyimg_mitsuba[:,:,:3], axis=(0,1))) + " - Max Value: " + str(np.amax(steadyimg_mitsuba[:,:,:3], axis=(0,1))))
    print("Streak: Min Value: " + str(np.amin(streakimg_mitsuba[:,:,:3], axis=(0,1))) + " - Max Value: " + str(np.amax(streakimg_mitsuba[:,:,:3], axis=(0,1))))

    # TONEMAPPING AFTER SEEING THAT MITSUBA/OPENEXR/IMAGEIO are the same
    print("TONEMAPPING")
    tonemap = cv.createTonemapReinhard(gamma=2.2, light_adapt=0., color_adapt=0.)
    # steadyimg_mitsuba_RGB = cv.cvtColor(steadyimg_mitsuba, code=cv.COLOR_XYZ2RGB)
    steadyimg_mitsuba_RGB = steadyimg_mitsuba
    steadyimg_mitsuba_ldr_RGB = tonemap.process(steadyimg_mitsuba_RGB)
    steadyimg_mitsuba_ldr_BGR = cv.cvtColor(steadyimg_mitsuba_ldr_RGB, code=cv.COLOR_RGB2BGR)

    # TONEMAPPING STREAKIMG accumulated
    # streakimg_mitsuba_RGB = cv.cvtColor(streakimg_mitsuba, code=cv.COLOR_XYZ2RGB)
    streakimg_mitsuba_RGB = streakimg_mitsuba
    streakimg_mitsuba_ldr_RGB = tonemap.process(streakimg_mitsuba_RGB)
    streakimg_mitsuba_ldr_BGR = cv.cvtColor(streakimg_mitsuba_ldr_RGB, code=cv.COLOR_RGB2BGR)

    # SHOW THE IMGS
    if True:
        cv.imshow("STEADY", steadyimg_mitsuba_ldr_BGR)
        cv.imshow("ACCUMULATED", streakimg_mitsuba_ldr_BGR)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # exit()
    # WRITING THE IMGS
    imageio.imwrite("../../resources/data/scenes/cbox/cbox_processed_RGB.png", (steadyimg_mitsuba_ldr_RGB * 255).astype(np.uint8))
    imageio.imwrite("../../resources/data/scenes/cbox/cbox_accumulated_RGB.png", (streakimg_mitsuba_ldr_RGB * 255).astype(np.uint8))

    # WRITING THE VIDEO
    if False:
        # 1. Get the streak image (already done) and define the output
        writer = imageio.get_writer("../../resources/data/scenes/cbox/cbox_transient.mp4", fps=10)
        # 2. Iterate over the streak img
        for i in range(streakimg_mitsuba_original.shape[2]):
            img = np.squeeze(streakimg_mitsuba_original[:, :, i, :3])
            img_RGB = img
            img_ldr_RGB = tonemap.process(img_RGB)
            img_ldr_BGR = cv.cvtColor(img_ldr_RGB, cv.COLOR_RGB2BGR)
            """
            cv.imshow("ACCUMULATED", (img_ldr_BGR * 255).astype(np.uint8))
            cv.waitKey(0)
            """
            writer.append_data((img_ldr_RGB * 255).astype(np.uint8))
        writer.close()
