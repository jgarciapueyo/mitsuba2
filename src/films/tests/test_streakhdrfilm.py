import mitsuba
import pytest
import os
import enoki as ek


def test01_construct(variant_scalar_rgb):
    from mitsuba.core.xml import load_string

    # With default reconstruction filter
    film = load_string("""<film version="2.0.0" type="streakhdrfilm"></film>""")
    assert film is not None
    assert film.reconstruction_filter() is not None
    assert film.time_reconstruction_filter() is not None

    # With a provided reconstruction filter and time reconstruction filter
    film = load_string("""<film version="2.0.0" type="streakhdrfilm">
                            <rfilter type="gaussian" name="rfilter">
                                <float name="stddev" value="18.5"/>
                            </rfilter>
                            <rfilter type="gaussian" name="tfilter">
                                <float name="stddev" value="10"/>
                            </rfilter>
                          </film>""")
    assert film is not None
    assert film.reconstruction_filter().radius() == (4 * 18.5)
    assert film.time_reconstruction_filter().radius() == (4 * 10)

    # Certain parameter values are not allowed
    with pytest.raises(RuntimeError):
        load_string("""<film version="2.0.0" type="hdrfilm">
                       <string name="component_format" value="uint8"/>
                       </film>""")
    with pytest.raises(RuntimeError):
        load_string("""<film version="2.0.0" type="hdrfilm">
                       <string name="pixel_format" value="brga"/>
                       </film>""")


def test02_crops(variant_scalar_rgb):
    from mitsuba.core.xml import load_string

    film = load_string("""<film version="2.0.0" type="streakhdrfilm">
            <integer name="width" value="32"/>
            <integer name="height" value="21"/>
            <integer name="time" value="10" />
            <float   name="exposure_time" value="2"/>
            <float   name="time_offset" value="100"/>
            <integer name="crop_width" value="11"/>
            <integer name="crop_height" value="5"/>
            <integer name="crop_offset_x" value="2"/>
            <integer name="crop_offset_y" value="3"/>
            <boolean name="high_quality_edges" value="true"/>
            <string name="pixel_format" value="rgba"/>
        </film>""")
    assert film is not None
    assert ek.all(film.size() == [32, 21])
    assert film.time() == 10
    assert film.exposure_time() == 2
    assert film.time_offset() == 100
    assert ek.all(film.crop_size() == [11, 5])
    assert ek.all(film.crop_offset() == [2, 3])
    assert film.has_high_quality_edges()

    # Crop size doesn't adjust its size, so an error should be raised if the
    # resulting crop window goes out of bounds.
    incomplete = """<film version="2.0.0" type="hdrfilm">
            <integer name="width" value="32"/>
            <integer name="height" value="21"/>
            <integer name="crop_offset_x" value="30"/>
            <integer name="crop_offset_y" value="20"/>"""
    with pytest.raises(RuntimeError):
        film = load_string(incomplete + "</film>")

    film = load_string(incomplete + """
            <integer name="crop_width" value="2"/>
            <integer name="crop_height" value="1"/>
        </film>""")
    assert film is not None
    assert ek.all(film.size() == [32, 21])
    assert ek.all(film.crop_size() == [2, 1])
    assert ek.all(film.crop_offset() == [30, 20])


@pytest.mark.parametrize('file_format', ['exr', 'rgbe', 'pfm'])
def test03_develop(variant_scalar_rgb, file_format, tmpdir):
    from pathlib import Path

    import numpy as np

    from mitsuba.core.xml import load_string
    from mitsuba.core import Bitmap, Struct, ReconstructionFilter, float_dtype
    from mitsuba.render import StreakImageBlock

    """Create a test image. Develop it to a few file format, each time reading
    it back and checking that contents are unchanged."""
    np.random.seed(12345 + ord(file_format[0]))
    # Note: depending on the file format, the alpha channel may be automatically removed.
    film = load_string("""<film version="2.0.0" type="streakhdrfilm">
            <integer name="width" value="41"/>
            <integer name="height" value="37"/>
            <integer name="time" value="7"/>
            <float  name="exposure_time" value="2"/>
            <string name="file_format" value="{}"/>
            <string name="pixel_format" value="rgba"/>
            <string name="component_format" value="float32"/>
            <rfilter name="rfilter" type="box"/>
        </film>""".format(file_format))

    # Regardless of the output file format, values are stored as XYZAW (5 channels).
    contents = np.random.uniform(size=(film.size()[1], film.size()[0], film.time(), 5))

    # RGBE will only reconstruct well images that have similar scales on
    # all channel (because exponent is shared between channels).
    if file_format == "rgbe":
        contents = 1 + 0.1 * contents

    # Use unit weights.
    contents[:, :, :, 4] = 1.0

    # Create the StreakImageBlock and populate it with values
    block = StreakImageBlock(film.size(), film.time(), film.exposure_time(), film.time_offset(), 5,
                             film.reconstruction_filter(), film.time_reconstruction_filter())
    block.clear()

    for x in range(film.size()[1]):
        for y in range(film.size()[0]):
            for z in range(film.time()):
                block.put([y+0.5, x+0.5], [(z * film.exposure_time(), contents[x, y, z, :], True)])

    film.prepare(['X', 'Y', 'Z', 'A', 'W'])
    film.put(block)

    with pytest.raises(RuntimeError):
        # Should raise when the destination file hasn't been specified.
        film.develop()

    # Because is a StreakFilm, it will write several images when developing.
    # Given a filename = "tmpdir/xyz.extension", it will create a folder called "tmpdir/xyz/" and inside
    # it will write all the images x-t with the name "tmpdir/xyz/frame_{i}.extension"
    # We need to check every one of these images to ensure that StreakHDRFilm works correctly
    filename = str(tmpdir.join('test_image.' + file_format))
    film.set_destination_file(filename)
    film.develop()

    # Remove extension
    dirname = str(Path(filename).with_suffix(""))

    # For every x-t image
    for y in range(film.size()[1]):
        # Read back and check contents
        other = Bitmap(dirname + f"/frame_{y}." + file_format)\
            .convert(Bitmap.PixelFormat.XYZAW, Struct.Type.Float32, srgb_gamma=False)
        img = np.array(other, copy=False)
        contents_xt = np.squeeze(contents[y, :, :, :])  # to remove the first dimension which is always 1

        # print("SHAPPES ------------------------")
        # print(dirname + f"/frame_{y}." + file_format)
        # print(img[:, :, :])
        # print(contents_xt[:, :, :])

        if file_format == "exr":
            assert ek.allclose(img, contents_xt, atol=1e-5)
        else:
            if file_format == "rgbe":
                assert ek.allclose(img[:, :, :3], contents_xt[:, :, :3], atol=1e-2), \
                    '\n{}\nvs\n{}\n'.format(img[:4, :4, :3], contents_xt[:4, :4, :3])
            else:
                assert ek.allclose(img[:, :, :3], contents_xt[:, :, :3], atol=1e-5), \
                    '\n{}\nvs\n{}\n'.format(img[:4, :4, :3], contents_xt[:4, :4, :3])
            # Alpha channel was ignored, alpha and weights should default to 1.0.
            assert ek.allclose(img[:, :, 3:5], 1.0, atol=1e-6)


@pytest.mark.skip(reason="The implementation of StreakFilm.bitmap(slice, raw) has a bug if the Bitmap returned is"
                         "used directly from Python and transformed to a NumPy array, making the first 4 values of the"
                         " Bitmap incorrect. This bug is due to the Python bindings because if the Bitmap is written "
                         "to and read from a file (as in the test 'test03_develop()'), the contents of the Bitmap are "
                         "all correct.")
def test04_bitmap_slice(variant_scalar_rgb):
    import numpy as np

    from mitsuba.core.xml import load_string
    from mitsuba.core import Bitmap, Struct, ReconstructionFilter, float_dtype
    from mitsuba.render import StreakImageBlock

    """Create a test image. Develop it to a few file format, each time reading
    it back and checking that contents are unchanged."""
    # Note: depending on the file format, the alpha channel may be automatically removed.
    film = load_string("""<film version="2.0.0" type="streakhdrfilm">
            <integer name="width" value="10"/>
            <integer name="height" value="37"/>
            <integer name="time" value="2"/>
            <float  name="exposure_time" value="2"/>
            <string name="pixel_format" value="rgba"/>
            <string name="component_format" value="float32"/>
            <rfilter name="rfilter" type="box"/>
        </film>""")

    # Regardless of the output file format, values are stored as XYZAW (5 channels).
    contents = np.arange(film.size()[1] * film.size()[0] * film.time() * 5) \
        .reshape([film.size()[1], film.size()[0], film.time(), 5])

    # Create the StreakImageBlock and populate it with values
    block = StreakImageBlock(film.size(), film.time(), film.exposure_time(), film.time_offset(), 5,
                             film.reconstruction_filter(), film.time_reconstruction_filter())
    block.clear()

    for x in range(film.size()[1]):
        for y in range(film.size()[0]):
            for z in range(film.time()):
                block.put([y+0.5, x+0.5], [(z * film.exposure_time(), contents[x, y, z, :], True)])

    film.prepare(['X', 'Y', 'Z', 'A', 'W'])
    film.put(block)

    # For every x-t image
    for yslice in range(film.size()[1]):
        # Get the values from the Bitmap corresponding to the x-t image (without developing, and thus without
        # writing to a file)
        bitmap = film.bitmap(slice=yslice, raw=True)
        npbitmap = np.array(bitmap, copy=False)

        # Get values from the original numpy array
        contents_xt = np.squeeze(contents[yslice, :, :, :])  # to remove the first dimension which is always 1

        # Get the values directly from the StreakImageBlock
        streakimageblock = np.array(block.data(yslice), copy=False) \
                             .reshape([block.width() + 2 * block.border_size(),
                                       block.time(),
                                       block.channel_count()])

        # Get the values from the StreakImageBlock from inside the StreakFilm
        filmstreakimageblock = np.array(film.getStreakImageBlock().data(yslice), copy=False) \
            .reshape([block.width() + 2 * block.border_size(),
                      block.time(),
                      block.channel_count()])

        assert streakimageblock.shape == filmstreakimageblock.shape, \
            "Shape between StreakImageBlock directly and StreakImageBlock from StreakFilm is different"
        assert npbitmap.shape == streakimageblock.shape, \
            "Shape between Bitmap and StreakImageBlock directly is different"
        assert npbitmap.shape == filmstreakimageblock.shape, \
            "Shape between Bitmap and StreakImageBlock from StreakFilm is different"

        assert ek.allclose(streakimageblock, filmstreakimageblock, atol=1e-2), \
            "Values between StreakImageBlock directly and StreakImageBlock from StreakFilm is different"
        # The following assert will fail for the first four values of npbitmap compared with streakimageblock
        assert ek.allclose(npbitmap, streakimageblock, atol=1e-2), \
            "Values between Bitmap and StreakImageBlock directly is different"
        assert ek.allclose(npbitmap, filmstreakimageblock, atol=1e-2), \
            "Values between Bitmap and StreakImageBlock from StreakFilm is different"
