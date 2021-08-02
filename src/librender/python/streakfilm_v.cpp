#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/imageblock.h>
#include <mitsuba/render/streakfilm.h>
#include <mitsuba/render/streakimageblock.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/python/python.h>

/// Trampoline for derived types implemented in Python
MTS_VARIANT class PyStreakFilm : public StreakFilm<Float, Spectrum> {
public:
    MTS_IMPORT_TYPES(StreakFilm, ImageBlock, StreakImageBlock)

    PyStreakFilm(const Properties &props) : StreakFilm(props) { }

    void prepare(const std::vector<std::string> &channels) override {
        PYBIND11_OVERLOAD_PURE(void, StreakFilm, prepare, channels);
    }

    void put(const ImageBlock *block) override {
        PYBIND11_OVERLOAD_PURE(void, StreakFilm, put, block);
    }

    void put(const StreakImageBlock *block) override {
        PYBIND11_OVERLOAD_PURE(void, StreakFilm, put, block);
    }

    void develop() override {
        PYBIND11_OVERLOAD_PURE(void, StreakFilm, develop, );
    }

    bool develop(const ScalarPoint2i  &offset, const ScalarVector2i &size,
                 const ScalarPoint2i  &target_offset, Bitmap *target) const override {
        PYBIND11_OVERLOAD_PURE(bool, StreakFilm, develop, offset, size, target_offset, target);
    }

    ref<Bitmap> bitmap(bool raw = false) override {
        PYBIND11_OVERLOAD_PURE(ref<Bitmap>, StreakFilm, bitmap, raw);
    }

    ref<Bitmap> bitmap(int slice, bool raw) override {
        PYBIND11_OVERLOAD_PURE(ref<Bitmap>, StreakFilm, bitmap, slice, raw);
    }

    void set_destination_file(const fs::path &filename) override {
        PYBIND11_OVERLOAD_PURE(void, StreakFilm, set_destination_file, filename);
    }

    bool destination_exists(const fs::path &basename) const override {
        PYBIND11_OVERLOAD_PURE(bool, StreakFilm, destination_exists, basename);
    }

    std::string to_string() const override {
        PYBIND11_OVERLOAD_PURE(std::string, StreakFilm, to_string, );
    }

    ref<StreakImageBlock> getStreakImageBlock() const override {
        PYBIND11_OVERLOAD_PURE(ref<StreakImageBlock>, StreakFilm, getStreakImageBlock, );
    }
};

// TODO: refactor this not to use PyStreakFilm (trampoline) because the problem
//       of StreakHDRFilm not being the correct class was in main_v.cpp
MTS_PY_EXPORT(StreakFilm) {
    MTS_PY_IMPORT_TYPES(StreakFilm, Film)
    using PyStreakFilm = PyStreakFilm<Float, Spectrum>;

    py::class_<StreakFilm, PyStreakFilm, Film, ref<StreakFilm>>(m, "StreakFilm", D(StreakFilm))
        .def(py::init<const Properties&>())
        .def("put", py::overload_cast<const StreakImageBlock *>(&StreakFilm::put), "block"_a)
        .def("bitmap", py::overload_cast<bool>(&StreakFilm::bitmap), "raw"_a = false)
        .def("bitmap", py::overload_cast<int, bool>(&StreakFilm::bitmap), "slice"_a, "raw"_a)
        .def_method(StreakFilm, time)
        .def_method(StreakFilm, exposure_time)
        .def_method(StreakFilm, time_offset)
        .def_method(StreakFilm, time_reconstruction_filter)
        .def_method(StreakFilm, getStreakImageBlock);

    MTS_PY_REGISTER_OBJECT("register_streakfilm", StreakFilm)
}
