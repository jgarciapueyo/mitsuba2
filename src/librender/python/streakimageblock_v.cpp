#include <mitsuba/core/bitmap.h>
#include <mitsuba/render/streakimageblock.h>
#include <mitsuba/python/python.h>

MTS_PY_EXPORT(StreakImageBlock) {
    MTS_PY_IMPORT_TYPES(StreakImageBlock, ReconstructionFilter)
    MTS_PY_CLASS(StreakImageBlock, Object)
        .def(py::init<const ScalarVector2i &, int32_t, float, float, size_t,
                const ReconstructionFilter *, const ReconstructionFilter *,
                bool, bool, bool, bool>(),
            "size"_a, "time"_a, "exposure_time"_a, "time_offset"_a, "channel_count"_a, "filter"_a = nullptr,
             "time_filter"_a = nullptr, "warn_negative"_a = true, "warn_invalid"_a = true,
            "border"_a = true, "normalize"_a = false)
        .def("put", py::overload_cast<const StreakImageBlock *>(&StreakImageBlock::put),
            D(StreakImageBlock, put), "block"_a)
        /**
        .def("put",
             // TODO: for this version, vectorize has been deleted
             // TODO: check if it is possible to declare values as a vector of RadianceSample directly so there is no need to iterate to transform them (an thus, no need for the wrapper)
             [](StreakImageBlock &ib, const Point2f &pos, const wavelength_t<Spectrum> &wavelengths,
                 const std::vector<std::tuple<Float, Spectrum, Mask>> &values, const Float &alpha) {
               std::vector<RadianceSample<Float, Spectrum, Mask>> values_transformed;
               for(const auto &[time, data, mask] : values) {
                   values_transformed.emplace_back(time, data, mask);
               }
               ib.put(pos, wavelengths, values_transformed, alpha);
             }, "pos"_a, "wavelengths"_a, "radianceSamplesRecordVector"_a, "alpha"_a = 1.f)
        .def("put",
            // TODO: check if it is possible to declare values as a vector of RadianceSample directly so there is no need to iterate to transform them (an thus, no need for the wrapper)
            [](StreakImageBlock &ib, const Point2f &pos,
                const std::vector<std::tuple<Float, std::vector<Float>, Mask>> &values) {
                std::vector<RadianceSample<Float, std::array<Float, 5>, Mask>> values_transformed;
                for(const auto &[time, data, mask] : values) {
                    std::array<Float, 5> data_array;
                    std::copy_n(data.begin(), 5, data_array.begin());
                    values_transformed.emplace_back(time, data_array, mask);
                }
                ib.put(pos, values_transformed);
            }, "pos"_a, "values"_a)
            **/
        .def_method(StreakImageBlock, clear)
        .def_method(StreakImageBlock, set_offset, "offset"_a)
        .def_method(StreakImageBlock, offset)
        .def_method(StreakImageBlock, size)
        .def_method(StreakImageBlock, time)
        .def_method(StreakImageBlock, exposure_time)
        .def_method(StreakImageBlock, time_offset)
        .def_method(StreakImageBlock, width)
        .def_method(StreakImageBlock, height)
        .def_method(StreakImageBlock, length)
        .def_method(StreakImageBlock, warn_invalid)
        .def_method(StreakImageBlock, warn_negative)
        .def_method(StreakImageBlock, set_warn_invalid, "value"_a)
        .def_method(StreakImageBlock, set_warn_negative, "value"_a)
        .def_method(StreakImageBlock, border_size)
        .def_method(StreakImageBlock, time_border_size)
        .def_method(StreakImageBlock, channel_count)
        // TODO: check about D(..) because both are using same docstring from include/python/docstr.h
        .def("data", py::overload_cast<>(&StreakImageBlock::data, py::const_), D(StreakImageBlock, data))
        .def("data", py::overload_cast<int>(&StreakImageBlock::data, py::const_), "slice"_a, D(StreakImageBlock, data));
}