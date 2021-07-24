#include <mitsuba/core/bitmap.h>
#include <mitsuba/render/streakimageblock.h>
#include <mitsuba/python/python.h>

MTS_PY_EXPORT(StreakImageBlock) {
    MTS_PY_IMPORT_TYPES(StreakImageBlock, ReconstructionFilter)
    MTS_PY_CLASS(StreakImageBlock, Object)
        .def(py::init<const ScalarVector2i &, uint32_t, float, size_t,
                const ReconstructionFilter *, const ReconstructionFilter *,
                bool, bool, bool, bool>(),
            "size"_a, "time"_a, "exposure_time"_a, "channel_count"_a, "filter"_a = nullptr,
             "time_filter"_a = nullptr, "warn_negative"_a = true, "warn_invalid"_a = true,
            "border"_a = true, "normalize"_a = false)
        .def("put", py::overload_cast<const StreakImageBlock *>(&StreakImageBlock::put),
            D(StreakImageBlock, put), "block"_a)
        .def("put", vectorize(
             [](StreakImageBlock &ib, const Point2f &pos, const wavelength_t<Spectrum> &wavelengths,
                 const std::vector<std::tuple<Float, Spectrum, Mask>> &values, const Float &alpha) {
               std::vector<RadianceSample<Float, Spectrum, Mask>> values_transformed;
               for(const auto &[time, data, mask] : values) {
                   RadianceSample<Float, Spectrum, Mask> rs = RadianceSample<Float, Spectrum, Mask>(time, data, mask);
                   values_transformed.push_back(rs);
               }
               ib.put(pos, wavelengths, values_transformed, alpha);}
             ), "pos"_a, "wavelengths"_a, "radianceSamplesRecordVector"_a, "alpha"_a = 1.f)
        .def("put",
            [](StreakImageBlock &ib, const Point2f &pos,
                const std::vector<std::tuple<Float, std::vector<Float>, Mask>> &values) {
                std::vector<RadianceSample<Float, const Float *, Mask>> values_transformed;
                for(const auto &[time, data, mask] : values) {
                    values_transformed.emplace_back(time, data.data(), mask);
                }
                ib.put(pos, values_transformed);
            }, "pos"_a, "values"_a)
        .def_method(StreakImageBlock, clear)
        .def_method(StreakImageBlock, set_offset, "offset"_a)
        .def_method(StreakImageBlock, offset)
        .def_method(StreakImageBlock, size)
        .def_method(StreakImageBlock, time)
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
        .def("data", py::overload_cast<>(&StreakImageBlock::data, py::const_), D(StreakImageBlock, data));
}