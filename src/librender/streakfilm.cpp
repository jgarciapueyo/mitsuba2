#include <mitsuba/render/streakfilm.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>

NAMESPACE_BEGIN(mitsuba)

MTS_VARIANT StreakFilm<Float, Spectrum>::StreakFilm(const Properties &props) : Base(props) {
    m_time = props.int_("time", 1000);
    m_exposure_time = props.float_("exposure_time", 1);
    m_time_offset = props.float_("time_offset", 0);

    // Use the provided reconstruction filter, if any.
    for (auto &[name, obj] : props.objects(false)) {
        if(name != "tfilter") {
           continue;
        }
        auto *tfilter = dynamic_cast<ReconstructionFilter *>(obj.get());
        if (tfilter) {
            if (m_time_filter)
                Throw("A film can only have one reconstruction filter.");
            m_time_filter = tfilter;
            props.mark_queried(name);
        }
    }

    if (!m_time_filter) {
        // No reconstruction filter has been selected. Load a Gaussian filter by default
        m_time_filter =
            PluginManager::instance()->create_object<ReconstructionFilter>(Properties("gaussian"));
    }
}

// MTS_VARIANT void StreakFilm<Float, Spectrum>::put(const ImageBlock *block) {
//    NotImplementedError("put");
// }

MTS_VARIANT StreakFilm<Float, Spectrum>::~StreakFilm() {}

MTS_VARIANT std::string StreakFilm<Float, Spectrum>::to_string() const {
    std::ostringstream oss;
    oss << "StreakFilm[" << std::endl
        << "  size = "        << m_size        << "," << std::endl
        << "  crop_size = "   << m_crop_size   << "," << std::endl
        << "  crop_offset = " << m_crop_offset << "," << std::endl
        << "  time = "        << m_time        << "," << std::endl
        << "  exposure_time = " << m_exposure_time << "," << std::endl
        << "  time_offset = "      << m_time_offset      << "," << std::endl
        << "  high_quality_edges = " << m_high_quality_edges << "," << std::endl
        << "  filter = " << m_filter << std::endl
        << "  time_filter = " << m_time_filter << std::endl
        << "]";
    return oss.str();
}


MTS_IMPLEMENT_CLASS_VARIANT(StreakFilm, Film)
MTS_INSTANTIATE_CLASS(StreakFilm)
NAMESPACE_END(mitsuba)
