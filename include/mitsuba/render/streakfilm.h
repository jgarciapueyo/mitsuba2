#pragma once

#include <mitsuba/mitsuba.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/object.h>
#include <mitsuba/core/rfilter.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/render/film.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/fwd.h>

NAMESPACE_BEGIN(mitsuba)

/**
 * TODO: add documentation
 * */
template <typename Float, typename Spectrum>
class MTS_EXPORT_RENDER StreakFilm : public Film<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Film, m_size, m_crop_size, m_crop_offset,
                    m_filter, m_high_quality_edges, bitmap)
    MTS_IMPORT_TYPES(ImageBlock, StreakImageBlock, ReconstructionFilter)

    /// Merge an image block into the film. This methods should be thread-safe.
    // void put(const ImageBlock *block) override;

    /// Merge an image block into the film. This methods should be thread-safe.
    virtual void put(const StreakImageBlock *block) = 0;

    // TODO: fill documentation
    virtual ref<Bitmap> bitmap(int slice, bool raw) = 0;

    // =============================================================
    //! @{ \name Accessor functions
    // =============================================================

    size_t time() const { return m_time; }

    float exposure_time() const { return m_exposure_time; }

    float time_offset() const { return m_time_offset; }

    const ReconstructionFilter *time_reconstruction_filter() const {
        return m_time_filter.get();
    }

    //! @}
    // =============================================================

    virtual std::string to_string() const override;

    virtual ref<StreakImageBlock> getStreakImageBlock() const = 0;

    MTS_DECLARE_CLASS()
protected:

    /// Create a film
    StreakFilm(const Properties &props);

    /// Virtual destructor
    virtual ~StreakFilm();

protected:
    uint32_t m_time; // TODO: look for better naming: frames, length, etc ..
    float m_exposure_time; // TODO: look for better naming: exposure_per_frame, ...
    float m_time_offset; // TODO: look for more uniformity in m_time, m_exposure_time and m_offset
    ref<ReconstructionFilter> m_time_filter;
};

MTS_EXTERN_CLASS_RENDER(StreakFilm)
NAMESPACE_END(mitsuba)
