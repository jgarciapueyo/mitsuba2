#pragma once

#include <mitsuba/core/fwd.h>
#include <mitsuba/core/object.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/radiancesample.h>

NAMESPACE_BEGIN(mitsuba)

/**
 * TODO:
 */
template <typename Float, typename Spectrum>
class MTS_EXPORT_RENDER StreakImageBlock : public Object {
public:
    MTS_IMPORT_TYPES(ReconstructionFilter)

    /**
     * Construct a new streak image block of the requested properties
     *
     * \param size
     *    Specifies the block dimensions (not accounting for additional
     *    border pixels required to support image reconstruction filters).
     *    Because it is a Streak Image Block, the size is 3D.
     *
     * // TODO: see if StreakImageBlock should be agnostic to this and the third
     * //       dimension should be part of size as Vector3i and delete time_resolution/exposuretime
     * //       since that would be part of the film containing the streak img block
     * \param time_resolution
     *
     * \param exposure_time
     *
     * \param channel_count
     *    Specifies the number of image channels.
     *
     * \param filter
     *    Pointer to the film's reconstruction filter. If passed, it is used to
     *    compute and store reconstruction weights. Note that it is mandatory
     *    when any of the block's \ref put operations are used, except for
     *    \c put(const StreakImageBlock*).
     *
     * \param time_filter
     *    Pointer to the film's time reconstruction filter. If passed, it is
     *    used to compute and store the time reconstruction weights. Note that
     *    it is mandatory when any of the block's \ref put operations are used,
     *    except for \c put(const StreakImageBlock*).
     *
     * \param warn_negative
     *    Warn when writing samples with negative components?
     *
     * \param warn_invalid
     *    Warn when writing samples with components that are equal to
     *    NaN (not a number) or +/- infinity?
     *
     * \param border
     *    Allocate a border region around the image block to support
     *    contributions to adjacent pixels when using wide (i.e. non-box)
     *    reconstruction filters?
     *
     * \param normalize
     *    Ensure that splats created via ``ImageBlock::put()`` add a
     *    unit amount of energy? Stratified sampling techniques that
     *    sample rays in image space should set this to \c false, since
     *    the samples will eventually be divided by the accumulated
     *    sample weight to remove any non-uniformity.
     */
    StreakImageBlock(const ScalarVector2i &size,
               int32_t time,
               float exposure_time,
               float time_offset,
               size_t channel_count,
               const ReconstructionFilter *filter = nullptr,
               const ReconstructionFilter *time_filter = nullptr,
               bool warn_negative = true,
               bool warn_invalid = true,
               bool border = true,
               bool normalize = false);

    /// Accumulate another streak image block into this one
    void put(const StreakImageBlock *block);

    /**
     * \brief Store a vector of samples / packets of samples inside the streak
     * image block.
     *
     * \note This method is only valid if a reconstruction filter was given at
     * the construction of the block.
     *
     * \param pos
     *    Denotes the sample position in fractional pixel coordinates. It is
     *    not checked, and so must be valid. The block's offset is subtracted
     *    from the given position to obtain the final pixel position.
     *
     * \param wavelengths
     *    Sample wavelengths in nanometers
     *
     * \param radianceSampleVector
     *    Samples value associated with the specified wavelengths
     *
     * \param alpha
     *    Alpha value assocated with the sample
     *
     * \return \c false if one of the sample values was \a invalid, e.g.
     *    NaN or negative. A warning is also printed if \c m_warn_negative
     *    or \c m_warn_invalid is enabled.
     */
    void put(const Point2f &pos,
             const Wavelength &wavelengths,
             const std::vector<RadianceSample<Float, Spectrum, Mask>> &radianceSampleVector,
             const Float &alpha) {
        if (unlikely(m_channel_count != 5))
            Throw("ImageBlock::put(): non-standard image block configuration! (AOVs?)");
        std::vector<RadianceSample<Float, UnpolarizedSpectrum, Mask>> radianceSampleVector_u = {};
        for(const auto &radianceSampleRecord : radianceSampleVector) {
            radianceSampleVector_u.emplace_back(
                radianceSampleRecord.time,
                depolarize(radianceSampleRecord.radiance),
                radianceSampleRecord.mask
                );
        }
        std::vector<RadianceSample<Float, Color3f, Mask>> xyzVector;
        if constexpr (is_monochromatic_v<Spectrum>) {
            ENOKI_MARK_USED(wavelengths);
            for(const auto &radianceSampleRecord_u : radianceSampleVector_u) {
                xyzVector.emplace_back(
                    radianceSampleRecord_u.time,
                    radianceSampleRecord_u.radiance.x(),
                    radianceSampleRecord_u.mask
                    );
            }
        } else if constexpr (is_rgb_v<Spectrum>) {
            ENOKI_MARK_USED(wavelengths);
            for(const auto &radianceSampleRecord_u : radianceSampleVector_u) {
                xyzVector.emplace_back(
                    radianceSampleRecord_u.time,
                    srgb_to_xyz(radianceSampleRecord_u.radiance, radianceSampleRecord_u.mask),
                    radianceSampleRecord_u.mask
                );
            }
        } else {
            static_assert(is_spectral_v<Spectrum>);
            for(const auto &radianceSampleRecord_u : radianceSampleVector_u) {
                xyzVector.emplace_back(
                    radianceSampleRecord_u.time,
                    spectrum_to_xyz(radianceSampleRecord_u.radiance, wavelengths, radianceSampleRecord_u.mask),
                    radianceSampleRecord_u.mask
                );
            }
        }

        std::vector<RadianceSample<Float, std::array<Float, 5>, Mask>> values;
        for(const auto &xyzRecord: xyzVector) {
            std::array<Float, 5> color;
            color[0] = xyzRecord.radiance.x();
            color[1] = xyzRecord.radiance.y();
            color[2] = xyzRecord.radiance.z();
            color[3] = alpha;
            color[4] = 1;
            values.emplace_back(xyzRecord.time, color, xyzRecord.mask);
        }
        // TODO: before all the put methods returned Mask.
        put(pos, values);
    }

    // TODO: check if this approach of std::array<Float, 5> works for aovs
    void put(const Point2f &pos, const std::vector<RadianceSample<Float, std::array<Float, 5>, Mask>> &values);

    /// Clear everything to zero.
    void clear();

    // =============================================================
    //! @{ \name Accesors
    // =============================================================

    /**\brief Set the current block offset.
     *
     * This corresponds to the offset from the top-left corner of a larger
     * image (e.g. a Film) to the top-left corner of this ImageBlock instance.
     */
    void set_offset(const ScalarPoint2i &offset) { m_offset = offset; }

    /// Set the block size. This potentially destroys the block's content.
    void set_size(const ScalarVector2i &size, int32_t time);

    /// Return the current block offset
    const ScalarPoint2i &offset() const { return m_offset; }

    /// Return the current block size
    const ScalarVector2i &size() const { return m_size; }

    /// Return the current block time (length)
    size_t time() const { return m_time; }

    /// Return the current exposure time for each time bin
    float exposure_time() const { return m_exposure_time; }

    /// Return the current exposure time for each time bin
    float time_offset() const { return m_time_offset; }

    /// Return the bitmap's width in pixels
    size_t width() const { return m_size.x(); }

    /// Return the bitmap's height in pixels
    size_t height() const { return m_size.y(); }

    /// Return the bitmap's length in pixels
    // TODO: change if size is transformed to a Vector3i and delete m_exposure_time
    size_t length() const { return m_time; }

    /// Warn when writing invalid (NaN, +/- infinity) sample values?
    void set_warn_invalid(bool value) { m_warn_invalid = value; }

    /// Warn when writing invalid (NaN, +/- infinity) sample values?
    bool warn_invalid() const { return m_warn_invalid; }

    /// Warn when writing negative sample values?
    void set_warn_negative(bool value) { m_warn_negative = value; }

    /// Warn when writing negative sample values?
    bool warn_negative() const { return m_warn_negative; }

    /// Return the number of channels stored by the image block
    size_t channel_count() const { return (size_t) m_channel_count; }

    /// Return the border region used by the reconstruction filter
    int border_size() const { return m_border_size; }

    /// Return the border region used by the reconstruction filter (temporal)
    int time_border_size() const { return m_time_border_size; };

    /// Return the underlying pixel buffer
    DynamicBuffer<Float> &data() { return m_data; }

    /// Return the underlying pixel buffer (const version)
    const DynamicBuffer<Float> &data() const { return m_data; }

    DynamicBuffer<Float> data(int slice) const;

    //! @}
    // =============================================================

    std::string to_string() const override;

    MTS_DECLARE_CLASS()
protected:
    /// Virtual destructor
    virtual ~StreakImageBlock();
protected:
    ScalarPoint2i m_offset;
    ScalarVector2i m_size;
    int32_t m_time;
    float m_exposure_time;
    float m_time_offset;
    uint32_t m_channel_count;
    int m_border_size;
    int m_time_border_size;
    DynamicBuffer<Float> m_data;
    const ReconstructionFilter *m_filter;
    const ReconstructionFilter *m_time_filter;
    Float *m_weights_x, *m_weights_y;
    bool m_warn_negative;
    bool m_warn_invalid;
    bool m_normalize;
};

MTS_EXTERN_CLASS_RENDER(StreakImageBlock)
NAMESPACE_END(mitsuba)
