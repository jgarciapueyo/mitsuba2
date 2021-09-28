#pragma once

#include <mitsuba/core/fwd.h>
#include <mitsuba/core/object.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/radiancesample.h>

NAMESPACE_BEGIN(mitsuba)

/**
 * \brief Storage for a streak image sub-block (a.k.a render bucket)
 *
 * This class is used by image-based parallel processes and encapsulates
 * computed rectangular spatial dimensions of a streak image (and all the temporal dimensions).
 * This allows for easy and efficient distributed rendering of large images.
 * Image blocks usually also include a border region storing contributions that
 * slightly outside of the block, which is required to support image reconstruction
 * filters.
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
     *
     * \param time_resolution
     *    Specifies the temporal dimension of the Streak Film.
     *
     * \param exposure_time
     *   Specifies the effective exposure time of every temporal window (time frame).
     *
     * \param time_offset
     *   Specifies the minimum path length recorded by the streak image block.
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

        std::vector<FloatTimeSample<Float, Mask>> values;
        for(const auto &xyzRecord: xyzVector) {
            FloatTimeSample<Float, Mask> color(-1);
            color.set_time(xyzRecord.time, xyzRecord.mask);
            // Reversed
            color.push_front(1.f);
            color.push_front(select(xyzRecord.mask, Float(1.f), Float(0.f)));
            color.push_front(xyzRecord.radiance.z());
            color.push_front(xyzRecord.radiance.y());
            color.push_front(xyzRecord.radiance.x());
            values.push_back(color);
        }
        put(pos, values);
    }


    /**
     * \brief Store a list of samples inside the block.
     *
     * \note This method is only valid if a reconstruction filter was provided
     * when the block was constructed.
     *
     * \param pos
     *    Denotes the sample position in fractional pixel coordinates. It is
     *    not checked, and so must be valid. The block's offset is subtracted
     *    from the given position to obtain the final pixel position.
     *
     * \param values
     *    List of samples. Each sample contains an array containing each channel
     *    of the sample values. The array length of each sample must match the
     *    length given by \ref channel_count()
     */
    void put(const Point2f &pos, const std::vector<FloatTimeSample<Float, Mask>> &values);

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
