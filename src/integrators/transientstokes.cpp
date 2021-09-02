#include <mitsuba/render/transientintegrator.h>
#include <mitsuba/render/records.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class TransientStokesIntegrator final : public TransientSamplingIntegrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(TransientSamplingIntegrator)
    MTS_IMPORT_TYPES(Scene, Sampler, Medium)

    TransientStokesIntegrator(const Properties &props) : Base(props) {
        if constexpr (!is_polarized_v<Spectrum>)
            Throw("This integrator should only be used in polarized mode!");
        for (auto &kv : props.objects()) {
            Base *integrator = dynamic_cast<Base *>(kv.second.get());
            if (!integrator)
                Throw("Child objects must be of type 'TransientSamplingIntegrator'!");
            if (m_integrator)
                Throw("More than one sub-integrator specified!");
            m_integrator = integrator;
        }

        if (!m_integrator)
            Throw("Must specify a sub-integrator!");
    }

    void sample(const Scene *scene, Sampler *sampler, const RayDifferential3f &ray,
                const Medium * medium,
                std::vector<FloatTimeSample<Float, Mask>> & aovsRecordVector,
                Mask active,
                std::vector<RadianceSample<Float, Spectrum, Mask>> &radianceSamplesRecordVector) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        m_integrator->sample(scene, sampler, ray, medium, aovsRecordVector, active, radianceSamplesRecordVector);

        // Either there are no aovs samples because the m_integrator does not produce
        // aovs and the aov vector is empty or the integrator produces aovs and
        // there are as many aov samples as radiance samples (one aov sample for
        // each radiance sample)
        assert(aovsRecordVector.empty() ||
               (aovsRecordVector.size() == radianceSamplesRecordVector.size()));

        if(aovsRecordVector.empty()) {
            for (const auto &result : radianceSamplesRecordVector) {
                FloatTimeSample<Float, Mask> color(-1);
                color.set_time(result.time, result.mask);
                if constexpr (is_polarized_v<Spectrum>) {
                    auto const &stokes = result.radiance.coeff(0);
                    for (int i = 3; i >= 0; --i) {
                        Color3f rgb;
                        if constexpr (is_monochromatic_v<Spectrum>) {
                            rgb = stokes[i].x();
                        } else if constexpr (is_rgb_v<Spectrum>) {
                            rgb = stokes[i];
                        } else {
                            static_assert(is_spectral_v<Spectrum>);
                            /// Note: this assumes that sensor used sample_rgb_spectrum() to generate 'ray.wavelengths'
                            auto pdf = pdf_rgb_spectrum(ray.wavelengths);
                            UnpolarizedSpectrum spec =
                                stokes[i] *
                                select(neq(pdf, 0.f), rcp(pdf), 0.f);
                            rgb = xyz_to_srgb(
                                spectrum_to_xyz(spec, ray.wavelengths, active));
                        }

                        // Reversed
                        color.push_front(rgb.b());
                        color.push_front(rgb.g());
                        color.push_front(rgb.r());
                    }
                }
                aovsRecordVector.push_back(color);
            }
        } else {
            for (int i = 0; i < aovsRecordVector.size(); ++i) {
                if constexpr (is_polarized_v<Spectrum>) {
                    auto const &stokes = radianceSamplesRecordVector[i].radiance.coeff(0);
                    for (int i = 3; i >= 0; --i) {
                        Color3f rgb;
                        if constexpr (is_monochromatic_v<Spectrum>) {
                            rgb = stokes[i].x();
                        } else if constexpr (is_rgb_v<Spectrum>) {
                            rgb = stokes[i];
                        } else {
                            static_assert(is_spectral_v<Spectrum>);
                            /// Note: this assumes that sensor used sample_rgb_spectrum() to generate 'ray.wavelengths'
                            auto pdf = pdf_rgb_spectrum(ray.wavelengths);
                            UnpolarizedSpectrum spec =
                                stokes[i] *
                                select(neq(pdf, 0.f), rcp(pdf), 0.f);
                            rgb = xyz_to_srgb(
                                spectrum_to_xyz(spec, ray.wavelengths, active));
                        }

                        // Reversed
                        aovsRecordVector[i].push_front(rgb.b());
                        aovsRecordVector[i].push_front(rgb.g());
                        aovsRecordVector[i].push_front(rgb.r());
                    }
                }
            }
        }
    }

    std::vector<std::string> aov_names() const override {
        std::vector<std::string> result = m_integrator->aov_names();
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 3; ++j)
                result.insert(result.begin() + 3*i + j, "S" + std::to_string(i) + "." + ("RGB"[j]));
        return result;
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_object("integrator", m_integrator.get());
    }

    MTS_DECLARE_CLASS()
private:
    ref<Base> m_integrator;
};

MTS_IMPLEMENT_CLASS_VARIANT(TransientStokesIntegrator, TransientSamplingIntegrator)
MTS_EXPORT_PLUGIN(TransientStokesIntegrator, "Transient Stokes integrator");
NAMESPACE_END(mitsuba)
