#pragma once

#include <mitsuba/core/fwd.h>
#include <mitsuba/render/fwd.h>

NAMESPACE_BEGIN(mitsuba)

// TODO: see if this should have an interface to be called from Python
template <typename Float, typename Value, typename Mask>
struct MTS_EXPORT_RENDER RadianceSample {
    Float time;
    Value radiance;
    Mask mask;

    RadianceSample(Float time, Value radiance, Mask mask)
        : time(time), radiance(radiance), mask(mask) {}
};

NAMESPACE_END(mitsuba)