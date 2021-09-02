#pragma once

#include <deque>

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

template <typename Float, typename Mask>
struct MTS_EXPORT_RENDER FloatTimeSample {
    Float time;
    std::deque<Float> values;
    Mask mask;

    int capacity;
    int filled_from;

public:
    FloatTimeSample(int _capacity) : capacity(_capacity), filled_from(_capacity) {}

    void set_time(Float _time, Mask _mask) { this->time = _time; this->mask = _mask;}

    void push_front(Float _value) {
        // assert(filled_from > 0);
        values.push_front(_value);
    }
};

NAMESPACE_END(mitsuba)