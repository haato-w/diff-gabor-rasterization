/*
 * Warp-local gradient aggregation following DISTWAR, Sections IV-B and V-A2.
 * All 32 lanes must visit the same primitive. Rejected pixels contribute zero;
 * a sparse participation mask is not sufficient for the shuffle tree itself.
 */
#pragma once

static_assert(BLOCK_SIZE % 32 == 0, "Warp reduction requires complete hardware warps");

template <bool Reduction>
__device__ __forceinline__ void record_pixel_gradient(
    float* destination, float& local, const float value, const bool aggregate)
{
    if constexpr (Reduction)
    {
        if (aggregate) local += value;
        else atomicAdd(destination, value);
    }
    else atomicAdd(destination, value);
}

template <int Components>
__device__ __forceinline__ void butterfly_atomic_add(
    float* destination, const float* values, const unsigned int lane)
{
#pragma unroll
    for (int component = 0; component < Components; ++component)
    {
        float sum = values[component];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0 && sum != 0.0f)
            atomicAdd(destination + component, sum);
    }
}

// Compile-time sizes keep all gradient components in registers. These buffers
// are per-thread values, not the global checkpoints used by Gaussian backward.
template <int Channels, bool HasSpecular = false, bool HasAttributes = false>
struct PixelGradients
{
    static constexpr int Color1 = 0;
    static constexpr int Color2 = Color1 + Channels;
    static constexpr int Specular = Color2 + Channels;
    static constexpr int Attributes = Specular + (HasSpecular ? Channels : 0);
    static constexpr int Frequency = Attributes + (HasAttributes ? 6 : 0);
    static constexpr int Phases = Frequency + 2;
    static constexpr int Transform = Phases + 1;
    static constexpr int Mean = Transform + 9;
    static constexpr int Normal = Mean + 2;
    static constexpr int Opacity = Normal + 3;
    float values[Opacity + 1] = {};

    __device__ __forceinline__ void flush(
        const unsigned int primitive, const unsigned int lane,
        float* color1, float* color2, float* frequency_vectors,
        float* phases, float* transform, float3* mean, float* normal,
        float* opacity, float* specular, float* attributes) const
    {
        butterfly_atomic_add<Channels>(color1 + primitive * Channels, values + Color1, lane);
        butterfly_atomic_add<Channels>(color2 + primitive * Channels, values + Color2, lane);
        if constexpr (HasSpecular)
            butterfly_atomic_add<Channels>(specular + primitive * Channels, values + Specular, lane);
        if constexpr (HasAttributes)
            butterfly_atomic_add<6>(attributes + primitive * 6, values + Attributes, lane);
        butterfly_atomic_add<2>(frequency_vectors + primitive * 2, values + Frequency, lane);
        butterfly_atomic_add<1>(phases + primitive, values + Phases, lane);
        butterfly_atomic_add<9>(transform + primitive * 9, values + Transform, lane);
        butterfly_atomic_add<2>(reinterpret_cast<float*>(mean + primitive), values + Mean, lane);
        butterfly_atomic_add<3>(normal + primitive * 3, values + Normal, lane);
        butterfly_atomic_add<1>(opacity + primitive, values + Opacity, lane);
    }
};
