
#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_SHAPE_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_SHAPE_HPP

#include <migraphx/gpu/device/array.hpp>
#include <migraphx/gpu/device/fast_div.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

template <std::size_t N>
struct hip_shape
{
    using hip_index                   = hip_array<std::size_t, N>;
    hip_array<std::size_t, N> lens    = {};
    hip_array<std::size_t, N> strides = {};
    hip_array<std::size_t, N> divs    = {};
    bool standard                     = false;

    __device__ __host__ hip_shape() = default;

    hip_shape(const shape& s) : standard(s.standard())
    {
        assert(s.lens().size() == N);
        assert(s.strides().size() == N);
        std::copy(s.lens().begin(), s.lens().end(), lens.begin());
        std::copy(s.strides().begin(), s.strides().end(), strides.begin());
        std::transform(s.strides().begin(), s.strides().end(), divs.begin(), &encode_divisor);
    }

    MIGRAPHX_DEVICE_CONSTEXPR std::size_t elements() const { return lens.product(); }

    MIGRAPHX_DEVICE_CONSTEXPR std::size_t index(hip_index x) const { return x.dot(strides); }

    MIGRAPHX_DEVICE_CONSTEXPR std::size_t index(std::initializer_list<std::size_t> x) const
    {
        std::size_t idx = 0;
        for(std::size_t i = 0; i < x.size(); i++)
            idx += *(x.begin() + i) * strides[i];
        return idx;
    }

    MIGRAPHX_DEVICE_CONSTEXPR std::size_t index(std::size_t i) const
    {
        if(this->standard)
            return i;
        else
        {
            const std::size_t rank = this->lens.size();
            std::size_t s          = 1;
            std::size_t result     = 0;
            for(std::size_t j = 0; j < this->lens.size(); j++)
            {
                const std::size_t k      = rank - j - 1;
                const std::size_t stride = this->strides[k];
                const std::size_t len    = this->lens[k];
                const std::size_t slen   = s * len;
                const std::size_t idx    = (i % slen) / s;
                result += stride * idx;
                s = slen;
            }
            return result;
        }
    }

    MIGRAPHX_DEVICE_CONSTEXPR hip_index multi(std::size_t idx) const
    {
        hip_index result;
        std::size_t tidx = idx;
        for(std::size_t is = 0; is < result.size(); is++)
        {
            auto d     = fast_div(tidx, divs[is]);
            result[is] = d;
            tidx       = remainder(d, tidx, strides[is]);
        }
        return result;
    }

    MIGRAPHX_DEVICE_CONSTEXPR hip_index carry(hip_index result) const
    {
        std::ptrdiff_t rem = 0;
        for(std::ptrdiff_t i = result.size() - 1; i >= 0; i--)
        {
            auto z = result[i] + rem;
            rem    = z - std::ptrdiff_t(lens[i]) + 1;
            if(rem > 0)
                z -= rem;
            else
                rem = 0;
            result[i] = z;
        }
        return result;
    }
};

template <std::size_t N>
hip_shape<N> make_hip_shape(const shape& x)
{
    return x;
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
