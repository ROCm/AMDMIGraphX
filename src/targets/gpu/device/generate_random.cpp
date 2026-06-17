/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <migraphx/gpu/device/generate_random.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/visit.hpp>
#include <migraphx/generate.hpp>
#include <cstdint>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

static constexpr unsigned long splitmix64(unsigned long z)
{
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
}

// Golden-ratio counter step: neighbouring indices land far apart.
static constexpr unsigned long golden_ratio_step = 0x9E3779B97F4A7C15ULL;

void generate_random(hipStream_t stream, const argument& result, unsigned long seed)
{
    const shape& s = result.get_shape();
    if(s.element_space() == 0)
        return;

    if(s.computable())
    {
        visit_all(result)([&](auto out) {
            using type   = typename decltype(out)::value_type;
            auto* output = device_cast(out.data());
            gs_launch(stream, s.element_space())([=](unsigned long i) __device__ {
                auto z    = splitmix64(seed + i * golden_ratio_step);
                output[i] = normalize<type>(z, random_mode::random);
            });
        });
    }
    else
    {
        // Non-computable types (e.g. fp4x2) have no visitor: fill raw bytes.
        auto* output = reinterpret_cast<uint8_t*>(result.data());
        gs_launch(stream, s.bytes())([=](unsigned long i) __device__ {
            auto z    = splitmix64(seed + i * golden_ratio_step);
            output[i] = normalize<uint8_t>(z, random_mode::random);
        });
    }
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
