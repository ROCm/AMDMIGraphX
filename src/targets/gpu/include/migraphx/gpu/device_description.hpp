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
#ifndef MIGRAPHX_GUARD_GPU_DEVICE_DESCRIPTION_HPP
#define MIGRAPHX_GUARD_GPU_DEVICE_DESCRIPTION_HPP

#include <migraphx/gpu/config.hpp>
#include <cstddef>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

/// The properties of a GPU that are needed to compile for it. It can either be
/// queried from a local device or filled in by the user to compile for a device
/// that is not present.
struct MIGRAPHX_GPU_EXPORT device_description
{
    /// Query the properties of the device with the given hip device id.
    static device_description from_device(std::size_t device);

    /// Resolve the fields that can be derived from the other fields: a
    /// `wavefront_size` of 0 is derived from the `arch`. Counts are clamped to
    /// at least one. Throws when a field is set to an unsupported value.
    void normalize();

    std::string arch                  = {};
    std::size_t num_cu                = 120;
    std::size_t num_chiplets          = 1;
    std::size_t max_threads_per_cu    = 2048;
    std::size_t max_threads_per_block = 1024;
    std::size_t wavefront_size        = 0;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_GPU_DEVICE_DESCRIPTION_HPP
