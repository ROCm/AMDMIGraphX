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
#include <migraphx/gpu/cross_compile_device.hpp>
#include <migraphx/gpu/device_name.hpp>
#include <migraphx/stringutils.hpp>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// RDNA architectures (gfx10xx and newer) natively execute in wave32 mode, while
// GCN/CDNA architectures (gfx9 and older) use wave64. The wavefront size selects
// device-specific code paths (e.g. DPP row_bcast cross-lane ops, which only exist
// on wave64 GCN/CDNA and are illegal on GFX10+), so cross-compilation must report
// the same size a physical device would.
static int arch_wavefront_size(const std::string& arch_name)
{
    const auto gfx = get_gfx_name(arch_name);
    if(starts_with(gfx, "gfx10") or starts_with(gfx, "gfx11") or starts_with(gfx, "gfx12"))
        return 32;
    return 64;
}

hipDeviceProp_t make_cross_compile_device_props(const std::string& arch_name, std::size_t cu_count)
{
    hipDeviceProp_t props{};
    auto n = std::min(arch_name.size(), sizeof(props.gcnArchName) - 1);
    std::copy_n(arch_name.begin(), n, props.gcnArchName);
    props.gcnArchName[n] = '\0';
    props.warpSize       = arch_wavefront_size(arch_name);
    // these are placeholders
    props.maxThreadsPerMultiProcessor = 2048;
    props.maxThreadsPerBlock          = 1024;
    props.multiProcessorCount         = cu_count;
    return props;
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
