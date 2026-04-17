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
#include <cstring>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

hipDeviceProp_t
make_cross_compile_device_props(const std::string& arch_name, std::size_t cu_count)
{
    hipDeviceProp_t props{};
    std::strncpy(props.gcnArchName, arch_name.c_str(), sizeof(props.gcnArchName) - 1);
    props.gcnArchName[sizeof(props.gcnArchName) - 1] = '\0';
    // these are placeholders
    props.warpSize                    = 64;
    props.maxThreadsPerMultiProcessor = 2048;
    props.maxThreadsPerBlock          = 1024;
    props.multiProcessorCount         = cu_count;
    return props;
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
