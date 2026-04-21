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
#ifndef MIGRAPHX_GUARD_RTGLIB_KERNEL_HPP
#define MIGRAPHX_GUARD_RTGLIB_KERNEL_HPP

#include <migraphx/gpu/config.hpp>
#include <migraphx/gpu/pack_args.hpp>
#include <migraphx/pmr/vector.hpp>
#include <hip/hip_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct kernel_impl;

struct MIGRAPHX_GPU_EXPORT kernel
{
    struct pointers
    {
        pointers() {}

        pointers(void** pp, std::size_t pn) : p(pp), n(pn) {}

        pointers(std::vector<void*>& v) : p(v.data()), n(v.size()) {}
        pointers(pmr::vector<void*>& v) : p(v.data()), n(v.size()) {}

        void** data() const { return p; }

        std::size_t size() const { return n; }

        std::size_t bytes() const { return n * sizeof(void*); }

        private:
        void** p      = nullptr;
        std::size_t n = 0;
    };
    kernel() = default;
    kernel(const char* image, const std::string& name);
    template <class T, MIGRAPHX_REQUIRES(sizeof(T) == 1)>
    kernel(const std::vector<T>& image, const std::string& name)
        : kernel(reinterpret_cast<const char*>(image.data()), name)
    {
    }

    bool empty() const;

    void launch(hipStream_t stream,
                std::size_t global,
                std::size_t global_y,
                std::size_t global_z,
                std::size_t local,
                std::size_t local_y,
                std::size_t local_z,
                const std::vector<kernel_argument>& args,
                hipEvent_t start = nullptr,
                hipEvent_t stop  = nullptr) const;

    void launch(hipStream_t stream,
                std::size_t global,
                std::size_t global_y,
                std::size_t global_z,
                std::size_t local,
                std::size_t local_y,
                std::size_t local_z,
                pointers args,
                hipEvent_t start = nullptr,
                hipEvent_t stop  = nullptr) const;

    void launch(hipStream_t stream,
                std::size_t global,
                std::size_t local,
                const std::vector<kernel_argument>& args,
                hipEvent_t start = nullptr,
                hipEvent_t stop  = nullptr) const
    {
        launch(stream, global, 1, 1, local, 1, 1, args, start, stop);
    }

    void launch(hipStream_t stream,
                std::size_t global,
                std::size_t local,
                pointers args,
                hipEvent_t start = nullptr,
                hipEvent_t stop  = nullptr) const
    {
        launch(stream, global, 1, 1, local, 1, 1, args, start, stop);
    }

    template <class... Ts, MIGRAPHX_REQUIRES(std::is_convertible<Ts, hipEvent_t>{}...)>
    auto launch(hipStream_t stream, std::size_t global, std::size_t local, Ts... zs) const
    {
        return [=](auto&&... xs) {
            launch(stream, global, local, std::vector<kernel_argument>{xs...}, zs...);
        };
    }

    private:
    std::shared_ptr<kernel_impl> impl;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
