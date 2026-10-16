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
#ifndef MIGRAPHX_GUARD_RTGLIB_PACK_ARGS_HPP
#define MIGRAPHX_GUARD_RTGLIB_PACK_ARGS_HPP

#include <migraphx/gpu/config.hpp>
#include <migraphx/bit_cast.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/requires.hpp>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <map>
#include <type_traits>
#include <utility>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// Padding inserted before an argument at buffer position `pos` to reach
// `align`; the single definition of the kernarg alignment rule.
constexpr std::size_t pack_padding(std::size_t pos, std::size_t align)
{
    assert(align > 0);
    return (align - (pos % align)) % align;
}

struct kernel_argument
{
    template <class T,
              class U = std::remove_reference_t<T>,
              MIGRAPHX_REQUIRES(not std::is_base_of<kernel_argument, T>{})>
    kernel_argument(T&& x) : size(sizeof(U)), align(alignof(U)), data(&x) // NOLINT
    {
    }
    std::size_t size;
    std::size_t align;
    void* data;
};

struct kernel_argument_value
{
    kernel_argument_value() = default;

    template <class T,
              class U = std::remove_reference_t<T>,
              MIGRAPHX_REQUIRES(not std::is_base_of<kernel_argument_value, U>{} and
                                std::is_trivially_copyable<U>{})>
    kernel_argument_value(T&& x) : align(alignof(U)), data(sizeof(U))
    {
        auto as_bytes = migraphx::bit_cast<std::array<char, sizeof(U)>>(x);
        std::copy(as_bytes.begin(), as_bytes.end(), data.begin());
    }

    std::size_t align = 0;
    std::vector<char> data{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.align, "align"), f(self.data, "data"));
    }

    friend bool operator==(const kernel_argument_value& a, const kernel_argument_value& b)
    {
        return a.align == b.align and a.data == b.data;
    }
    friend bool operator!=(const kernel_argument_value& a, const kernel_argument_value& b)
    {
        return not(a == b);
    }
};

MIGRAPHX_GPU_EXPORT std::vector<char> pack_args(const std::vector<kernel_argument>& args);
MIGRAPHX_GPU_EXPORT std::vector<char> pack_args(const std::vector<kernel_argument_value>& args);

// Walk the packed-buffer layout of `kernel_args` in index order, calling
// f(offset, is_pointer) for each argument; the single definition of how a
// code_object_op's kernel_args map onto packed bytes (a pointer slot has empty
// data and takes 8 bytes at align 8, a scalar its own bytes and alignment).
template <class F>
void for_each_kernarg_slot(const std::map<std::size_t, kernel_argument_value>& kernel_args, F f)
{
    std::size_t pos = 0;
    for(const auto& [idx, v] : kernel_args)
    {
        bool is_pointer = v.data.empty();
        pos += pack_padding(pos, is_pointer ? sizeof(char*) : v.align);
        f(pos, is_pointer);
        pos += is_pointer ? sizeof(char*) : v.data.size();
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
