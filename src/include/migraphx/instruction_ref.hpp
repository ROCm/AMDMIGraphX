/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_INSTRUCTION_REF_HPP
#define MIGRAPHX_GUARD_INSTRUCTION_REF_HPP

#include <list>
#include <functional>
#include <migraphx/config.hpp>
#include <migraphx/requires.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct instruction;

MIGRAPHX_EXPORT migraphx::instruction*
as_address(const std::list<instruction>::iterator& ins) noexcept;
MIGRAPHX_EXPORT const migraphx::instruction*
as_address(const std::list<instruction>::const_iterator& ins) noexcept;

#if defined(CPPCHECK)
using instruction_ref = std::list<instruction>::iterator;
#else
struct instruction_ref : std::list<instruction>::iterator
{
    using instruction_iter       = std::list<instruction>::iterator;
    using instruction_const_iter = std::list<instruction>::const_iterator;

    instruction_ref() = default;
    instruction_ref(const instruction_iter& other) : instruction_iter(other) {}

    template <class T,
              class U,
              MIGRAPHX_REQUIRES(std::is_same<T, instruction_ref>{} or
                                std::is_same<U, instruction_ref>{})>
    friend auto operator==(const T& x, const U& y) -> decltype(bool(as_address(x) == as_address(y)))
    {
        return as_address(x) == as_address(y);
    }

    template <class T,
              class U,
              MIGRAPHX_REQUIRES(std::is_same<T, instruction_ref>{} or
                                std::is_same<U, instruction_ref>{})>
    friend auto operator!=(const T& x, const U& y) -> decltype(bool(as_address(x) != as_address(y)))
    {
        return as_address(x) != as_address(y);
    }
};
#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

namespace std {
template <>
struct hash<migraphx::instruction_ref>
{
    using argument_type = migraphx::instruction_ref;
    using result_type   = std::size_t;
    result_type operator()(const migraphx::instruction_ref& x) const noexcept
    {
        return std::hash<migraphx::instruction*>{}(migraphx::as_address(x));
    }
};

} // namespace std

#ifdef _MSC_VER
#include <migraphx/instruction.hpp>
#endif

#endif
