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
#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_SYMBOLIC_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_SYMBOLIC_HPP

#include <cstddef>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <vector>

#include <symengine/expression.h>
#include <symengine/integer.h>
#include <symengine/symbol.h>

#include <migraphx/config.hpp>
#include <migraphx/shape.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct value;

struct MIGRAPHX_EXPORT symbolic_dim
{
    SymEngine::Expression expr = SymEngine::Expression(0);
    std::size_t min            = 0;
    std::size_t max            = 0;
    std::set<std::size_t> optimals{};

    symbolic_dim() = default;

    symbolic_dim(std::size_t value);

    symbolic_dim(const std::string& name, std::size_t min_val, std::size_t max_val,
                 std::set<std::size_t> opt = {});

    symbolic_dim(SymEngine::Expression e, std::size_t min_val, std::size_t max_val,
                 std::set<std::size_t> opt = {});

    explicit symbolic_dim(const shape::dynamic_dimension& dd);

    bool is_fixed() const;
    std::string to_string() const;

    shape::dynamic_dimension to_dynamic_dimension() const;

    std::optional<symbolic_dim> intersection(const symbolic_dim& other) const;

    symbolic_dim& operator+=(const symbolic_dim& x);
    symbolic_dim& operator-=(const symbolic_dim& x);
    symbolic_dim& operator*=(const symbolic_dim& x);
    MIGRAPHX_EXPORT friend symbolic_dim operator+(const symbolic_dim& x, const symbolic_dim& y);
    MIGRAPHX_EXPORT friend symbolic_dim operator-(const symbolic_dim& x, const symbolic_dim& y);
    MIGRAPHX_EXPORT friend symbolic_dim operator*(const symbolic_dim& x, const symbolic_dim& y);

    symbolic_dim& operator+=(const std::size_t& x);
    symbolic_dim& operator-=(const std::size_t& x);
    symbolic_dim& operator*=(const std::size_t& x);
    MIGRAPHX_EXPORT friend symbolic_dim operator+(const symbolic_dim& x, const std::size_t& y);
    MIGRAPHX_EXPORT friend symbolic_dim operator+(const std::size_t& x, const symbolic_dim& y);
    MIGRAPHX_EXPORT friend symbolic_dim operator-(const symbolic_dim& x, const std::size_t& y);
    MIGRAPHX_EXPORT friend symbolic_dim operator*(const symbolic_dim& x, const std::size_t& y);
    MIGRAPHX_EXPORT friend symbolic_dim operator*(const std::size_t& x, const symbolic_dim& y);

    MIGRAPHX_EXPORT friend bool operator==(const symbolic_dim& x, const symbolic_dim& y);
    MIGRAPHX_EXPORT friend bool operator!=(const symbolic_dim& x, const symbolic_dim& y);
    MIGRAPHX_EXPORT friend std::ostream& operator<<(std::ostream& os, const symbolic_dim& x);
};

MIGRAPHX_EXPORT void migraphx_to_value(value& v, const symbolic_dim& sd);
MIGRAPHX_EXPORT void migraphx_from_value(const value& v, symbolic_dim& sd);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
