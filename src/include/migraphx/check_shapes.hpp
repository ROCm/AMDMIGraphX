/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_RTGLIB_CHECK_SHAPES_HPP
#define MIGRAPHX_GUARD_RTGLIB_CHECK_SHAPES_HPP

#include <migraphx/shape.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/config.hpp>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct check_shapes
{
    const shape* begin;
    const shape* end;
    const std::string name;

    check_shapes(const shape* b, const shape* e, const std::string& n) : begin(b), end(e), name(n)
    {
    }

    template <class Op>
    check_shapes(const shape* b, const shape* e, const Op& op) : begin(b), end(e), name(op.name())
    {
    }

    template <class Op>
    check_shapes(const std::vector<shape>& s, const Op& op)
        : begin(s.data()), end(s.data() + s.size()), name(op.name())
    {
    }

    std::string prefix() const
    {
        if(name.empty())
            return "";
        else
            return name + ": ";
    }

    std::size_t size() const
    {
        if(begin == end)
            return 0;
        assert(begin != nullptr);
        assert(end != nullptr);
        return end - begin;
    }

    /*!
     * Check if the number of shape objects is equal to atleast one of the
     * given sizes.
     * \param ns template parameter pack of sizes to check against
     */
    template <class... Ts>
    const check_shapes& has(Ts... ns) const
    {
        if(migraphx::none_of({ns...}, [&](auto i) { return this->size() == i; }))
            MIGRAPHX_THROW(prefix() + "Wrong number of arguments: expected " +
                           to_string_range({ns...}) + " but given " + std::to_string(size()));
        return *this;
    }

    const check_shapes& nelements(std::size_t n) const
    {
        if(!this->all_of([&](const shape& s) { return s.elements() == n; }))
            MIGRAPHX_THROW(prefix() + "Shapes must have only " + std::to_string(n) + " elements");
        return *this;
    }

    const check_shapes& only_dims(std::size_t n) const
    {
        assert(begin != nullptr);
        assert(end != nullptr);
        if(begin != end)
        {
            if(begin->lens().size() != n)
                MIGRAPHX_THROW(prefix() + "Only " + std::to_string(n) + "d supported");
        }
        return *this;
    }

    const check_shapes& max_ndims(std::size_t n) const
    {
        assert(begin != nullptr);
        assert(end != nullptr);
        if(begin != end)
        {
            if(begin->lens().size() > n)
                MIGRAPHX_THROW(prefix() + "Shape must have at most " + std::to_string(n) +
                               " dimensions");
        }
        return *this;
    }

    const check_shapes& min_ndims(std::size_t n) const
    {
        assert(begin != nullptr);
        assert(end != nullptr);
        if(begin != end)
        {
            if(begin->lens().size() < n)
                MIGRAPHX_THROW(prefix() + "Shape must have at least " + std::to_string(n) +
                               " dimensions");
        }
        return *this;
    }

    const check_shapes& same_shape() const
    {
        if(!this->same([](const shape& s) { return s; }))
            MIGRAPHX_THROW(prefix() + "Shapes do not match");
        return *this;
    }

    const check_shapes& same_type() const
    {
        if(!this->same([](const shape& s) { return s.type(); }))
            MIGRAPHX_THROW(prefix() + "Types do not match");
        return *this;
    }

    const check_shapes& same_dims() const
    {
        if(!this->same([](const shape& s) { return s.lens(); }))
            MIGRAPHX_THROW(prefix() + "Dimensions do not match");
        return *this;
    }

    const check_shapes& same_ndims() const
    {
        if(!this->same([](const shape& s) { return s.lens().size(); }))
            MIGRAPHX_THROW(prefix() + "Number of dimensions do not match");
        return *this;
    }

    const check_shapes& standard() const
    {
        if(!this->all_of([](const shape& s) { return s.standard(); }))
            MIGRAPHX_THROW(prefix() + "Shapes are not in standard layout");
        return *this;
    }

    const check_shapes& standard_or_scalar() const
    {
        if(!this->all_of([](const shape& s) { return s.standard() or s.scalar(); }))
            MIGRAPHX_THROW(prefix() + "Shapes are not a scalar or in standard layout");
        return *this;
    }

    const check_shapes& packed() const
    {
        if(!this->all_of([](const shape& s) { return s.packed(); }))
            MIGRAPHX_THROW(prefix() + "Shapes are not packed");
        return *this;
    }

    const check_shapes& packed_or_broadcasted() const
    {
        if(!this->all_of([](const shape& s) { return s.packed() or s.broadcasted(); }))
            MIGRAPHX_THROW(prefix() + "Shapes are not packed nor broadcasted");
        return *this;
    }

    const check_shapes& tuple_type() const
    {
        if(!this->all_of([](const shape& s) { return s.type() == shape::tuple_type; }))
            MIGRAPHX_THROW(prefix() + "Shapes are not tuple!");
        return *this;
    }

    const check_shapes& not_transposed() const
    {
        if(!this->all_of([](const shape& s) { return not s.transposed(); }))
            MIGRAPHX_THROW(prefix() + "Shapes are transposed");
        return *this;
    }

    const check_shapes& not_broadcasted() const
    {
        if(!this->all_of([](const shape& s) { return not s.broadcasted(); }))
            MIGRAPHX_THROW(prefix() + "Shapes are broadcasted");
        return *this;
    }

    const check_shapes& elements(std::size_t n) const
    {
        if(!this->all_of([&](const shape& s) { return s.elements() == n; }))
            MIGRAPHX_THROW(prefix() + "Wrong number of elements");
        return *this;
    }

    const check_shapes& batch_not_transposed() const
    {
        if(!this->all_of([&](const shape& s) { return batch_not_transposed_strides(s.strides()); }))
            MIGRAPHX_THROW(prefix() + "Batch size is transposed");
        return *this;
    }

    template <class F>
    bool same(F f) const
    {
        if(begin == end)
            return true;
        assert(begin != nullptr);
        assert(end != nullptr);
        auto&& key = f(*begin);
        return this->all_of([&](const shape& s) { return f(s) == key; });
    }

    template <class Predicate>
    bool all_of(Predicate p) const
    {
        if(begin == end)
            return true;
        assert(begin != nullptr);
        assert(end != nullptr);
        return std::all_of(begin, end, p);
    }

    const shape* get(long i) const
    {
        if(i >= size())
            MIGRAPHX_THROW(prefix() + "Accessing shape out of bounds");
        assert(begin != nullptr);
        assert(end != nullptr);
        if(i < 0)
            return end - i;
        return begin + i;
    }

    check_shapes slice(long start) const { return {get(start), end, name}; }

    check_shapes slice(long start, long last) const { return {get(start), get(last), name}; }

    private:
    static bool batch_not_transposed_strides(const std::vector<std::size_t>& strides)
    {
        if(strides.size() <= 2)
            return true;
        auto dim_0       = strides.size() - 2;
        auto matrix_size = std::max(strides[dim_0], strides[dim_0 + 1]);
        std::vector<std::size_t> batch(strides.begin(), strides.begin() + dim_0);
        if(std::all_of(batch.begin(), batch.end(), [&](auto i) { return (i < matrix_size); }))
        {
            return false;
        }

        if(std::adjacent_find(batch.begin(), batch.end(), [&](auto i, auto j) {
               return (i < j or i < matrix_size or j < matrix_size);
           }) != batch.end())
        {
            return false;
        }
        return true;
    }
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
