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
#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_LITERAL_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_LITERAL_HPP

#include <migraphx/shape.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/tensor_view.hpp>
#include <migraphx/raw_data.hpp>
#include <migraphx/make_shared_array.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/config.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct external_data;
struct external_data_source;
struct mapped_file;

/**
 * @brief Represents a raw literal
 * @details This stores the literal has a raw buffer that is owned by this class
 */
struct literal : raw_data<literal>
{
    literal() {}

    /*!
     * Empty literal with a specific shape type
     */
    explicit literal(shape::type_t shape_type) : m_shape(shape_type, {}) {}

    template <class U, class T = deduce<U>, shape::type_t ShapeType = shape::get_type<T>{}>
    literal(U x) : m_shape(ShapeType)
    {
        static_assert(std::is_trivially_copyable<T>{}, "Literals can only be trivial types");
        auto b                           = make_shared_array<char>(sizeof(T));
        *(reinterpret_cast<T*>(b.get())) = x;
        buffer                           = std::move(b);
    }

    template <class T>
    literal(const shape& s, const std::vector<T>& x) : m_shape(s)
    {
        static_assert(std::is_trivially_copyable<T>{}, "Literals can only be trivial types");
        fill(x.begin(), x.end());
    }

    template <class T>
    literal(const shape& s, const std::initializer_list<T>& x) : m_shape(s)
    {
        static_assert(std::is_trivially_copyable<T>{}, "Literals can only be trivial types");
        fill(x.begin(), x.end());
    }

    template <class Iterator>
    literal(const shape& s, Iterator start, Iterator end) : m_shape(s)
    {
        fill(start, end);
    }

    // x holds s.bytes(): raw storage in the layout described by s's strides
    template <class T, MIGRAPHX_REQUIRES(sizeof(T) == 1)>
    static literal from_raw_buffer(const shape& s, T* x)
    {
        literal result;
        result.m_shape = s;
        result.buffer  = make_shared_array<char>(x, x + s.bytes());
        return result;
    }

    // x holds nbytes: the elements of s in row-major order of s.lens(), regardless of s's strides
    template <class T, MIGRAPHX_REQUIRES(sizeof(T) == 1)>
    static literal from_standard_buffer(const shape& s, T* x, std::size_t nbytes)
    {
        if(nbytes != s.elements() * s.type_size())
            MIGRAPHX_THROW("literal: buffer size " + std::to_string(nbytes) +
                           " does not hold shape elements " + std::to_string(s.elements()) +
                           " of type size " + std::to_string(s.type_size()));
        literal result;
        result.m_shape = s;
        if(s.standard())
        {
            result.buffer = make_shared_array<char>(x, x + nbytes);
            return result;
        }
        // x may not be aligned for the element type, so copy it into typed storage first
        s.visit_type([&](auto as) {
            std::vector<typename decltype(as)::type> values(s.elements());
            std::copy(x, x + nbytes, reinterpret_cast<char*>(values.data()));
            result.fill(values.begin(), values.end());
        });
        return result;
    }

    /// The literal whose data is ext.size bytes at ext.offset of file, in the layout described by
    /// s's strides. The data aliases the mapping when the offset is aligned for the element type,
    /// otherwise it is copied; either way the literal keeps the reference and the mapping.
    MIGRAPHX_EXPORT static literal
    from_mapped_file(const shape& s, std::shared_ptr<mapped_file> file, external_data ext);

    /// Whether data is available
    bool empty() const { return this->buffer == nullptr; }

    /// Provides a raw pointer to the data
    const char* data() const { return this->buffer.get(); }

    const shape& get_shape() const { return this->m_shape; }

    /// Whether the data was read from an external file
    bool is_external() const { return this->m_source != nullptr; }

    /// Where the data is stored outside the program file, or nullptr when it is inline
    MIGRAPHX_EXPORT const external_data* get_external_data() const;

    /// The mapping the external data was read from, or nullptr when it is inline
    MIGRAPHX_EXPORT std::shared_ptr<mapped_file> get_mapped_file() const;

    std::vector<literal> get_sub_objects() const { return {}; }

    /// Convert the data to an argument
    argument get_argument() const
    {
        auto b = make_shared_array<char>(buffer.get(), buffer.get() + m_shape.bytes());
        return {m_shape, [b]() { return b.get(); }};
    }

    /// An argument aliasing the data without copying it. The data may be a read-only mapping, so
    /// the argument must only be read.
    MIGRAPHX_EXPORT argument share_argument() const;

    private:
    std::shared_ptr<const char> buffer;
    shape m_shape;
    std::shared_ptr<const external_data_source> m_source = nullptr;

    // Keeps the same data ordering as the given container. Fewer values than the shape's
    // elements fills only the leading elements; the rest remain zero.
    template <class Iterator>
    void fill(Iterator start, Iterator end)
    {
        if(m_shape.elements() < std::distance(start, end))
            MIGRAPHX_THROW("literal: number of values " +
                           std::to_string(std::distance(start, end)) + " exceeds shape elements " +
                           std::to_string(m_shape.elements()));
        auto b = make_shared_array<char>(m_shape.bytes());
        m_shape.visit_type([&](auto as) {
            auto output = make_view(m_shape, as.from(b.get()));
            std::copy(start, end, output.begin());
        });
        buffer = std::move(b);
    }
};

template <class F>
literal transform(literal l, F f)
{
    literal result;
    l.visit([&](auto x) {
        using type = std::remove_cv_t<typename decltype(x)::value_type>;
        std::vector<type> output(x.size(), type(0));
        std::transform(x.begin(), x.end(), output.begin(), f);
        result = literal{l.get_shape(), output};
    });
    return result;
}

template <class F>
literal transform(literal l1, literal l2, F f)
{
    assert(l1.get_shape() == l2.get_shape());
    literal result;
    visit_all(l1, l2)([&](auto x, auto y) {
        using type = std::remove_cv_t<typename decltype(x)::value_type>;
        std::vector<type> output(x.size(), type(0));
        std::transform(x.begin(), x.end(), y.begin(), output.begin(), f);
        result = literal{l1.get_shape(), output};
    });
    return result;
}

MIGRAPHX_EXPORT void migraphx_to_value(value& v, const literal& l);
MIGRAPHX_EXPORT void migraphx_from_value(const value& v, literal& l);
MIGRAPHX_EXPORT void migraphx_to_value(value& v, const literal& l, const serialize_context& sctx);
MIGRAPHX_EXPORT void migraphx_from_value(const value& v, literal& l, const serialize_context& sctx);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
