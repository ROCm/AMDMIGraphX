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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_SPAN_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_SPAN_HPP

#include <migraphx/byte.hpp>
#include <migraphx/config.hpp>
#include <migraphx/requires.hpp>
#include <array>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

inline constexpr std::size_t dynamic_extent = std::numeric_limits<std::size_t>::max();

template <class T, std::size_t Extent = dynamic_extent>
struct span;

namespace detail {

template <class T>
struct is_span : std::false_type
{
};

template <class T, std::size_t N>
struct is_span<span<T, N>> : std::true_type
{
};

template <class T>
struct is_std_array : std::false_type
{
};

template <class T, std::size_t N>
struct is_std_array<std::array<T, N>> : std::true_type
{
};

/// True when U* converts to T* by adding qualifiers only, never derived-to-base
template <class U, class T>
using is_span_convertible = std::is_convertible<U (*)[], T (*)[]>;

/// Pointers are the only contiguous iterators C++17 can name
template <class Iterator, class T>
struct is_span_iterator : std::false_type
{
};

template <class U, class T>
struct is_span_iterator<U*, T> : is_span_convertible<U, T>
{
};

template <class Range>
using range_pointer_t = decltype(std::data(std::declval<Range&>()));

template <class Range, class T, class = void>
struct is_span_range : std::false_type
{
};

/// Contiguous ranges expose data() and size(); arrays and spans have their own constructors
template <class Range, class T>
struct is_span_range<
    Range,
    T,
    std::void_t<range_pointer_t<Range>, decltype(std::size(std::declval<Range&>()))>>
    : std::integral_constant<bool,
                             (not is_span<std::remove_cv_t<Range>>{} and
                              not is_std_array<std::remove_cv_t<Range>>{} and
                              not std::is_array<Range>{} and
                              is_span_iterator<range_pointer_t<Range>, T>{})>
{
};

/// A span of a temporary range may only be formed when it cannot be written through
template <class Range, class T>
using is_span_borrowable =
    std::integral_constant<bool, (std::is_lvalue_reference<Range>{} or std::is_const<T>{})>;

template <std::size_t Extent, std::size_t Offset, std::size_t Count>
constexpr std::size_t subspan_extent()
{
    if(Count != dynamic_extent)
        return Count;
    if(Extent != dynamic_extent)
        return Extent - Offset;
    return dynamic_extent;
}

template <class T, std::size_t Extent>
struct span_storage
{
    constexpr span_storage(T* p, std::size_t) noexcept : ptr(p) {}
    constexpr std::size_t size() const noexcept { return Extent; }
    T* ptr;
};

template <class T>
struct span_storage<T, dynamic_extent>
{
    constexpr span_storage(T* p, std::size_t n) noexcept : ptr(p), len(n) {}
    constexpr std::size_t size() const noexcept { return len; }
    T* ptr;
    std::size_t len;
};

} // namespace detail

/// A non-owning view over a contiguous sequence, following std::span. The extent is
/// either fixed at compile time or dynamic_extent, in which case the size is stored.
template <class T, std::size_t Extent>
struct span
{
    static_assert(std::is_object<T>{} and not std::is_abstract<T>{},
                  "span requires a complete, non-abstract element type");

    using element_type           = T;
    using value_type             = std::remove_cv_t<T>;
    using size_type              = std::size_t;
    using difference_type        = std::ptrdiff_t;
    using pointer                = T*;
    using const_pointer          = const T*;
    using reference              = T&;
    using const_reference        = const T&;
    using iterator               = T*;
    using const_iterator         = const T*;
    using reverse_iterator       = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    static constexpr size_type extent = Extent;

    template <MIGRAPHX_REQUIRES(Extent == 0 or Extent == dynamic_extent)>
    constexpr span() noexcept : storage(nullptr, 0)
    {
    }

    template <class Iterator,
              MIGRAPHX_REQUIRES(Extent == dynamic_extent and
                                detail::is_span_iterator<Iterator, T>{})>
    constexpr span(Iterator first, size_type count) : storage(first, count)
    {
    }

    template <class Iterator,
              MIGRAPHX_REQUIRES(Extent != dynamic_extent and
                                detail::is_span_iterator<Iterator, T>{})>
    constexpr explicit span(Iterator first, size_type count) : storage(first, count)
    {
        assert(count == Extent);
    }

    template <class Iterator,
              MIGRAPHX_REQUIRES(Extent == dynamic_extent and
                                detail::is_span_iterator<Iterator, T>{})>
    constexpr span(Iterator first, Iterator last)
        : span(first, static_cast<size_type>(last - first))
    {
    }

    template <class Iterator,
              MIGRAPHX_REQUIRES(Extent != dynamic_extent and
                                detail::is_span_iterator<Iterator, T>{})>
    constexpr explicit span(Iterator first, Iterator last)
        : span(first, static_cast<size_type>(last - first))
    {
    }

    template <std::size_t N, MIGRAPHX_REQUIRES(Extent == dynamic_extent or Extent == N)>
    constexpr span(T (&arr)[N]) noexcept : storage(arr, N)
    {
    }

    template <class U,
              std::size_t N,
              MIGRAPHX_REQUIRES((Extent == dynamic_extent or Extent == N) and
                                detail::is_span_convertible<U, T>{})>
    constexpr span(std::array<U, N>& arr) noexcept : storage(arr.data(), N)
    {
    }

    template <class U,
              std::size_t N,
              MIGRAPHX_REQUIRES((Extent == dynamic_extent or Extent == N) and
                                detail::is_span_convertible<const U, T>{})>
    constexpr span(const std::array<U, N>& arr) noexcept : storage(arr.data(), N)
    {
    }

    template <class Range,
              MIGRAPHX_REQUIRES(Extent == dynamic_extent and
                                detail::is_span_range<std::remove_reference_t<Range>, T>{} and
                                detail::is_span_borrowable<Range, T>{})>
    constexpr span(Range&& r) : storage(std::data(r), std::size(r))
    {
    }

    template <class Range,
              MIGRAPHX_REQUIRES(Extent != dynamic_extent and
                                detail::is_span_range<std::remove_reference_t<Range>, T>{} and
                                detail::is_span_borrowable<Range, T>{})>
    constexpr explicit span(Range&& r) : storage(std::data(r), std::size(r))
    {
        assert(std::size(r) == Extent);
    }

    /// A braced list may only be viewed through a span of const elements, as in C++26
    template <MIGRAPHX_REQUIRES(Extent == dynamic_extent and std::is_const<T>{})>
    constexpr span(std::initializer_list<value_type> il) noexcept : storage(il.begin(), il.size())
    {
    }

    template <MIGRAPHX_REQUIRES(Extent != dynamic_extent and std::is_const<T>{})>
    constexpr explicit span(std::initializer_list<value_type> il) noexcept
        : storage(il.begin(), il.size())
    {
        assert(il.size() == Extent);
    }

    template <class U,
              std::size_t N,
              MIGRAPHX_REQUIRES((Extent == dynamic_extent or N == Extent) and
                                detail::is_span_convertible<U, T>{})>
    constexpr span(const span<U, N>& s) noexcept : storage(s.data(), s.size())
    {
    }

    template <class U,
              std::size_t N,
              MIGRAPHX_REQUIRES(Extent != dynamic_extent and N == dynamic_extent and
                                detail::is_span_convertible<U, T>{})>
    constexpr explicit span(const span<U, N>& s) noexcept : storage(s.data(), s.size())
    {
        assert(s.size() == Extent);
    }

    constexpr size_type size() const noexcept { return storage.size(); }
    constexpr size_type size_bytes() const noexcept { return size() * sizeof(T); }
    [[nodiscard]] constexpr bool empty() const noexcept { return size() == 0; }

    constexpr reference operator[](size_type i) const
    {
        assert(i < size());
        return storage.ptr[i];
    }

    constexpr reference at(size_type i) const
    {
        if(i >= size())
            throw std::out_of_range("span::at: index " + std::to_string(i) +
                                    " is out of range for size " + std::to_string(size()));
        return storage.ptr[i];
    }

    constexpr reference front() const
    {
        assert(not empty());
        return storage.ptr[0];
    }

    constexpr reference back() const
    {
        assert(not empty());
        return storage.ptr[size() - 1];
    }

    constexpr pointer data() const noexcept { return storage.ptr; }

    constexpr iterator begin() const noexcept { return storage.ptr; }
    constexpr iterator end() const noexcept { return storage.ptr + size(); }
    constexpr const_iterator cbegin() const noexcept { return begin(); }
    constexpr const_iterator cend() const noexcept { return end(); }
    constexpr reverse_iterator rbegin() const noexcept { return reverse_iterator(end()); }
    constexpr reverse_iterator rend() const noexcept { return reverse_iterator(begin()); }
    constexpr const_reverse_iterator crbegin() const noexcept
    {
        return const_reverse_iterator(cend());
    }
    constexpr const_reverse_iterator crend() const noexcept
    {
        return const_reverse_iterator(cbegin());
    }

    template <std::size_t Count>
    constexpr span<T, Count> first() const
    {
        static_assert(Count <= Extent, "span::first: count exceeds the extent");
        assert(Count <= size());
        return span<T, Count>(data(), Count);
    }

    constexpr span<T, dynamic_extent> first(size_type count) const
    {
        assert(count <= size());
        return {data(), count};
    }

    template <std::size_t Count>
    constexpr span<T, Count> last() const
    {
        static_assert(Count <= Extent, "span::last: count exceeds the extent");
        assert(Count <= size());
        return span<T, Count>(data() + (size() - Count), Count);
    }

    constexpr span<T, dynamic_extent> last(size_type count) const
    {
        assert(count <= size());
        return {data() + (size() - count), count};
    }

    template <std::size_t Offset, std::size_t Count = dynamic_extent>
    constexpr span<T, detail::subspan_extent<Extent, Offset, Count>()> subspan() const
    {
        static_assert(Offset <= Extent, "span::subspan: offset exceeds the extent");
        static_assert(Count == dynamic_extent or Count <= Extent - Offset,
                      "span::subspan: count exceeds the extent");
        assert(Offset <= size());
        assert(Count == dynamic_extent or Count <= size() - Offset);
        return span<T, detail::subspan_extent<Extent, Offset, Count>()>(
            data() + Offset, Count == dynamic_extent ? size() - Offset : Count);
    }

    constexpr span<T, dynamic_extent> subspan(size_type offset,
                                              size_type count = dynamic_extent) const
    {
        assert(offset <= size());
        assert(count == dynamic_extent or count <= size() - offset);
        return {data() + offset, count == dynamic_extent ? size() - offset : count};
    }

    private:
    detail::span_storage<T, Extent> storage;
};

template <class T, std::size_t N>
inline constexpr std::size_t span_bytes_extent =
    N == dynamic_extent ? dynamic_extent : sizeof(T) * N;

template <class T, std::size_t N>
span<const byte, span_bytes_extent<T, N>> as_bytes(span<T, N> s) noexcept
{
    return span<const byte, span_bytes_extent<T, N>>(reinterpret_cast<const byte*>(s.data()),
                                                     s.size_bytes());
}

template <class T, std::size_t N, MIGRAPHX_REQUIRES(not std::is_const<T>{})>
span<byte, span_bytes_extent<T, N>> as_writable_bytes(span<T, N> s) noexcept
{
    return span<byte, span_bytes_extent<T, N>>(reinterpret_cast<byte*>(s.data()), s.size_bytes());
}

template <class T>
span(T*, std::size_t) -> span<T>;

template <class T>
span(T*, T*) -> span<T>;

template <class T, std::size_t N>
span(T (&)[N]) -> span<T, N>;

template <class T, std::size_t N>
span(std::array<T, N>&) -> span<T, N>;

template <class T, std::size_t N>
span(const std::array<T, N>&) -> span<const T, N>;

template <class Range>
span(Range&&) -> span<std::remove_pointer_t<detail::range_pointer_t<Range>>>;

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
