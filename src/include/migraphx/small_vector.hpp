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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_SMALL_VECTOR_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_SMALL_VECTOR_HPP

#include <migraphx/config.hpp>
#include <migraphx/iota_iterator.hpp>
#include <migraphx/requires.hpp>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/// Number of inline elements that fit in 32 bytes, with a minimum of 2
template <class T>
constexpr std::size_t small_vector_default_size()
{
    return std::max<std::size_t>(2, 32 / sizeof(T));
}

namespace detail {

template <class Iterator, class = void>
struct is_input_iterator : std::false_type
{
};

template <class Iterator>
struct is_input_iterator<Iterator,
                         std::void_t<typename std::iterator_traits<Iterator>::iterator_category>>
    : std::is_base_of<std::input_iterator_tag,
                      typename std::iterator_traits<Iterator>::iterator_category>
{
};

template <class Iterator>
using is_forward_iterator =
    std::is_base_of<std::forward_iterator_tag,
                    typename std::iterator_traits<Iterator>::iterator_category>;

} // namespace detail

/// A vector that stores up to N elements inline before allocating on the heap.
/// Mirrors the std::vector interface without the allocator.
template <class T, std::size_t N = small_vector_default_size<T>()>
class small_vector
{
    public:
    using value_type             = T;
    using size_type              = std::size_t;
    using difference_type        = std::ptrdiff_t;
    using reference              = T&;
    using const_reference        = const T&;
    using pointer                = T*;
    using const_pointer          = const T*;
    using iterator               = T*;
    using const_iterator         = const T*;
    using reverse_iterator       = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    small_vector() noexcept {}

    explicit small_vector(size_type n) : small_vector() { resize(n); }

    small_vector(size_type n, const T& value) : small_vector() { assign(n, value); }

    template <class Iterator, MIGRAPHX_REQUIRES(detail::is_input_iterator<Iterator>{})>
    small_vector(Iterator first, Iterator last) : small_vector()
    {
        assign(first, last);
    }

    small_vector(std::initializer_list<T> il) : small_vector() { assign(il); }

    // cppcheck-suppress missingMemberCopy
    small_vector(const small_vector& other) : small_vector() { assign(other.begin(), other.end()); }

    small_vector(small_vector&& other) noexcept(std::is_nothrow_move_constructible<T>{})
        : small_vector()
    {
        take(other);
    }

    ~small_vector()
    {
        destroy_elements();
        free_heap();
    }

    // Members are set through assign(), which cppcheck cannot follow
    // cppcheck-suppress operatorEqVarError
    small_vector& operator=(const small_vector& other)
    {
        if(this != &other)
            assign(other.begin(), other.end());
        return *this;
    }

    // The raw buffer is never assigned as a whole; elements are constructed in place
    // cppcheck-suppress operatorEqVarError
    small_vector& operator=(small_vector&& other) noexcept(std::is_nothrow_move_constructible<T>{})
    {
        if(this != &other)
            take(other);
        return *this;
    }

    small_vector& operator=(std::initializer_list<T> il)
    {
        assign(il);
        return *this;
    }

    void assign(size_type n, const T& value)
    {
        auto [first, last] = repeat(value, n);
        assign(first, last);
    }

    template <class Iterator, MIGRAPHX_REQUIRES(detail::is_input_iterator<Iterator>{})>
    void assign(Iterator first, Iterator last)
    {
        if constexpr(detail::is_forward_iterator<Iterator>{})
        {
            size_type n = std::distance(first, last);
            if(n > cap)
            {
                auto buf = allocate(n);
                std::uninitialized_copy(first, last, buf.get());
                adopt(buf, n, n);
            }
            else if(n > len)
            {
                Iterator mid = std::next(first, len);
                std::copy(first, mid, ptr);
                std::uninitialized_copy(mid, last, ptr + len);
                len = n;
            }
            else
            {
                T* new_end = std::copy(first, last, ptr);
                std::destroy(new_end, ptr + len);
                len = n;
            }
        }
        else
        {
            clear();
            std::copy(first, last, std::back_inserter(*this));
        }
    }

    void assign(std::initializer_list<T> il) { assign(il.begin(), il.end()); }

    // Element access
    reference at(size_type i)
    {
        check_index(i);
        return ptr[i];
    }

    const_reference at(size_type i) const
    {
        check_index(i);
        return ptr[i];
    }

    reference operator[](size_type i)
    {
        assert(i < len);
        return ptr[i];
    }

    const_reference operator[](size_type i) const
    {
        assert(i < len);
        return ptr[i];
    }

    reference front()
    {
        assert(len > 0);
        return ptr[0];
    }

    const_reference front() const
    {
        assert(len > 0);
        return ptr[0];
    }

    reference back()
    {
        assert(len > 0);
        return ptr[len - 1];
    }

    const_reference back() const
    {
        assert(len > 0);
        return ptr[len - 1];
    }

    T* data() noexcept { return ptr; }
    const T* data() const noexcept { return ptr; }

    // Iterators
    iterator begin() noexcept { return ptr; }
    const_iterator begin() const noexcept { return ptr; }
    const_iterator cbegin() const noexcept { return ptr; }

    iterator end() noexcept { return ptr + len; }
    const_iterator end() const noexcept { return ptr + len; }
    const_iterator cend() const noexcept { return ptr + len; }

    reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
    const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
    const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(end()); }

    reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
    const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
    const_reverse_iterator crend() const noexcept { return const_reverse_iterator(begin()); }

    // Capacity
    bool empty() const noexcept { return len == 0; }
    size_type size() const noexcept { return len; }
    size_type max_size() const noexcept { return alloc_traits::max_size(std::allocator<T>{}); }
    size_type capacity() const noexcept { return cap; }

    void reserve(size_type n)
    {
        if(n > max_size())
            throw std::length_error("small_vector::reserve: requested size exceeds max_size");
        if(n > cap)
            reallocate(n);
    }

    void shrink_to_fit()
    {
        if(is_inline())
            return;
        if(len <= N)
        {
            relocate(begin(), end(), inline_data());
            destroy_elements();
            free_heap();
            ptr = inline_data();
            cap = N;
        }
        else if(len < cap)
        {
            reallocate(len);
        }
    }

    // Modifiers
    void clear() noexcept
    {
        destroy_elements();
        len = 0;
    }

    iterator insert(const_iterator pos, const T& value) { return emplace(pos, value); }

    iterator insert(const_iterator pos, T&& value) { return emplace(pos, std::move(value)); }

    iterator insert(const_iterator pos, size_type n, const T& value)
    {
        // Copy first since value may alias an element that gets shifted
        // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
        const T tmp(value);
        auto [first, last] = repeat(tmp, n);
        return insert(pos, first, last);
    }

    template <class Iterator, MIGRAPHX_REQUIRES(detail::is_input_iterator<Iterator>{})>
    iterator insert(const_iterator pos, Iterator first, Iterator last)
    {
        if constexpr(detail::is_forward_iterator<Iterator>{})
        {
            size_type index = pos - cbegin();
            size_type n     = std::distance(first, last);
            if(n == 0)
                return ptr + index;
            if(len + n > cap)
                return grow_insert(
                    index, n, [&](T* gap) { std::uninitialized_copy(first, last, gap); });
            return insert_shift(index, first, last, n);
        }
        else
        {
            small_vector tmp(first, last);
            return insert(
                pos, std::make_move_iterator(tmp.begin()), std::make_move_iterator(tmp.end()));
        }
    }

    iterator insert(const_iterator pos, std::initializer_list<T> il)
    {
        return insert(pos, il.begin(), il.end());
    }

    template <class... Ts>
    iterator emplace(const_iterator pos, Ts&&... xs)
    {
        size_type index = pos - cbegin();
        if(len == cap)
            return grow_insert(index, 1, [&](T* gap) { construct(gap, std::forward<Ts>(xs)...); });
        T* p = ptr + index;
        if(index == len)
        {
            construct(p, std::forward<Ts>(xs)...);
            ++len;
            return p;
        }
        // Build first since the arguments may alias an element that gets shifted
        T tmp(std::forward<Ts>(xs)...);
        T* last = ptr + len;
        construct(last, std::move(*(last - 1)));
        ++len;
        std::move_backward(p, last - 1, last);
        *p = std::move(tmp);
        return p;
    }

    iterator erase(const_iterator pos) { return erase(pos, pos + 1); }

    iterator erase(const_iterator first, const_iterator last)
    {
        T* p = ptr + (first - cbegin());
        if(first == last)
            return p;
        T* new_end = std::move(p + (last - first), end(), p);
        std::destroy(new_end, end());
        len -= last - first;
        return p;
    }

    void push_back(const T& value) { emplace_back(value); }

    void push_back(T&& value) { emplace_back(std::move(value)); }

    template <class... Ts>
    reference emplace_back(Ts&&... xs)
    {
        return *emplace(end(), std::forward<Ts>(xs)...);
    }

    void pop_back()
    {
        assert(len > 0);
        --len;
        std::destroy_at(ptr + len);
    }

    void resize(size_type n)
    {
        if(n < len)
            erase(begin() + n, end());
        else
            append(n - len,
                   [count = n - len](T* gap) { std::uninitialized_value_construct_n(gap, count); });
    }

    void resize(size_type n, const T& value)
    {
        if(n < len)
            erase(begin() + n, end());
        else
            append(n - len, [count = n - len, &value](T* gap) {
                std::uninitialized_fill_n(gap, count, value);
            });
    }

    void swap(small_vector& other) noexcept(std::is_nothrow_move_constructible<T>{} and
                                            std::is_nothrow_swappable<T>{})
    {
        if(this == &other)
            return;
        if(is_inline() and other.is_inline())
        {
            if(len < other.len)
                swap_inline(*this, other);
            else
                swap_inline(other, *this);
        }
        else if(is_inline())
        {
            swap_mixed(*this, other);
        }
        else if(other.is_inline())
        {
            swap_mixed(other, *this);
        }
        else
        {
            std::swap(ptr, other.ptr);
            std::swap(len, other.len);
            std::swap(cap, other.cap);
        }
    }

    private:
    using alloc_traits = std::allocator_traits<std::allocator<T>>;

    struct deallocator
    {
        size_type n;
        void operator()(T* p) const noexcept { std::allocator<T>{}.deallocate(p, n); }
    };
    using buffer_ptr = std::unique_ptr<T, deallocator>;

    // Destroys the constructed range [first, last) unless released
    struct destroy_guard
    {
        T* first;
        T* last;
        destroy_guard(T* pfirst, T* plast) : first(pfirst), last(plast) {}
        destroy_guard(const destroy_guard&)            = delete;
        destroy_guard& operator=(const destroy_guard&) = delete;
        ~destroy_guard() { std::destroy(first, last); }
        void release() { first = last; }
    };

    // Raw storage for the inline elements; their lifetimes are managed by the vector
    union inline_storage
    {
        // cppcheck-suppress uninitMemberVar
        inline_storage() noexcept {}
        ~inline_storage() {}
        inline_storage(const inline_storage&)            = delete;
        inline_storage& operator=(const inline_storage&) = delete;
        std::array<T, N> elements;
    };

    T* inline_data() noexcept { return buffer.elements.data(); }
    const T* inline_data() const noexcept { return buffer.elements.data(); }
    bool is_inline() const noexcept { return ptr == inline_data(); }

    void check_index(size_type i) const
    {
        if(i >= len)
            throw std::out_of_range("small_vector::at: index " + std::to_string(i) +
                                    " is out of range for size " + std::to_string(len));
    }

    static buffer_ptr allocate(size_type n)
    {
        return buffer_ptr{std::allocator<T>{}.allocate(n), deallocator{n}};
    }

    template <class... Ts>
    static void construct(T* p, Ts&&... xs)
    {
        std::allocator<T> a;
        alloc_traits::construct(a, p, std::forward<Ts>(xs)...);
    }

    // Moves elements into uninitialized storage, copying instead when the move could throw
    static T* relocate(T* first, T* last, T* dest)
    {
        if constexpr(std::is_nothrow_move_constructible<T>{} or not std::is_copy_constructible<T>{})
            return std::uninitialized_move(first, last, dest);
        else
            return std::uninitialized_copy(first, last, dest);
    }

    // Random access range that yields value n times
    static auto repeat(const T& value, size_type n)
    {
        std::ptrdiff_t count = n;
        auto f               = [&value](std::ptrdiff_t) -> const T& { return value; };
        return std::make_pair(make_basic_iota_iterator(std::ptrdiff_t{0}, f),
                              make_basic_iota_iterator(count, f));
    }

    void destroy_elements() noexcept { std::destroy(begin(), end()); }

    void free_heap() noexcept
    {
        if(not is_inline())
            deallocator{cap}(ptr);
    }

    // Replaces the storage with buf, whose elements are already constructed
    void adopt(buffer_ptr& buf, size_type new_cap, size_type new_len) noexcept
    {
        destroy_elements();
        free_heap();
        ptr = buf.release();
        cap = new_cap;
        len = new_len;
    }

    size_type grow_capacity(size_type min_cap) const noexcept { return std::max(min_cap, 2 * cap); }

    void reallocate(size_type new_cap)
    {
        auto buf = allocate(new_cap);
        relocate(begin(), end(), buf.get());
        adopt(buf, new_cap, len);
    }

    // Reallocates with an n element gap at index that construct_gap fills before the
    // existing elements are relocated, so the arguments may alias the old elements
    template <class F>
    T* grow_insert(size_type index, size_type n, F construct_gap)
    {
        size_type new_cap = grow_capacity(len + n);
        auto buf          = allocate(new_cap);
        T* gap            = buf.get() + index;
        construct_gap(gap);
        destroy_guard guard{gap, gap + n};
        relocate(ptr, ptr + index, buf.get());
        guard.first = buf.get();
        relocate(ptr + index, ptr + len, gap + n);
        guard.release();
        adopt(buf, new_cap, len + n);
        return ptr + index;
    }

    // Inserts n elements at index within the current capacity
    template <class Iterator>
    T* insert_shift(size_type index, Iterator first, Iterator last, size_type n)
    {
        T* p           = ptr + index;
        T* old_end     = ptr + len;
        size_type tail = len - index;
        if(tail > n)
        {
            std::uninitialized_move(old_end - n, old_end, old_end);
            len += n;
            std::move_backward(p, old_end - n, old_end);
            std::copy(first, last, p);
        }
        else
        {
            Iterator mid = std::next(first, tail);
            std::uninitialized_copy(mid, last, old_end);
            len += n - tail;
            std::uninitialized_move(p, old_end, old_end + (n - tail));
            len += tail;
            std::copy(first, mid, p);
        }
        return p;
    }

    // Appends n elements that construct_gap fills
    template <class F>
    void append(size_type n, F construct_gap)
    {
        if(len + n > cap)
        {
            grow_insert(len, n, construct_gap);
        }
        else
        {
            construct_gap(ptr + len);
            len += n;
        }
    }

    // Steals the heap buffer or moves the inline elements, leaving other empty
    void take(small_vector& other)
    {
        if(other.is_inline())
        {
            clear();
            std::uninitialized_move(other.begin(), other.end(), ptr);
            len = other.len;
            other.clear();
        }
        else
        {
            destroy_elements();
            free_heap();
            ptr       = other.ptr;
            len       = other.len;
            cap       = other.cap;
            other.ptr = other.inline_data();
            other.len = 0;
            other.cap = N;
        }
    }

    // Both inline: swap the common prefix and move the tail of the longer one over
    static void swap_inline(small_vector& shorter, small_vector& longer)
    {
        std::swap_ranges(shorter.begin(), shorter.end(), longer.begin());
        std::uninitialized_move(longer.begin() + shorter.len, longer.end(), shorter.end());
        std::destroy(longer.begin() + shorter.len, longer.end());
        std::swap(shorter.len, longer.len);
    }

    // Move the inline elements into the unused inline buffer of the heap one, then
    // hand its heap buffer over
    static void swap_mixed(small_vector& inl, small_vector& heap)
    {
        std::uninitialized_move(inl.begin(), inl.end(), heap.inline_data());
        inl.destroy_elements();
        inl.ptr  = heap.ptr;
        heap.ptr = heap.inline_data();
        std::swap(inl.len, heap.len);
        std::swap(inl.cap, heap.cap);
    }

    inline_storage buffer{};
    T* ptr        = inline_data();
    size_type len = 0;
    size_type cap = N;
};

template <class Iterator, MIGRAPHX_REQUIRES(detail::is_input_iterator<Iterator>{})>
small_vector(Iterator, Iterator)
    -> small_vector<typename std::iterator_traits<Iterator>::value_type>;

template <class T, std::size_t N, std::size_t M>
bool operator==(const small_vector<T, N>& x, const small_vector<T, M>& y)
{
    return std::equal(x.begin(), x.end(), y.begin(), y.end());
}

template <class T, std::size_t N, std::size_t M>
bool operator!=(const small_vector<T, N>& x, const small_vector<T, M>& y)
{
    return not(x == y);
}

template <class T, std::size_t N, std::size_t M>
bool operator<(const small_vector<T, N>& x, const small_vector<T, M>& y)
{
    return std::lexicographical_compare(x.begin(), x.end(), y.begin(), y.end());
}

template <class T, std::size_t N, std::size_t M>
bool operator>(const small_vector<T, N>& x, const small_vector<T, M>& y)
{
    return y < x;
}

template <class T, std::size_t N, std::size_t M>
bool operator<=(const small_vector<T, N>& x, const small_vector<T, M>& y)
{
    return not(y < x);
}

template <class T, std::size_t N, std::size_t M>
bool operator>=(const small_vector<T, N>& x, const small_vector<T, M>& y)
{
    return not(x < y);
}

template <class T, std::size_t N>
void swap(small_vector<T, N>& x, small_vector<T, N>& y) noexcept(noexcept(x.swap(y)))
{
    x.swap(y);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
