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
#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_KERNELS_ALGORITHM_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_KERNELS_ALGORITHM_HPP

namespace migraphx {

struct less
{
    template <class T, class U>
    constexpr auto operator()(T x, U y) const
    {
        return x < y;
    }
};

struct greater
{
    template <class T, class U>
    constexpr auto operator()(T x, U y) const
    {
        return x > y;
    }
};

template <class InputIt, class T, class BinaryOperation>
constexpr T accumulate(InputIt first, InputIt last, T init, BinaryOperation op)
{
    for(; first != last; ++first)
    {
        init = op(static_cast<T&&>(init), *first);
    }
    return init;
}

template <class InputIt, class OutputIt>
constexpr OutputIt copy(InputIt first, InputIt last, OutputIt d_first)
{
    while(first != last)
    {
        *d_first++ = *first++;
    }
    return d_first;
}

template <class InputIt, class OutputIt, class UnaryPredicate>
constexpr OutputIt copy_if(InputIt first, InputIt last, OutputIt d_first, UnaryPredicate pred)
{
    for(; first != last; ++first)
    {
        if(pred(*first))
        {
            *d_first = *first;
            ++d_first;
        }
    }
    return d_first;
}

template <class Iterator, class Compare>
constexpr Iterator is_sorted_until(Iterator first, Iterator last, Compare comp)
{
    if(first != last)
    {
        Iterator next = first;
        while(++next != last)
        {
            if(comp(*next, *first))
                return next;
            first = next;
        }
    }
    return last;
}

template <class Iterator, class Compare>
constexpr bool is_sorted(Iterator first, Iterator last, Compare comp)
{
    return is_sorted_until(first, last, comp) == last;
}

template <class Iterator, class F>
constexpr F for_each(Iterator first, Iterator last, F f)
{
    for(; first != last; ++first)
    {
        f(*first);
    }
    return f;
}

template <class Iterator, class Predicate>
constexpr Iterator find_if(Iterator first, Iterator last, Predicate p)
{
    for(; first != last; ++first)
    {
        if(p(*first))
        {
            return first;
        }
    }
    return last;
}

template <class Iterator, class T>
constexpr Iterator find(Iterator first, Iterator last, const T& value)
{
    return find_if(first, last, [&](const auto& x) { return x == value; });
}

template <class InputIt, class UnaryPredicate>
constexpr bool any_of(InputIt first, InputIt last, UnaryPredicate p)
{
    return find_if(first, last, p) != last;
}

template <class InputIt, class UnaryPredicate>
constexpr bool none_of(InputIt first, InputIt last, UnaryPredicate p)
{
    return find_if(first, last, p) == last;
}

template <class InputIt, class UnaryPredicate>
constexpr bool all_of(InputIt first, InputIt last, UnaryPredicate p)
{
    return none_of(first, last, [=](auto&& x) { return not p(x); });
}

template <class Iterator1, class Iterator2>
constexpr Iterator1 search(Iterator1 first, Iterator1 last, Iterator2 s_first, Iterator2 s_last)
{
    for(;; ++first)
    {
        Iterator1 it = first;
        for(Iterator2 s_it = s_first;; ++it, ++s_it)
        {
            if(s_it == s_last)
            {
                return first;
            }
            if(it == last)
            {
                return last;
            }
            if(not(*it == *s_it))
            {
                break;
            }
        }
    }
}

template <class InputIt1, class InputIt2, class T, class BinaryOperation1, class BinaryOperation2>
constexpr T inner_product(InputIt1 first1,
                          InputIt1 last1,
                          InputIt2 first2,
                          T init,
                          BinaryOperation1 op1,
                          BinaryOperation2 op2)
{
    while(first1 != last1)
    {
        init = op1(init, op2(*first1, *first2));
        ++first1;
        ++first2;
    }
    return init;
}

template <class InputIt1, class InputIt2, class T>
constexpr T inner_product(InputIt1 first1, InputIt1 last1, InputIt2 first2, T init)
{
    return inner_product(
        first1,
        last1,
        first2,
        init,
        [](auto x, auto y) { return x + y; },
        [](auto x, auto y) { return x * y; });
}

} // namespace migraphx

#endif
