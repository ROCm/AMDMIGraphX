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

template <class InputIt, class OutputIt>
constexpr OutputIt copy(InputIt first, InputIt last, OutputIt d_first)
{
    while(first != last)
    {
        *d_first++ = *first++;
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
            if(!(*it == *s_it))
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
