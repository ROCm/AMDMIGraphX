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

} // namespace migraphx

#endif
