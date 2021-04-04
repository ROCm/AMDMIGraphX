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

} // namespace migraphx

#endif
