#ifndef MIGRAPHX_GUARD_MIGRAPHX_ITERATOR_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_ITERATOR_HPP

#include <migraphx/config.hpp>
#include <migraphx/rank.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class Iterator, class EndIterator>
auto is_end(rank<2>, Iterator it, EndIterator) -> decltype(!it._M_dereferenceable())
{
    return !it._M_dereferenceable();
}

template <class Iterator, class EndIterator>
auto is_end(rank<1>, Iterator it, EndIterator last)
{
    return it == last;
}

template <class Iterator, class EndIterator>
bool is_end(Iterator it, EndIterator last)
{
    return is_end(rank<2>{}, it, last);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_ITERATOR_HPP
