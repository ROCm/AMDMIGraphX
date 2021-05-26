#ifndef MIGRAPHX_GUARD_ERASE_HPP
#define MIGRAPHX_GUARD_ERASE_HPP

#include <algorithm>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/**
 * @brief Erase all elements from a container
 *
 * @param r The container to erase elements from
 * @param value The value to be erased
 * @return Returns iterator to erased element
 */
template <class R, class T>
auto erase(R&& r, const T& value)
{
    return r.erase(std::remove(r.begin(), r.end(), value), r.end());
}

/**
 * @brief Erase all elements from a container
 *
 * @param r The container to erase elements from
 * @param pred Predicate function that selects which elements should be erased.
 */
template <class R, class P>
void erase_if(R&& r, P&& pred)
{
    auto first = r.begin();
    auto last  = r.end();
    while(first != last)
    {
        if(pred(*first))
            first = r.erase(first);
        else
            first++;
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
