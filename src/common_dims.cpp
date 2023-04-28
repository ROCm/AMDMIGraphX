#include <migraphx/common_dims.hpp>
#include <migraphx/ranges.hpp>
#include <algorithm>
#include <cassert>
#include <numeric>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class Iterator>
static auto compute_end_dim(Iterator start, Iterator last, std::size_t dim)
{
    std::size_t x = 1;
    auto it       = std::find_if(start, last, [&](auto i) {
        x *= i;
        return x >= dim;
    });
    if(x != dim)
        return start;
    return it;
}

template<class Iterator>
static auto elements(Iterator start, Iterator last)
{
    return std::accumulate(start, last, std::size_t{1}, std::multiplies<>{});
}
template<class Range>
static auto elements(const Range& r)
{
    return elements(r.begin(), r.end());
}

common_dims common_dims::compute(const std::vector<std::size_t>& dims1, const std::vector<std::size_t>& dims2)
{
    assert(elements(dims1) == elements(dims2));
    common_dims cd;
    auto it1 = dims1.begin();
    auto it2 = dims2.begin();
    std::size_t rem1 = 1;
    std::size_t rem2 = 1;
    while(it1 != dims1.end() and it2 != dims2.end())
    {
        auto d1 = *it1;
        auto d2 = *it2;
        if (d1 == d2)
        {
            cd.axes_map1.push_back({cd.dims.size()});
            cd.axes_map2.push_back({cd.dims.size()});
            cd.dims.push_back(d1);
            it1++;
            it2++;
        }
        else if (d1 < d2)
        {
            auto dim_end = compute_end_dim(it1, dims1.begin(), d2);
            auto dims = range(it1, dim_end);
            auto n = elements(dims);
            if (n != d2)
            {
                // If not divisible then we can't compute a common dims
                if ((d2 % n) != 0)
                    return {};
                rem1 = d2 / n;
            }
            std::vector<std::size_t> axes(distance(dims));
            std::iota(axes.begin(), axes.end(), cd.dims.size());
            cd.axes_map1.push_back(axes);
            cd.axes_map2.push_back(axes);

            cd.dims.insert(cd.dims.end(), dims.begin(), dims.end());
            if (rem1 != 1)
                cd.dims.push_back(rem1);
            it1 += distance(dims);
            it2++;
        }
    }
    return cd;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
