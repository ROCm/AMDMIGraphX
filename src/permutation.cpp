
#include <migraphx/permutation.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/algorithm.hpp>
#include <map>
#include <functional>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

shape reorder_shape(const shape& s, const std::vector<int64_t>& permutation)
{
    return {s.type(), reorder_dims(s.lens(), permutation), reorder_dims(s.strides(), permutation)};
}

std::vector<int64_t> invert_permutation(const std::vector<int64_t>& permutation)
{
    return sort_permutation(permutation, std::less<>{});
}

std::vector<int64_t> find_permutation(const shape& s)
{
    std::vector<std::int64_t> result(s.lens().size());
    std::iota(result.begin(), result.end(), 0);
    std::stable_sort(result.begin(), result.end(), by(std::greater<>{}, [&](auto x) {
                         return std::make_tuple(s.strides()[x], s.lens()[x]);
                     }));
    return result;
}

std::vector<int64_t> find_permutation(const std::vector<shape>& shapes)
{
    if(shapes.empty())
        return {};
    std::map<std::vector<int64_t>, std::size_t> count;
    for(auto&& s : shapes)
    {
        if(s.broadcasted())
            continue;
        count[find_permutation(s)]++;
    }
    if(count.empty())
    {
        std::vector<int64_t> r(shapes.front().lens().size());
        std::iota(r.begin(), r.end(), 0);
        return r;
    }
    auto it = std::max_element(
        count.begin(), count.end(), by(std::less<>{}, [](auto&& p) { return p.second; }));
    assert(it != count.end());
    return it->first;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
