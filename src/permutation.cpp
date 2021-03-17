
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
    std::vector<std::int64_t> result;
    std::multimap<std::size_t, int64_t, std::greater<>> m;
    int i = 0;
    for(auto&& x : s.strides())
        m.emplace(x, i++);
    auto pred     = by(std::equal_to<>{}, [](auto&& p) { return p.first; });
    int64_t base  = 0;
    int64_t delta = 0;
    auto each     = [&](auto start, auto last) {
        auto n = std::distance(start, last);
        std::transform(start, last, std::back_inserter(result), [](auto&& p) { return p.second; });
        auto mag = delta > 0 ? -1 : 1;
        std::stable_sort(result.end() - n, result.end(), by(std::less<>{}, [&](auto x) {
                             auto diff = (x - base);
                             return std::make_tuple(std::abs(diff), (x - base) * mag);
                         }));
        delta = result.back() - base;
        base  = result.back();
    };
    group_unique(m.begin(), m.end(), each, pred);
    return result;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
