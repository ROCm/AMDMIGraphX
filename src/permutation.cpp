
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

    // std::vector<std::int64_t> result;
    // std::multimap<std::size_t, int64_t, std::greater<>> m;
    // int i = 0;
    // for(auto&& x : s.strides())
    //     m.emplace(x, i++);
    // auto pred = by(std::equal_to<>{}, [](auto&& p) { return p.first; });
    // auto each = [&](auto start, auto last) {
    //     auto n = std::distance(start, last);
    //     std::transform(start, last, std::back_inserter(result), [](auto&& p) { return p.second;
    //     }); std::stable_sort(result.end() - n, result.end(), by(std::greater<>{}, [&](auto x) {
    //                          return s.lens()[x];
    //                      }));
    // };
    // group_unique(m.begin(), m.end(), each, pred);
    // return result;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
