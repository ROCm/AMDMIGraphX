#include <migraphx/gpu/compile_gen.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace gen {

static std::vector<std::size_t> vector_sizes(const std::vector<shape>& inputs)
{
    // If all inputs are half then only use half2
    if(std::all_of(inputs.begin(), inputs.end(), [](const auto& s) {
           return s.type() == shape::half_type;
       }))
        return {2};
    return {4, 2};
}

vectorize vectorize::elements(std::size_t axis, const std::vector<shape>& inputs)
{
    auto sizes = vector_sizes(inputs);
    std::vector<std::size_t> max_vec_size;
    std::transform(inputs.begin(),
                   inputs.end(),
                   std::back_inserter(max_vec_size),
                   [&](const auto& input) -> std::size_t {
                       auto stride = input.strides()[axis];
                       auto len    = input.lens()[axis];
                       if(stride != 0 and stride != 1)
                           return 1;
                       if(len == 1 and input.elements() > sizes.front())
                           return sizes.front();
                       auto it = std::find_if(
                           sizes.begin(), sizes.end(), [&](auto i) { return (len % i) == 0; });
                       if(it != sizes.end())
                           return *it;
                       return 1;
                   });
    return {*std::min_element(max_vec_size.begin(), max_vec_size.end()), axis};
}

std::string vectorize::str() const
{
    return "vectorize<" + to_string(size) + ", " + to_string(axis) + ">()";
}

preload preload::broadcasts(std::size_t axis, const std::vector<shape>& inputs)
{
    const std::size_t max_lds_bytes = 4096;
    std::vector<bool> result;
    std::transform(inputs.begin(),
                   inputs.end(),
                   std::back_inserter(result),
                   [&](const shape& input) { return input.strides()[axis] == 0; });
    auto bytes = std::inner_product(inputs.begin(),
                                    inputs.end(),
                                    result.begin(),
                                    std::size_t{0},
                                    std::plus<>{},
                                    [](const shape& s, bool b) -> std::size_t {
                                        if(b)
                                            return s.bytes();
                                        return 0;
                                    });
    if(bytes < max_lds_bytes)
        return {result};
    // TODO: Try to partially preload items
    std::fill(result.begin(), result.end(), false);
    return {result};
}

std::string preload::str() const
{
    std::vector<std::string> bool_strs;
    std::transform(args.begin(), std::prev(args.end()), std::back_inserter(bool_strs), [](bool b) {
        if(b)
            return "true";
        return "false";
    });
    return "auto_preload<false, " + join_strings(bool_strs, ", ") + ">(idx)";
}

bool preload::is_preloading() const
{
    return std::accumulate(args.begin(), args.end(), false, std::logical_or<>{});
}

std::size_t find_fast_axis(const std::vector<shape>& inputs)
{
    auto permutation = find_permutation(inputs);
    auto it          = std::max_element(permutation.begin(), permutation.end());
    return it - permutation.begin();
}

std::string make_transformer_args(std::vector<std::string> transformers)
{
    return join_strings(std::move(transformers), ", ");
}

} // namespace gen
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
