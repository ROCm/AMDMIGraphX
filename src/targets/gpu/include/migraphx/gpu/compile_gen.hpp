#ifndef MIGRAPHX_GUARD_GPU_COMPILE_GEN_HPP
#define MIGRAPHX_GUARD_GPU_COMPILE_GEN_HPP

#include <migraphx/config.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct shape;

namespace gpu {
namespace gen {

struct vectorize
{
    std::size_t size = 1;
    std::size_t axis = 0;
    static vectorize elements(std::size_t axis, const std::vector<shape>& inputs);
    std::string str() const;
};
struct preload
{
    std::vector<bool> args = {};
    static preload broadcasts(std::size_t axis, const std::vector<shape>& inputs);
    bool is_preloading() const;
    std::string str() const;
};

std::size_t find_fast_axis(const std::vector<shape>& inputs);

std::string make_transformer_args(std::vector<std::string> transformers);

template <class... Ts>
std::string make_transformer_args(Ts... xs)
{
    return make_transformer_args({xs.str()...});
}

} // namespace gen
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_COMPILE_GEN_HPP
