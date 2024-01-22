#ifndef MIGRAPHX_GUARD_GPU_PROBLEM_CACHE_HPP
#define MIGRAPHX_GUARD_GPU_PROBLEM_CACHE_HPP

#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/optional.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
struct problem_cache
{
    bool has(const std::string& name, const value& problem) const;
    void insert(const std::string& name, const value& problem, const value& solution);
    void mark(const std::string& name, const value& problem);
    optional<value> get(const std::string& name, const value& problem) const;
    std::unordered_map<value, value> cache;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_PROBLEM_CACHE_HPP
