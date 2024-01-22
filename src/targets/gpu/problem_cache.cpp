#include <migraphx/gpu/problem_cache.hpp>
#include <migraphx/ranges.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

static value create_key(const std::string& name, const value& problem)
{
    return {{"name", name}, {"problem", problem}};
}

bool problem_cache::has(const std::string& name, const value& problem) const
{
    return contains(cache, create_key(name, problem));
}
void problem_cache::insert(const std::string& name, const value& problem, const value& solution)
{
    assert(not solution.is_null());
    cache[create_key(name, problem)] = solution;
}
void problem_cache::mark(const std::string& name, const value& problem)
{
    cache.insert(std::make_pair(create_key(name, problem), value{}));
}
optional<value> problem_cache::get(const std::string& name, const value& problem) const
{
    auto it = cache.find(create_key(name, problem));
    if(it == cache.end())
        return nullopt;
    return it->second;
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
