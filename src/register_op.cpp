#include <migraphx/register_op.hpp>
#include <migraphx/ranges.hpp>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

std::unordered_map<std::string, operation>& op_map()
{
    static std::unordered_map<std::string, operation> m; // NOLINT
    return m;
}
void register_op(const operation& op) { op_map()[op.name()] = op; }
operation load_op(const std::string& name)
{
    return at(op_map(), name, "Operator not found: " + name);
}

bool has_op(const std::string& name) { return op_map().count(name) == 1; }

std::vector<std::string> get_operators()
{
    std::vector<std::string> result;
    std::transform(op_map().begin(), op_map().end(), std::back_inserter(result), [&](auto&& p) {
        return p.first;
    });
    std::sort(result.begin(), result.end());
    return result;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
