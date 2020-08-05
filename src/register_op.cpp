#include <migraphx/register_op.hpp>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

std::unordered_map<std::string, operation>& op_map() {
    static std::unordered_map<std::string, operation> m;
    return m;
}
void register_op(const operation& op) {
    op_map()[op.name()] = op;
}
operation load_op(const std::string& name) {
    return op_map().at(name);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
