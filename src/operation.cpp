#include <migraphx/operation.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void migraphx_to_value(value& v, const operation& op)
{
    v["name"]     = op.name();
    v["operator"] = op.to_value();
}
void migraphx_from_value(const value& v, operation& op)
{
    op = make_op(v.at("name").to<std::string>(), v.at("operator"));
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
