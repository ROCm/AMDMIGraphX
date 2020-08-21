#include <migraphx/make_op.hpp>
#include <migraphx/register_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

operation make_op(const std::string& name) { return load_op(name); }
operation make_op(const std::string& name, const value& v)
{
    auto op = load_op(name);
    op.from_value(v);
    return op;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
