#include <migraphx/make_op.hpp>
#include <migraphx/register_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

operation make_op(const std::string& name) { return load_op(name); }
operation make_op(const std::string& name,
                  const std::initializer_list<std::pair<std::string, value>>& v)
{
    // if(not(v.is_object() or (v.empty() and v.is_array())))
    //     MIGRAPHX_THROW("Value is not an object for make_op: " + name);
    auto op = load_op(name);
    // Merge values
    value w = op.to_value();
    for(auto&& [key, x] : v)
    {
        if(not w.contains(key))
            MIGRAPHX_THROW("No key '" + key + "' in " + name);
        w.at(key) = x;
    }
    op.from_value(w);
    return op;
}

operation make_op_from_value(const std::string& name, const value& v)
{
    if(not(v.is_object() or (v.empty() and v.is_array())))
        MIGRAPHX_THROW("Value is not an object for make_op: " + name);
    auto op = load_op(name);
    // Merge values
    value w = op.to_value();
    for(auto&& x : v)
    {
        if(not w.contains(x.get_key()))
            MIGRAPHX_THROW("No key '" + x.get_key() + "' in " + name);
        w.at(x.get_key()) = x.without_key();
    }
    op.from_value(w);
    return op;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
