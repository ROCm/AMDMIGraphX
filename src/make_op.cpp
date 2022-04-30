#include <migraphx/make_op.hpp>
#include <migraphx/register_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

operation make_op(const std::string& name) { return load_op(name); }

template<class F>
operation make_op_generic(const std::string& name,
                  F for_each)
{
    auto op = load_op(name);
    // Merge values
    value w = op.to_value();
    for_each([&](const auto& key, const auto& x) {
        if(not w.contains(key))
            // NOLINTNEXTLINE(performance-inefficient-string-concatenation)
            MIGRAPHX_THROW("No key '" + key + "' in " + name);
        w.at(key) = x;
    });
    op.from_value(w);
    return op;
}


operation make_op(const std::string& name,
                  const std::initializer_list<std::pair<std::string, value>>& v)
{
    return make_op_generic(name, [&](auto f) {
        for(auto&& [key, x] : v) 
            f(key, x); 
    });
}

operation make_op_from_value(const std::string& name, const value& v)
{
    if(not(v.is_object() or (v.empty() and v.is_array())))
        MIGRAPHX_THROW("Value is not an object for make_op: " + name);
    return make_op_generic(name, [&](auto f) {
        for(auto&& x : v)
            f(x.get_key(), x.without_key());
    });
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
