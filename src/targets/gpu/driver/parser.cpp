#include <migraphx/gpu/driver/parser.hpp>
#include <migraphx/gpu/driver/action.hpp>
#include <iostream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace driver {

[[noreturn]] void error(const std::string& msg)
{
    std::cout << msg << std::endl;
    std::abort();
}

shape parser::parse_shape(const value& v) const
{
    auto lens    = get(v, "lens", std::vector<std::size_t>{});
    auto strides = get(v, "strides", std::vector<std::size_t>{});
    auto type    = shape::parse_type(get<std::string>(v, "type", "float"));
    if(strides.empty())
        return shape{type, lens};
    else
        return shape{type, lens, strides};
}

std::vector<shape> parser::parse_shapes(const value& v) const
{
    std::vector<shape> result;
    std::transform(
        v.begin(), v.end(), std::back_inserter(result), [&](auto&& x) { return parse_shape(x); });
    return result;
}

void parser::load_settings(const value& v)
{
    if(v.contains("settings"))
        settings = v.at("settings");
}

void parser::process(const value& v)
{
    if(not v.is_object())
        error("Input is not an object");
    parser p{};
    p.load_settings(v);
    for(auto&& pp : v)
    {
        if(pp.get_key() == "settings")
            continue;
        get_action(pp.get_key())(p, pp.without_key());
    }
}

} // namespace driver
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
