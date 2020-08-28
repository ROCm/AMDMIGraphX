#include <migraphx/serialize.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class RawData>
void raw_data_to_value(value& v, const RawData& rd)
{
    value result;
    result["shape"] = migraphx::to_value(rd.get_shape());
    result["data"]  = std::string(rd.data(), rd.data() + rd.get_shape().bytes());
    v               = result;
}

void migraphx_to_value(value& v, const literal& l) { raw_data_to_value(v, l); }
void migraphx_from_value(const value& v, literal& l)
{
    auto s = migraphx::from_value<shape>(v.at("shape"));
    l      = literal(s, v.at("data").get_string().data());
}

void migraphx_to_value(value& v, const argument& a) { raw_data_to_value(v, a); }
void migraphx_from_value(const value& v, argument& a)
{
    literal l = migraphx::from_value<literal>(v);
    a         = l.get_argument();
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
