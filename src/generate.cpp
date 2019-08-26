#include <migraphx/generate.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

argument fill_argument(shape s, unsigned long value)
{
    argument result;
    s.visit_type([&](auto as) {
        using type = typename decltype(as)::type;
        auto v     = fill_tensor_data<type>(s, value);
        result     = {s, [v]() mutable { return reinterpret_cast<char*>(v.data()); }};
    });
    return result;
}

argument generate_argument(shape s, unsigned long seed)
{
    argument result;
    s.visit_type([&](auto as) {
        using type = typename decltype(as)::type;
        auto v     = generate_tensor_data<type>(s, seed);
        result     = {s, [v]() mutable { return reinterpret_cast<char*>(v.data()); }};
    });
    return result;
}

literal generate_literal(shape s, unsigned long seed)
{
    literal result;
    s.visit_type([&](auto as) {
        using type = typename decltype(as)::type;
        auto v     = generate_tensor_data<type>(s, seed);
        result     = {s, v};
    });
    return result;
}

// TODO: Move to literal.cpp
literal abs(literal l)
{
    return transform(std::move(l), [](auto x) { return std::fabs(x); });
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
