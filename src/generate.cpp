#include <migraphx/generate.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

argument fill_argument(shape s, unsigned long value)
{
    argument result;
    if(s.type() == shape::tuple_type)
    {
        std::vector<argument> sub_args;
        const auto& sub_ss = s.sub_shapes();
        std::transform(sub_ss.begin(), sub_ss.end(), std::back_inserter(sub_args), [&](auto ss) {
            return fill_argument(ss, value);
        });

        result = argument(sub_args);
    }
    else
    {
        s.visit_type([&](auto as) {
            using type = typename decltype(as)::type;
            auto v     = fill_tensor_data<type>(s, value);
            result     = {s, v};
        });
    }
    return result;
}

argument generate_argument(shape s, unsigned long seed)
{
    argument result;
    if(s.type() == shape::tuple_type)
    {
        const auto& sub_ss = s.sub_shapes();
        std::vector<argument> sub_args;
        std::transform(sub_ss.begin(), sub_ss.end(), std::back_inserter(sub_args), [&](auto ss) {
            return generate_argument(ss, seed);
        });

        result = argument(sub_args);
    }
    else
    {
        s.visit_type([&](auto as) {
            // we use char type to store bool type internally, so bool_type
            // needs special processing to generate data
            if(s.type() == shape::bool_type)
            {
                auto v = generate_tensor_data<bool>(s, seed);
                result = {s, v};
            }
            else
            {
                using type = typename decltype(as)::type;
                auto v     = generate_tensor_data<type>(s, seed);
                result     = {s, v};
            }
        });
    }

    return result;
}

literal generate_literal(shape s, unsigned long seed)
{
    literal result;
    s.visit_type([&](auto as) {
        using type = typename decltype(as)::type;
        auto v     = generate_tensor_data<type>(s, seed);
        result     = {s, reinterpret_cast<char*>(v.get())};
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
