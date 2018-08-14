#include <migraph/generate.hpp>

namespace migraph {

argument generate_argument(shape s, std::mt19937::result_type seed)
{
    argument result;
    s.visit_type([&](auto as) {
        using type = typename decltype(as)::type;
        auto v     = generate_tensor_data<type>(s, seed);
        result     = {s, [v]() mutable { return reinterpret_cast<char*>(v.data()); }};
    });
    return result;
}

literal generate_literal(shape s, std::mt19937::result_type seed)
{
    literal result;
    s.visit_type([&](auto as) {
        using type = typename decltype(as)::type;
        auto v     = generate_tensor_data<type>(s, seed);
        result     = {s, v};
    });
    return result;
}

} // namespace migraph
