#include <migraph/generate.hpp>

namespace migraph {

migraph::argument generate_argument(migraph::shape s, std::mt19937::result_type seed)
{
    migraph::argument result;
    s.visit_type([&](auto as) {
        using type = typename decltype(as)::type;
        auto v     = generate_tensor_data<type>(s, seed);
        result     = {s, [v]() mutable { return reinterpret_cast<char*>(v.data()); }};
    });
    return result;
}

} // namespace migraph
