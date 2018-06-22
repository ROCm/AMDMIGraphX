#include <rtg/generate.hpp>

namespace rtg {

rtg::argument generate_argument(rtg::shape s, std::mt19937::result_type seed)
{
    rtg::argument result;
    s.visit_type([&](auto as)
    {
        using type = typename decltype(as)::type;
        auto v = generate_tensor_data<type>(s, seed);
        result = {s, [v]() mutable { return reinterpret_cast<char*>(v.data()); }};
    });
    return result;
}

} // namespace rtg
