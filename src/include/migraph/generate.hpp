#ifndef MIGRAPH_GUARD_MIGRAPHLIB_GENERATE_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_GENERATE_HPP

#include <migraph/argument.hpp>
#include <migraph/literal.hpp>
#include <random>

namespace migraph {

template <class T>
std::vector<T> generate_tensor_data(migraph::shape s, std::mt19937::result_type seed = 0)
{
    std::vector<T> result(s.elements());
    std::mt19937 engine{seed};
    std::uniform_real_distribution<> dist;
    std::generate(result.begin(), result.end(), [&] { return dist(engine); });
    return result;
}

argument generate_argument(shape s, std::mt19937::result_type seed = 0);

literal generate_literal(shape s, std::mt19937::result_type seed = 0);

} // namespace migraph

#endif
