#ifndef RTG_GUARD_RTGLIB_GENERATE_HPP
#define RTG_GUARD_RTGLIB_GENERATE_HPP

#include <rtg/argument.hpp>
#include <random>

namespace rtg {

template <class T>
std::vector<T> generate_tensor_data(rtg::shape s, std::mt19937::result_type seed = 0)
{
    std::vector<T> result(s.elements());
    std::mt19937 engine{seed};
    std::uniform_real_distribution<> dist;
    std::generate(result.begin(), result.end(), [&] { return dist(engine); });
    return result;
}

rtg::argument generate_argument(rtg::shape s, std::mt19937::result_type seed = 0);

} // namespace rtg

#endif
