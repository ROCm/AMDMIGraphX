#include <vector>
#include <algorithm>
#include <cmath>

template <typename T>
std::vector<T> softmax(const std::vector<T>& p)
{
    size_t n = p.size();
    std::vector<T> result(n);
    std::transform(p.begin(), p.end(), result.begin(), [](auto x) { return std::exp(x); });
    T s = std::accumulate(result.begin(), result.end(), 0.0f, std::plus<T>());
    std::transform(result.begin(), result.end(), result.begin(), [=](auto x) { return x / s; });
    return result;
}
