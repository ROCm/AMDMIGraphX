#ifndef MIGRAPH_GUARD_MIGRAPHLIB_REQUIRES_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_REQUIRES_HPP

#include <type_traits>

namespace migraph {

template <bool... Bs>
struct and_ : std::is_same<and_<Bs...>, and_<(Bs || true)...>> // NOLINT
{
};

template <bool B>
using bool_c = std::integral_constant<bool, B>;

#ifdef CPPCHECK
#define MIGRAPH_REQUIRES(...) class = void
#else
#define MIGRAPH_REQUIRES(...)                  \
    bool PrivateRequires##__LINE__ = true, \
         class = typename std::enable_if<and_<__VA_ARGS__, PrivateRequires##__LINE__>{}>::type
#endif

} // namespace migraph

#endif
