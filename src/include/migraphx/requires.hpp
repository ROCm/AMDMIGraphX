#ifndef MIGRAPH_GUARD_MIGRAPHLIB_REQUIRES_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_REQUIRES_HPP

#include <type_traits>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {

template <bool... Bs>
struct and_ : std::is_same<and_<Bs...>, and_<(Bs || true)...>> // NOLINT
{
};

template <bool B>
using bool_c = std::integral_constant<bool, B>;

template <int N>
struct requires_enum
{
    enum e
    {
        a = 0
    };
};

#define MIGRAPH_REQUIRES_CAT(x, y) x##y

#ifdef CPPCHECK
#define MIGRAPH_REQUIRES(...) class = void
#else
#if 0
// TODO: This currently crashed on clang
#define MIGRAPH_REQUIRES(...)                                                                       \
    typename migraphx::requires_enum<__LINE__>::e MIGRAPH_REQUIRES_CAT(                             \
        PrivateRequires,                                                                            \
        __LINE__) = migraphx::requires_enum<__LINE__>::a,                                           \
        class     = typename std::enable_if<and_<__VA_ARGS__,                                       \
                                             MIGRAPH_REQUIRES_CAT(PrivateRequires, __LINE__) == \
                                                 migraphx::requires_enum<__LINE__>::a>{}>::type
#else
#define MIGRAPH_REQUIRES(...)                                              \
    typename migraphx::requires_enum<__LINE__>::e MIGRAPH_REQUIRES_CAT(    \
        PrivateRequires, __LINE__) = migraphx::requires_enum<__LINE__>::a, \
                         class     = typename std::enable_if<and_<__VA_ARGS__>{}>::type
#endif
#endif

} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx

#endif
