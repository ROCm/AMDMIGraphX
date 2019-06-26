#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_REQUIRES_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_REQUIRES_HPP

#include <type_traits>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <bool... Bs>
struct and_ : std::is_same<and_<Bs...>, and_<(Bs || true)...>> // NOLINT
{
};

template <bool B>
using bool_c = std::integral_constant<bool, B>;

#define MIGRAPHX_REQUIRES_PRIMITIVE_CAT(x, y) x##y
#define MIGRAPHX_REQUIRES_CAT(x, y) MIGRAPHX_REQUIRES_PRIMITIVE_CAT(x, y)

#define MIGRAPHX_REQUIRES_VAR() MIGRAPHX_REQUIRES_CAT(PrivateRequires, __LINE__)

#ifdef CPPCHECK
#define MIGRAPHX_REQUIRES(...) class = void
#else
#define MIGRAPHX_REQUIRES(...)                                                                 \
    bool MIGRAPHX_REQUIRES_VAR()            = true,                                            \
         typename std::enable_if<(MIGRAPHX_REQUIRES_VAR() && (migraphx::and_<__VA_ARGS__>{})), \
                                 int>::type = 0
#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
