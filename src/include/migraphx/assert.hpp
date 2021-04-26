#ifndef MIGRAPHX_GUARD_MIGRAPHX_ASSERT_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_ASSERT_HPP

#include <migraphx/config.hpp>
#include <cstdlib>
#include <iostream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class F>
auto abort_on_throw(F f) -> decltype(f())
{
    try
    {
        return f();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        std::abort();
    }
    catch(...)
    {
        std::cerr << "Unknown exception" << std::endl;
        std::abort();
    }
}
#ifdef NDEBUG
#define MIGRAPHX_ASSERT_NO_THROW(...) __VA_ARGS__
#else
#define MIGRAPHX_ASSERT_NO_THROW(...) \
    migraphx::abort_on_throw([&]() -> decltype(__VA_ARGS__) { return __VA_ARGS__; })
#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_ASSERT_HPP
