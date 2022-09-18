#ifndef MIGRAPHX_GUARD_TEST_GPU_MAKE_PRECOMPILE_OP_HPP
#define MIGRAPHX_GUARD_TEST_GPU_MAKE_PRECOMPILE_OP_HPP

#include <migraphx/operation.hpp>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/make_op.hpp>

#define MIGRAPHX_GPU_TEST_PRECOMPILE(...) \
struct test_compiler : migraphx::gpu::compiler<test_compiler> \
{ \
    std::vector<std::string> names() const { return {__VA_ARGS__}; } \
 \
    template<class... Ts> \
    migraphx::operation compile_op(Ts&&...) const \
    { \
        MIGRAPHX_THROW("Not compilable"); \
    } \
 \
    template<class... Ts> \
    migraphx::gpu::compiler_replace compile(Ts&&...) const \
    { \
        MIGRAPHX_THROW("Not compilable"); \
    } \
};

inline migraphx::operation make_precompile_op(migraphx::rank<0>, const migraphx::operation& op)
{
    return migraphx::make_op("gpu::precompile_op", {{"op", migraphx::to_value(op)}});
}

inline migraphx::operation make_precompile_op(migraphx::rank<1>, const std::string& name)
{
    return make_precompile_op(migraphx::rank<0>{}, migraphx::make_op(name));
}

template <class T>
auto make_precompile_op(const T& x)
{
    return make_precompile_op(migraphx::rank<1>{}, x);
}

#endif // MIGRAPHX_GUARD_TEST_GPU_MAKE_PRECOMPILE_OP_HPP
