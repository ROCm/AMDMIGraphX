#ifndef MIGRAPHX_GUARD_TEST_GPU_MAKE_PREALLOCATE_OP_HPP
#define MIGRAPHX_GUARD_TEST_GPU_MAKE_PREALLOCATE_OP_HPP

#include <migraphx/operation.hpp>
#include <migraphx/make_op.hpp>

inline migraphx::operation make_preallocate_op(migraphx::rank<0>, const migraphx::operation& op)
{
    return migraphx::make_op("gpu::precompile_op", {{"op", migraphx::to_value(op)}});
}

inline migraphx::operation make_preallocate_op(migraphx::rank<1>, const std::string& name)
{
    return make_preallocate_op(migraphx::rank<0>{}, migraphx::make_op(name));
}

template <class T>
auto make_preallocate_op(const T& x)
{
    return make_preallocate_op(migraphx::rank<1>{}, x);
}

#endif // MIGRAPHX_GUARD_TEST_GPU_MAKE_PREALLOCATE_OP_HPP
