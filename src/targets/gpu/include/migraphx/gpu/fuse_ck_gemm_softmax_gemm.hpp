#ifndef MIGRAPHX_GUARD_GPU_FUSE_CK_GEMM_SOFTMAX_GEMM_HPP
#define MIGRAPHX_GUARD_GPU_FUSE_CK_GEMM_SOFTMAX_GEMM_HPP

#include <migraphx/config.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module_pass_manager;

namespace gpu {

struct fuse_ck_gemm_softmax_gemm
{
    context* ctx = nullptr;
    std::string name() const { return "gpu::fuse_ck_gemm_softmax_gemm"; }
    void apply(module_pass_manager& mpm) const;
};

} // namespace gpu

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_FUSE_CK_HPP
