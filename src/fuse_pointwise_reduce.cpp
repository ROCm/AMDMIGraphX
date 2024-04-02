#include <migraphx/fuse_pointwise_reduce.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/fuse_pointwise.hpp>
#include <migraphx/fuse_reduce.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void fuse_pointwise_reduce::apply(module_pass_manager& mpm) const
{
    mpm.run_pass(fuse_pointwise{.enable_rewrite_reshapes=false});
    mpm.run_pass(fuse_reduce{.enable_rewrite_reshapes=false});
    mpm.run_pass(fuse_pointwise{.enable_rewrite_reshapes=true});
    mpm.run_pass(fuse_reduce{.enable_rewrite_reshapes=true});

}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

