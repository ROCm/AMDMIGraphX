#ifndef MIGRAPH_GUARD_RTGLIB_BATCHNORM_HPP
#define MIGRAPH_GUARD_RTGLIB_BATCHNORM_HPP

#include <migraph/gpu/lowering.hpp>
#include <migraph/manage_ptr.hpp>
#include <migraph/instruction.hpp>
#include <migraph/operators.hpp>
#include <migraph/generate.hpp>
#include <migraph/shape_for_each.hpp>
#include <migraph/gpu/miopen.hpp>
#include <migraph/gpu/hip.hpp>
#include <migraph/dfor.hpp>
#include <migraph/gpu/device/contiguous.hpp>
#include <migraph/gpu/device/add.hpp>
#include <migraph/iterator_for.hpp>
#include <migraph/gpu/rocblas.hpp>
#include <migraph/gpu/context.hpp>
#include <migraph/config.hpp>
#include <utility>

namespace migraph { inline namespace MIGRAPH_INLINE_NS {
namespace gpu {

struct miopen_batch_norm_inference
{
    op::batch_norm_inference op;
    std::string name() const { return "gpu::batch_norm_inference"; }
    shape compute_shape(const std::vector<shape>& inputs) const;
    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const;
    int output_alias(const std::vector<shape>& shapes) const { return shapes.size() - 1; }
};

} // namespace gpu
} // inline namespace MIGRAPH_INLINE_NS
} // namespace migraph

#endif
