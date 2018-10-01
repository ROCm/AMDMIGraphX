#ifndef MIGRAPH_GUARD_RTGLIB_CONTIGUOUS_HPP
#define MIGRAPH_GUARD_RTGLIB_CONTIGUOUS_HPP

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
#include <utility>

namespace migraph {
namespace gpu {


struct miopen_contiguous
{
    contiguous op;
    std::string name() const { return "gpu::contiguous"; }
    shape compute_shape(const std::vector<shape>& inputs) const;
    argument compute(context&, shape output_shape, const std::vector<argument>& args) const;
};

} // namespace gpu

} // namespace migraph

#endif
