#ifndef MIGRAPHX_GUARD_RTGLIB_GPU_MLIR_HPP
#define MIGRAPHX_GUARD_RTGLIB_GPU_MLIR_HPP

#include <string>
#include <vector>
#include <migraphx/config.hpp>
#include <migraphx/gpu/code_object_op.hpp>
#include <migraphx/instruction_ref.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
struct module;
namespace gpu {

std::string dump_mlir(const module& m);
code_object_op compile_mlir(const context& ctx, const module& m);

instruction_ref insert_mlir(module& m,
                            instruction_ref ins,
                            code_object_op co,
                            const std::vector<instruction_ref>& inputs);                            


} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
