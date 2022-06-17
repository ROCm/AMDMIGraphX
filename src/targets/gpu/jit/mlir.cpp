#include <migraphx/gpu/compiler.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/gpu/context.hpp>

#include <migraphx/gpu/mlir.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct mlir_compiler : compiler<mlir_compiler>
{
    std::vector<std::string> names() const { return {"gpu::mlir_conv"}; }

    operation compile_op(context&, const std::vector<shape>&, const value&) const { return {}; }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation&) const
    {
        auto* smod = ins->module_inputs().front();
        assert(smod->get_parameter_names().size() == ins->inputs().size() - 1);
        return insert(compile_mlir(ctx, *smod));
    }

    compiler_replace insert(code_object_op co) const
    {
        return [co=std::move(co)](module& m, instruction_ref ins) {
            auto mlir = insert_mlir(m, ins, co, ins->inputs());
            m.replace_instruction(ins, mlir);
        };
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
