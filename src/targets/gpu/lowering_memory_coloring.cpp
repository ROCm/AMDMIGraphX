#include <migraph/gpu/lowering_memory_coloring.hpp>
#include <migraph/iterator_for.hpp>
#include <migraph/gpu/hip.hpp>
#include <migraph/instruction.hpp>
#include <migraph/pass_config.hpp>

namespace migraph {

namespace gpu {

struct gen_base_addr
{
    shape s;
    std::string name() const { return "gen_base_addr"; }
    shape compute_shape(const std::vector<shape>&) const { return s; }
    argument compute(const context& ctx, const shape&, const std::vector<argument>&) const
    {
        return ctx.scratch;
    }
};

void lowering_memory_coloring::apply(program& p) const
{
    if(enabled(MIGRAPH_DISABLE_MEMORY_COLORING{}))
        return;
    if(!enabled(MIGRAPH_UNIFY_MEMORY_COLORING{}))
        return;

    assert(ctx != nullptr);
    auto scratch_ins = p.get_parameter("scratch");
    if(scratch_ins == p.end())
        return;

    shape s_scratch   = scratch_ins->get_shape();
    argument base_ptr = allocate_gpu(s_scratch, false);
    ctx->scratch      = base_ptr;
    scratch_ins       = p.replace_instruction(scratch_ins, gen_base_addr{s_scratch});

    for(auto ins : iterator_for(p))
    {
        if(ins->get_operator().name() == "write_literal")
        {
            const std::vector<instruction_ref>& args = ins->inputs();
            instruction_ref arg0                     = args.at(0);
            instruction_ref arg1                     = args.at(1);
            auto&& a           = any_cast<op::write_literal>(ins->get_operator());
            std::size_t offset = a.offset;
            p.replace_instruction(ins, hip_memcpy{offset}, arg0, arg1);
        }
    }
    //    std::cout << p << std::endl;
}
} // namespace gpu
} // namespace migraph
