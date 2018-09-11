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
    shape compute_shape(const std::vector<shape>&) const
    {
        return s;
    }
    argument compute(const context& ctx, const shape&, const std::vector<argument>&) const
    {
        return ctx.scratch;
    }
};

void lowering_memory_coloring::apply(program& p) const
{
    if (enabled(MIGRAPH_DISABLE_MEMORY_COLORING{}))
        return;

    assert(ctx != nullptr);
    auto scratch_ins = p.get_parameter("scratch");
    if (scratch_ins == p.end())
        return;

    bool can_resolve_addr = false;
    argument base_ptr;
    shape s_scratch = scratch_ins->result;

    if (ctx->params.find("scratch") == ctx->params.end()) {
        // scratch memory is not passed in, allocate memory here.
        can_resolve_addr = true;
        base_ptr = allocate_gpu(s_scratch, false);
        
    } else {
        argument a = ctx->params["scratch"];
        assert((a.get_shape().bytes() >= s_scratch.bytes()) && "insufficent scratch memory");
        if (!a.empty()) {
            // scratch memory is passed in and already has a known address.
            can_resolve_addr = true;
            base_ptr = a;
        }
    }
    if (can_resolve_addr) {
        ctx->scratch = base_ptr;
        scratch_ins = p.replace_instruction(scratch_ins, gen_base_addr{s_scratch});
    }
    
    for(auto ins : iterator_for(p))
        {
        if(ins->op.name() == "write_literal")
        {
            std::vector<instruction_ref>& args = ins->arguments;
            instruction_ref arg0 = args.at(0);
            instruction_ref arg1 = args.at(1);
            
            shape s_arg1 = arg1->get_shape();
            std::size_t size = s_arg1.bytes();
            auto&& a = any_cast<write_literal>(ins->op);
            std::size_t offset = a.offset;
            
            if (can_resolve_addr && a.pre_copy) {
                char* dst = base_ptr.data() + offset;
                const char* src = arg1->lit.data();
                copy_to_gpu(dst, src, size);
                gpu_sync();
                p.replace_instruction(ins, load{s_arg1, offset}, scratch_ins);
                p.remove_instruction(arg1);
            } else {
                p.replace_instruction(ins, hip_memcpy{offset}, arg0, arg1);
            }
        }
    }
    //    std::cout << p << std::endl;
}
} // namespace gpu
} // namespace migraph
