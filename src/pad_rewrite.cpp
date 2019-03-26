#include <migraphx/pad_rewrite.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void pad_rewrite::apply(program& p) const
{
    for(auto ins : iterator_for(p))
    {
        if(ins->name() != "pad")
            continue;
        for (auto output : ins->outputs())
        {
            auto op_name = output->name();
            if(op_name == "convolution")
                update_op(op::convolution{}, ins, output, p);
            else if(op_name == "im2col")
                update_op(op::im2col{}, ins, output, p);
            else if(op_name == "pooling")
                update_op(op::pooling{}, ins, output, p);
        }
    }
}

template<class T>
void pad_rewrite::update_op(T, instruction_ref ins, instruction_ref output, program& p) const
{
    auto pad_op = any_cast<op::pad>(ins->get_operator());
    if(!pad_op.symmetric())
        return;
    
    std::vector<int64_t> pads = pad_op.pads;
    assert(pads.size() == 8); // ensure input being padded has 4 dims (*2 for font and back padding)
    std::array<size_t, 2> new_pads{static_cast<size_t>(pads[2]),static_cast<size_t>(pads[3])};

    T op = any_cast<T>(output->get_operator()); 
    op.padding = new_pads;
    
    std::vector<instruction_ref> new_inputs{output->inputs()};
    new_inputs.front() = ins->inputs().front();
    
    p.replace_instruction(output, op, new_inputs);
} 

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
