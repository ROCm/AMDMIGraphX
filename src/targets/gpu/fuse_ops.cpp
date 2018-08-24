#include <migraph/gpu/fuse_ops.hpp>
#include <migraph/iterator_for.hpp>
#include <migraph/gpu/device/add_relu.hpp>
#include <migraph/instruction.hpp>

namespace migraph {

namespace gpu {

struct hip_add_relu
{
    std::string name() const { return "hip::add_relu"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs}.has(3).standard();
        return inputs.front();
    }
    argument compute(context&, const shape&, const std::vector<argument>& args) const
    {
        device::add_relu(args.at(2), args.at(0), args.at(1));
        return args.at(2);
    }
};

void fuse_ops::apply(program& p) const
{
    for(auto ins : iterator_for(p))
    {
        if(ins->op.name() != "gpu::relu")
            continue;
        auto add_ins = ins->arguments.front();
        if(add_ins->op.name() != "gpu::add")
            continue;
        p.replace_instruction(ins, hip_add_relu{}, add_ins->arguments);
    }
}

} // namespace gpu

} // namespace migraph
