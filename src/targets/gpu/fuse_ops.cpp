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
        check_shapes{inputs, *this}.has(3);
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
        if(ins->name() != "gpu::relu")
            continue;
        auto add_ins = ins->inputs().front();
        if(add_ins->name() != "gpu::add")
            continue;
        auto args = add_ins->inputs();
        // Use the allocation from the relu operator
        args.back() = ins->inputs().back();
        p.replace_instruction(ins, hip_add_relu{}, args);
    }
}

} // namespace gpu

} // namespace migraph
