#include <migraphx/partition.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct partition_op
{
    std::string label;
    std::string name() const { return "partition"; }
    // TODO: Implement compute_shape and compute
};
MIGRAPHX_REGISTER_OP(partition_op);

void partition(program& p,
               module& root,
               const std::unordered_map<instruction_ref, std::string>& assignments)
{
    // Group instructions based on label
    
    // TODO: Recurse traverse submodule from root
    for(auto ins:iterator_for(root))
    {
        // Rough idea of inserting submodules
        std::string label;
        auto sm = p.create_module(label);
        root.add_instruction(partition_op{label}, {}, {sm});
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
