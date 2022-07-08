#include <migraphx/partition.hpp>
#include <migraphx/program.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct partition_op
{
    std::string label;
    std::string name() const { return "partition"; }
    // TODO: Implement compute_shape and compute
};

void partition(program& p, module& root, const std::unordered_map<instruction_ref, std::string>& assignments)
{

}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
