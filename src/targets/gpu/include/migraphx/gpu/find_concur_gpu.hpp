#ifndef MIGRAPHX_GUARD_RTGLIB_FIND_CONCUR_GPU_HPP
#define MIGRAPHX_GUARD_RTGLIB_FIND_CONCUR_GPU_HPP

#include <migraphx/dom_info.hpp>
#include <migraphx/common_header.hpp>
#include <migraphx/gpu/event.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct find_concur_gpu
{
    void get_concur(program* p,
                    int num_of_streams,
                    std::unordered_map<const instruction*,
                                       std::vector<std::vector<const instruction*>>>& concur_instrs,
                    std::unordered_map<const instruction*, int>& instr2_points)
    {
        dom_info info(p);
        info.compute_dom(true);
        info.propagate_splits(num_of_streams, concur_instrs, instr2_points);
    }
};
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
