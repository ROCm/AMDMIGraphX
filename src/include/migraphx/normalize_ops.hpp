#ifndef MIGRAPHX_GUARD_RTGLIB_NORMALIZE_OPS_HPP
#define MIGRAPHX_GUARD_RTGLIB_NORMALIZE_OPS_HPP

#include <string>
#include <vector>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/program.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;

/**
 * Process negative axis attributes of ops
 */
struct normalize_ops
{
    std::string name() const { return "normalize_ops"; }
    void apply(program& p) const;

    private:
    bool tune_axis(value& val, int64_t n_dim) const;
    bool tune_slice_inputs(std::vector<int64_t>& axes,
                           std::vector<int64_t>& starts,
                           std::vector<int64_t>& ends,
                           const std::vector<std::size_t>& lens) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
