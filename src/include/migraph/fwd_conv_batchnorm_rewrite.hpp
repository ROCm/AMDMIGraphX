#ifndef MIGRAPH_GUARD_RTGLIB_FWD_CONV_BATCHNORM_REWRITE_HPP
#define MIGRAPH_GUARD_RTGLIB_FWD_CONV_BATCHNORM_REWRITE_HPP

#include <string>
#include <migraph/instruction_ref.hpp>
#include <migraph/config.hpp>

namespace migraph {
inline namespace MIGRAPH_INLINE_NS {

struct program;

struct fwd_conv_batchnorm_rewrite
{
    std::string name() const { return "fwd_conv_batchnorm_rewrite"; }
    void apply(program& p) const;
};

} // namespace MIGRAPH_INLINE_NS
} // namespace migraph

#endif
