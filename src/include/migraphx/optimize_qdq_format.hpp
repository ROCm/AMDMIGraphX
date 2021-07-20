#ifndef MIGRAPHX_GUARD_RTGLIB_OPTIMIZE_QDQ_FORMAT_HPP
#define MIGRAPHX_GUARD_RTGLIB_OPTIMIZE_QDQ_FORMAT_HPP

#include <string>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

/**
 * Inserts quantized operators in place of dq->quantizable_op->q
 * then removes remaining fake quantization (q->dq pairs)
 */
struct optimize_qdq_format
{
    std::string name() const { return "optimize_qdq_format"; }
    void apply(module& m) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
