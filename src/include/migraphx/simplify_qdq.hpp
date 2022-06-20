#ifndef MIGRAPHX_GUARD_RTGLIB_SIMPLIFY_QDQ_HPP
#define MIGRAPHX_GUARD_RTGLIB_SIMPLIFY_QDQ_HPP

#include <string>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

/**
 * Inserts quantized operators in place of dq->quantizable_op->q
 * then removes remaining fake quantization (q->dq pairs)
 */
struct simplify_qdq
{
    std::string name() const { return "simplify_qdq"; }
    void apply(module& m) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
