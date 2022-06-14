#ifndef MIGRAPHX_GUARD_RTGLIB_QUANTIZE_FP16_HPP
#define MIGRAPHX_GUARD_RTGLIB_QUANTIZE_FP16_HPP

#include <string>
#include <vector>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;
struct module;

/**
 * quantize a program to fp16
 */
struct quantize_fp16_pass
{
    std::vector<std::string> ins_names = {"all"};
    std::string name() const { return "quantize_fp16"; }
    void apply(module& m) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
