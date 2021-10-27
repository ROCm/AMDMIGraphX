#ifndef MIGRAPHX_GUARD_RTGLIB_QUANTIZE_INT8_HPP
#define MIGRAPHX_GUARD_RTGLIB_QUANTIZE_INT8_HPP

#include <string>
#include <vector>
#include <functional>
#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;
struct module;

/**
 * capture inputs of operators to be quantized to int8
 */
struct capture_arguments_pass
{
    std::vector<std::string> ins_names = {"dot", "convolution"};
    std::function<void(std::size_t, std::vector<argument>)> f{};
    std::size_t* param_index = nullptr;
    std::string name() const { return "capture_arguments"; }
    void apply(module& m) const;
};

/**
 * quantize a program to int8
 */
struct quantize_int8_pass
{
    std::vector<std::string> ins_names = {"dot", "convolution"};
    std::vector<std::pair<float, float>> quant_params;
    std::string name() const { return "quantize_int8"; }
    void apply(module& m) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
