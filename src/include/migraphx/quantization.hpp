#ifndef MIGRAPHX_GUARD_RTGLIB_QUANTIZATION_HPP
#define MIGRAPHX_GUARD_RTGLIB_QUANTIZATION_HPP

#include <string>
#include <vector>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/config.hpp>
#include <migraphx/target.hpp>
#include <migraphx/program.hpp>
#include <migraphx/env.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_INT8_QUANTIZATION_PARAMS)

void quantize_fp16(program& prog, const std::vector<std::string>& ins_names);
void quantize_fp16(program& prog);

// insert the capture operator for the inputs of each operator to be quantized
// to int8
std::size_t capture_arguments(program& prog,
                              const std::vector<std::string>& ins_names,
                              const std::function<void(std::size_t, std::vector<argument>)>& func);

std::shared_ptr<std::vector<std::pair<float, float>>>
capture_arguments_impl(program& prog, const target& t, const std::vector<std::string>& ins_names);

template <class T>
std::shared_ptr<std::vector<std::pair<float, float>>> capture_arguments(
    program& prog, T&& t, const std::vector<std::string>& ins_names = {"dot", "convolution"})
{
    static_assert(std::is_same<std::remove_cv_t<std::remove_reference_t<T>>, target>{} &&
                      std::is_lvalue_reference<T>{},
                  "Dangling reference to target!");
    return capture_arguments_impl(prog, t, ins_names);
}

void quantize_int8(program& prog,
                   const target& t,
                   std::vector<program::parameter_map>& calibration_args,
                   const std::vector<std::string>& ins_names = {"dot", "convolution"});
void quantize_int8(program& prog,
                   const std::vector<std::pair<float, float>>& quant_params,
                   const std::vector<std::string>& ins_names);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
