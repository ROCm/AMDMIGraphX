#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_ONNX_PADDING_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_ONNX_PADDING_HPP

#include <migraphx/config.hpp>
#include <migraphx/onnx/onnx_parser.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

bool is_asym_padding(const std::vector<int64_t>& padding);

void cal_auto_padding_size(onnx_parser::node_info info,
                           value& v,
                           const std::vector<std::size_t>& k_lens,
                           const std::vector<std::size_t>& dilation,
                           const std::vector<std::size_t>& in_lens,
                           std::vector<int64_t>& paddings);

void check_padding_mode(const onnx_parser::node_info& info, const std::string& op_name);

void tune_padding_size(const value& v,
                       std::vector<int64_t>& padding,
                       int count_include_pad,
                       std::vector<int64_t>& s_start);

void check_asym_padding(const onnx_parser::node_info& info,
                        instruction_ref& ins,
                        const std::vector<int64_t>& padding,
                        value& v,
                        int count_include_pad = 0,
                        float pad_val         = 0);

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
