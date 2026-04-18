
#include <migraphx/op/builder/kit.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

struct torch_kit : kit<torch_kit>
{
    std::string prefix() const { return "tm::"; }
    void apply() const
    {
        this->common_ops({
            "ceil", "convert", "cos",   "cosh",        "div",     "dot",   "elu",     "equal",
            "erf",  "exp",     "floor", "fmod",        "greater", "isinf", "isnan",   "leaky_relu",
            "less", "log",     "log2",  "logical_and", "max",     "min",   "mul",     "mul",
            "neg",  "not",     "pow",   "recip",       "relu",    "rsqrt", "sigmoid", "sign",
            "sin",  "sinh",    "sqrt",  "sub",         "tan",     "tanh",
        });
        this->common_ops({"where"}, {.common_type = false});

        this->ops({
            "argmax",
            "argmin",
            "broadcast",
            "concat",
            "contiguous",
            "contiguous",
            "convolution",
            "deconvolution",
            "dequantizelinear",
            "gather",
            "gathernd",
            "get_tuple_elem",
            "multibroadcast",
            "multibroadcast",
            "pad",
            "pooling",
            "prefix_scan_sum",
            "quantizelinear",
            "reduce_all",
            "reduce_any",
            "reduce_max",
            "reduce_mean",
            "reduce_min",
            "reduce_prod",
            "reduce_sum",
            "reshape",
            "scatter_none",
            "scatter_none",
            "slice",
            "softmax",
            "squeeze",
            "step",
            "topk",
            "transpose",
            "undefined",
            "unsqueeze",
        });

        // TODO: Make a wrapper for rnn ops
        this->ops({
            "lstm",
            "rnn_last_cell_output",
            "rnn_last_hs_output",
        });
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
