/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/*
-----------------------
SoftmaxCrossEntropyLoss
-----------------------
Loss function that measures the softmax cross entropy between 'scores' and 'labels'.
This operator first computes a loss tensor whose shape is identical to the labels input.
If the input is 2-D with shape (N, C), the loss tensor may be a N-element
vector L = (l_1, l_2, ..., l_N).  If the input is N-D tensor with
shape (N, C, D1, D2, ..., Dk), the loss tensor L may have (N, D1, D2, ..., Dk)
as its shape and L[i,][j_1][j_2]...[j_k] denotes a scalar element in L.
After L is available, this operator can optionally do a reduction operator.

shape(scores):  (N, C) where C is the number of classes, or (N, C, D1, D2,..., Dk),
                with K >= 1 in case of K-dimensional loss.
shape(labels): (N) where each value is 0 <= labels[i] <= C-1, or (N, D1, D2,..., Dk),
                with K >= 1 in case of K-dimensional loss.


The loss for one sample, l_i, can calculated as follows:

l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk], where i is the index of classes.
or

l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk] * weights[c], if 'weights' is provided.
loss is zero for the case when label-value equals ignore_index.

l[i][d1][d2]...[dk]  = 0, when labels[n][d1][d2]...[dk] = ignore_index
where:

p = Softmax(scores)
y = Log(p)
c = labels[i][d1][d2]...[dk]
Finally, L is optionally reduced:

If reduction = 'none', the output is L with shape (N, D1, D2, ..., Dk).
If reduction = 'sum', the output is scalar: Sum(L).
If reduction = 'mean', the output is scalar: ReduceMean(L), or
                if weight is provided: ReduceSum(L) / ReduceSum(W),
                where tensor W is of shape (N, D1, D2, ..., Dk) and W[n][d1][d2]...[dk] =
weights[labels[i][d1][d2]...[dk]].

Attributes
+++++++++++

ignore_index : int
                Specifies a target value that is ignored and
                does not contribute to the input gradient. It's an optional value.

reduction : string (default is mean)
                Type of reduction to apply to loss: none, sum, mean(default).
                - 'none': no reduction will be applied
                - 'sum': the output will be summed.
                - 'mean': the sum of the output will be divided by
                        the number of elements in the output.

Inputs (2 - 3)
++++++++++++++

scores (differentiable) : T
    The predicted outputs with shape [batch_size, class_size], or [batch_size, class_size, D1, D2 ,
..., Dk], where K is the number of dimensions.

labels (non-differentiable) : Tind
    The ground truth output tensor, with shape [batch_size], or [batch_size, D1, D2, ..., Dk],
    where K is the number of dimensions. Labels element value shall be in range of [0, C).
    If ignore_index is specified, it may have a value outside [0, C) and the label values should
    either be in the range [0, C) or have the value ignore_index.

weights (optional, non-differentiable) : T
    A manual rescaling weight given to each class. If given,
    it has to be a 1D Tensor assigning weight to each of the classes.
    Otherwise, it is treated as if having all ones.

Outputs (1 - 2)
==================

output (differentiable) : T
    Weighted loss float Tensor. If reduction is 'none', this has the shape of [batch_size],
    or [batch_size, D1, D2, ..., Dk] in case of K-dimensional loss. Otherwise, it is a scalar.

log_prob (optional, differentiable) : T
    Log probability tensor. If the output of softmax is prob, its value is log(prob).

Type Constraints
===================
    T : tensor(float16), tensor(float), tensor(double), tensor(bfloat16) (Currently not supported in
MIGX) Constrain input and output types to float tensors.

Tind : tensor(int32), tensor(int64)
    Constrain target to integer types
*/

#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/instruction.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_softmaxcrossentropyloss : op_parser<parse_softmaxcrossentropyloss>
{
    std::vector<op_desc> operators() const
    {
        return {{"SoftmaxCrossEntropyLoss", "crossentropyloss"}};
    }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& parser,
                          const onnx_parser::node_info& info,
                          const std::vector<instruction_ref>& args) const
    {
        // default in Onnx spec is mean setting
        std::string reduction = "mean";
        if(contains(info.attributes, "reduction"))
        {
            std::set<std::string> supported_modes = {"mean", "none", "sum"};

            reduction = parser.parse_value(info.attributes.at("reduction")).at<std::string>();
            if(not contains(supported_modes, reduction))
            {
                MIGRAPHX_THROW("Invalid reduction mode: " + reduction +
                               "\n Valid options are [none, mean, sum]");
            }
        }

        // ignore_index is optional attribute, assign this as a scalar literal input to the op
        auto ignore_index =
            info.add_literal(migraphx::literal(migraphx::shape(shape::int32_type, {1}, {0}), {-1}));
        bool has_ignore_index = contains(info.attributes, "ignore_index");
        if(has_ignore_index)
        {
            auto ignore_index_val =
                parser.parse_value(info.attributes.at("ignore_index")).at<int>();
            ignore_index = info.add_literal(migraphx::literal(
                migraphx::shape(migraphx::shape::int32_type, {1}, {0}), {ignore_index_val}));
        }

        // Get Inputs
        auto scores = args.at(0);
        auto labels = args.at(1);

        // Get optional input weights (Used for mean reduction)
        instruction_ref weights;
        bool has_weights = (args.size() > 2);
        if(has_weights)
        {
            weights = args.at(2);
        }
        else
        { // if weights isn't given, treat input as equal scaling for each class labels
            std::vector<float> ones_vec(scores->get_shape().elements(), 1);
            weights = info.add_literal(migraphx::literal(scores->get_shape(), ones_vec));
        }

        // Offload calculation of log(Softmax(scores)) for the input before we perform cross entropy
        // loss calculation
        auto softmax_scores = info.add_instruction(migraphx::make_op("softmax"), scores);
        auto log_sm_scores  = info.add_instruction(migraphx::make_op("log"), softmax_scores);

        // Returns tuple of two outputs (loss_tensor, weights)
        auto tuple_loss_tensor =
            info.add_instruction(make_op(opd.op_name,
                                         {{"has_ignore_index", has_ignore_index},
                                          {"weighted", has_weights},
                                          {"mode", reduction}}),
                                 log_sm_scores,
                                 labels,
                                 weights,
                                 ignore_index);

        auto loss_tensor =
            info.add_instruction(make_op("get_tuple_elem", {{"index", 0}}), tuple_loss_tensor);
        auto weight_tensor =
            info.add_instruction(make_op("get_tuple_elem", {{"index", 1}}), tuple_loss_tensor);

        if(reduction == "none" and (loss_tensor->get_shape().lens() != scores->get_shape().lens()))
        {
            MIGRAPHX_THROW("Invalid loss tensor shape");
        }

        // Add reduction step after we're generated crossentropyloss tensor and rearragned weight
        // scaling tensor
        if(reduction == "mean" and not has_weights)
        {
            loss_tensor = info.add_instruction(make_op("reduce_mean"), loss_tensor);
        }
        else if(reduction == "sum" or has_weights)
        {
            loss_tensor = info.add_instruction(make_op("reduce_sum"), loss_tensor);
            if(reduction == "mean")
            {
                auto reduced_weights = info.add_instruction(make_op("reduce_sum"), weight_tensor);
                loss_tensor = info.add_instruction(make_op("div"), loss_tensor, reduced_weights);
            }
        }

        return loss_tensor;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
