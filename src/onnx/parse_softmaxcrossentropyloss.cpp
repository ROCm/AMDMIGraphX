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
    std::vector<op_desc> operators() const { return {{"SoftmaxCrossEntropyLoss"}}; }

    // Handle ignore index if it's within range of allowable classes
    // return false if ignore index out of bounds and never add literal to graph
    // return true if ignore index is in bound and pass back literal
    bool normalize_input_index(const onnx_parser& parser,
                               const onnx_parser::node_info& info,
                               const int64_t input_classes,
                               instruction_ref& ignore_index) const
    {
        bool has_ignore_index = contains(info.attributes, "ignore_index");
        if(has_ignore_index)
        {
            auto ignore_index_val =
                parser.parse_value(info.attributes.at("ignore_index")).at<int64_t>();

            if(ignore_index_val >= input_classes || abs(ignore_index_val) > input_classes)
            {
                return false;
            }

            if(ignore_index_val < 0)
            {
                ignore_index_val = input_classes + ignore_index_val;
            }

            ignore_index = info.add_literal(migraphx::literal(
                migraphx::shape(migraphx::shape::int64_type, {1}, {0}), {ignore_index_val}));

            return true;
        }
        return false;
    }

    std::string get_reduction_param(const onnx_parser::node_info &info) const
    {
        std::string reduction = "mean";
        if(contains(info.attributes, "reduction"))
        {
            reduction = info.attributes.at("reduction").s();
            if(not contains({"mean", "sum", "none"}, reduction))
            {
                MIGRAPHX_THROW("Invalid reduction mode: " + reduction +
                               "\n Valid options are [none, mean, sum]");
            }
        }
        return reduction;
    }

    instruction_ref get_scores(const instruction_ref &arg) const
    {
        auto scores = arg;
        auto scores_shape = scores->get_shape();
        if(scores_shape.ndim() < 2)
        {
            MIGRAPHX_THROW("softmaxcrossentropyloss: Scores must be two or more dimensions [batch, "
                           "class_size, D1...Dk]");
        }

        if(migraphx::shape::is_integral(scores_shape.type()))
        {
            MIGRAPHX_THROW(
                "softmaxcrossentropyloss: Score must be either half, float, or double type");
        }
        return scores;
    }

    instruction_ref get_labels(const instruction_ref &arg, shape &scores_shape) const
    {
        auto labels = arg;
        auto label_shape = labels->get_shape();

        if(scores_shape.lens()[0] != label_shape.lens()[0])
        {
            MIGRAPHX_THROW(
                "softmaxcrossentropyloss: Score and Labels must identical batch size inputs");
        }

        if((scores_shape.ndim() - 1) != label_shape.ndim())
        {
            MIGRAPHX_THROW(
                "softmaxcrossentropyloss: Score and Labels must contain identical K-Dimensions");
        }   
        return labels;
    }

    instruction_ref get_weights(const onnx_parser::node_info &info, const std::vector<instruction_ref> &args, const shape &scores_shape, size_t class_size, bool &has_ignore_index, instruction_ref ignore_index) const 
    {
        // Default weights will always be 1
        auto weights = info.add_literal(
            migraphx::literal(migraphx::shape(scores_shape.type(), {1}, {0}), {1}));
        weights = info.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {class_size}}}), weights);

        bool has_weights = (args.size() > 2);
        // Get optional input weights (Used for mean reduction)
        if(has_weights)
        {
            weights = args.at(2);
            auto weights_shape = weights->get_shape();

            if(weights_shape.lens()[0] != scores_shape.lens()[1])
            {
                MIGRAPHX_THROW("softmaxcrossentropyloss: Invalid weight vector shape. Weight must "
                               "contain weight for each class");
            }

            if(migraphx::shape::is_integral(weights_shape.type()))
            {
                MIGRAPHX_THROW(
                    "softmaxcrossentropyloss: weight must be either half, float, or double type");
            }

            if(weights_shape.type() != scores_shape.type())
            {
                MIGRAPHX_THROW(
                    "softmaxcrossentropyloss: Weight and Scores inputs must be the same type");
            }
        }

        // adjust weights based on ignore index if that's set to reduce output after mul to zero
        // Saves us from doing a where() here and just scale at the end
        if(has_ignore_index)
        {
            auto weights_shape = weights->get_shape();
            std::vector<float> zero_val_vect(weights_shape.elements(), 0);
            auto zero_val = info.add_literal(migraphx::literal(weights_shape, zero_val_vect));
            weights       = info.add_instruction(
                migraphx::make_op("scatter_none", {{"axis", 0}}), weights, ignore_index, zero_val);
        }

        return weights;
    }

    std::vector<instruction_ref> parse(const op_desc& /*opd */,
                                       const onnx_parser& parser,
                                       const onnx_parser::node_info& info,
                                       const std::vector<instruction_ref>& args) const
    {
        // Get and handle attributes
        auto reduction = get_reduction_param(info);

        // Get and validate Inputs
        auto scores = get_scores(args.at(0));
        auto scores_shape = scores->get_shape();
        auto labels  = get_labels(args.at(1), scores_shape);
        auto label_shape = labels->get_shape();

        // meta parameters based on input scores shape
        bool is_k_dim  = (scores_shape.ndim() > 3);
        size_t batch_size = scores_shape.lens().at(0);
        size_t class_size = scores_shape.lens().at(1);

        // ignore_index is optional attribute, assign this as a scalar literal input to the op
        instruction_ref ignore_index;
        auto has_ignore_index =
            normalize_input_index(parser, info, static_cast<int64_t>(class_size), ignore_index);

        bool has_weights = (args.size() > 2);
        auto weights = get_weights(info, args, scores_shape, class_size, has_ignore_index, ignore_index);

        auto loss_tensor = info.add_instruction(
            migraphx::make_op("convert", {{"target_type", scores_shape.type()}}), labels);

        if(is_k_dim)
        {
            auto lens = label_shape.lens();
            size_t d =
                std::accumulate(lens.rbegin(), lens.rend() - 1, 1, std::multiplies<size_t>());
            scores = info.add_instruction(
                migraphx::make_op("reshape", {{"dims", {batch_size, class_size, d}}}), scores);
        }

        // Offload calculation of log(Softmax(scores)) for the input before we perform cross entropy
        // loss calculation
        auto softmax_scores =
            info.add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), scores);
        auto log_sm_scores  = info.add_instruction(migraphx::make_op("log"), softmax_scores);
        auto neg_lsm_scores = info.add_instruction(migraphx::make_op("neg"), log_sm_scores);

        loss_tensor = info.add_instruction(
            migraphx::make_op("scatter_none", {{"axis", 0}}), loss_tensor, labels, neg_lsm_scores);

        if(is_k_dim)
        {
            loss_tensor = info.add_instruction(
                migraphx::make_op("reshape", {{"dims", label_shape.lens()}}), loss_tensor);
        }

        instruction_ref weight_tensor;
        if(has_ignore_index or has_weights)
        {
            weight_tensor =
                info.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), weights, labels);

            // Mul op also handles ignored index as the corresponding weight will be zero if it
            // matches ignore index
            loss_tensor =
                info.add_instruction(migraphx::make_op("mul"), loss_tensor, weight_tensor);
        }

        // Add reduction step after we're generated crossentropyloss tensor and rearragned weight
        // scaling tensor
        if(reduction == "mean" and has_weights)
        {
            loss_tensor =
                info.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), loss_tensor);
            auto reduced_weights = info.add_instruction(
                migraphx::make_op("reduce_sum", {{"axes", {0}}}), weight_tensor);
            loss_tensor =
                info.add_instruction(migraphx::make_op("div"), loss_tensor, reduced_weights);
        }
        else if(reduction == "mean" and not has_weights)
        {
            loss_tensor = info.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {0}}}),
                                               loss_tensor);
        }
        else if(reduction == "sum")
        {
            loss_tensor =
                info.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), loss_tensor);
        }

        return {loss_tensor, log_sm_scores};
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
