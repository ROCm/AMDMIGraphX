#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <onnx.pb.h>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <functional>
#include <array>
#include <utility>
#include <vector>

#include <migraphx/fallthrough.hpp>
#include <migraphx/program.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/config.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/pad_calc.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct onnx_parser
{
    using attribute_map = std::unordered_map<std::string, onnx::AttributeProto>;
    struct node_info
    {
        attribute_map attributes{};
        std::size_t num_outputs = 1;
    };
    using node_map = std::unordered_map<std::string, onnx::NodeProto>;
    using op_func =
        std::function<std::vector<instruction_ref>(node_info, std::vector<instruction_ref>)>;
    node_map nodes;
    std::unordered_map<std::string, instruction_ref> instructions;
    program prog                = program();
    bool is_pytorch             = false;
    unsigned int batch_size     = 1;
    bool skip_unknown_operators = false;

    std::unordered_map<std::string, op_func> ops;
    std::unordered_map<std::string, operation> map_actv_funcs;

    onnx_parser()
    {
        // sort onnx operator alphabetically through name
        add_generic_op("Abs", op::abs{});
        add_generic_op("Acos", op::acos{});
        add_generic_op("Acosh", op::acosh{});
        add_generic_op("Asin", op::asin{});
        add_generic_op("Asinh", op::asinh{});
        add_generic_op("Atan", op::atan{});
        add_generic_op("Atanh", op::atanh{});
        add_generic_op("Ceil", op::ceil{});
        add_generic_op("Cos", op::cos{});
        add_generic_op("Cosh", op::cosh{});
        add_generic_op("Erf", op::erf{});
        add_generic_op("Exp", op::exp{});
        add_generic_op("Dropout", op::identity{});
        add_generic_op("Log", op::log{});
        add_generic_op("Floor", op::floor{});
        add_generic_op("Identity", op::identity{});
        add_generic_op("Relu", op::relu{});
        add_generic_op("Round", op::round{});
        add_generic_op("Sigmoid", op::sigmoid{});
        add_generic_op("Sign", op::sign{});
        add_generic_op("Sin", op::sin{});
        add_generic_op("Sinh", op::sinh{});
        add_generic_op("Sqrt", op::sqrt{});
        add_generic_op("Tan", op::tan{});
        add_generic_op("Tanh", op::tanh{});

        add_binary_op("Add", op::add{});
        add_binary_op("Div", op::div{});
        add_binary_op("Mul", op::mul{});
        add_binary_op("Pow", op::pow{});
        add_binary_op("PRelu", op::prelu{});
        add_binary_op("Sub", op::sub{});

        add_variadic_op("Sum", op::add{});
        add_variadic_op("Max", op::max{});
        add_variadic_op("Min", op::min{});

        add_mem_op("AveragePool", &onnx_parser::parse_pooling);
        add_mem_op("ArgMax", &onnx_parser::parse_arg_op<op::argmax>);
        add_mem_op("ArgMin", &onnx_parser::parse_arg_op<op::argmin>);
        add_mem_op("BatchNormalization", &onnx_parser::parse_batchnorm);
        add_mem_op("Cast", &onnx_parser::parse_cast);
        add_mem_op("Clip", &onnx_parser::parse_clip);
        add_mem_op("Concat", &onnx_parser::parse_concat);
        add_mem_op("Constant", &onnx_parser::parse_constant);
        add_mem_op("ConstantFill", &onnx_parser::parse_constant_fill);
        add_mem_op("ConstantOfShape", &onnx_parser::parse_constant_of_shape);
        add_mem_op("Conv", &onnx_parser::parse_conv<op::convolution>);
        add_mem_op("ConvInteger", &onnx_parser::parse_conv<op::quant_convolution>);
        add_mem_op("ConvTranspose", &onnx_parser::parse_conv_transpose);
        add_mem_op("Elu", &onnx_parser::parse_elu);
        add_mem_op("Expand", &onnx_parser::parse_expand);
        add_mem_op("Flatten", &onnx_parser::parse_flatten);
        add_mem_op("Gather", &onnx_parser::parse_gather);
        add_mem_op("Gemm", &onnx_parser::parse_gemm);
        add_mem_op("GlobalAveragePool", &onnx_parser::parse_pooling);
        add_mem_op("GlobalMaxPool", &onnx_parser::parse_pooling);
        add_mem_op("GRU", &onnx_parser::parse_gru);
        add_mem_op("ImageScaler", &onnx_parser::parse_imagescaler);
        add_mem_op("InstanceNormalization", &onnx_parser::parse_instancenorm);
        add_mem_op("LeakyRelu", &onnx_parser::parse_leaky_relu);
        add_mem_op("LogSoftmax", &onnx_parser::parse_softmax<op::logsoftmax>);
        add_mem_op("LRN", &onnx_parser::parse_lrn);
        add_mem_op("MatMul", &onnx_parser::parse_matmul<op::dot>);
        add_mem_op("MatMulInteger", &onnx_parser::parse_matmul<op::quant_dot>);
        add_mem_op("MaxPool", &onnx_parser::parse_pooling);
        add_mem_op("ReduceL1", &onnx_parser::parse_reduce_l1);
        add_mem_op("ReduceL2", &onnx_parser::parse_reduce_l2);
        add_mem_op("ReduceLogSum", &onnx_parser::parse_reduce_log_sum);
        add_mem_op("ReduceLogSumExp", &onnx_parser::parse_reduce_log_sum_exp);
        add_mem_op("ReduceMax", &onnx_parser::parse_reduce_oper<op::reduce_max>);
        add_mem_op("ReduceMean", &onnx_parser::parse_reduce_oper<op::reduce_mean>);
        add_mem_op("ReduceMin", &onnx_parser::parse_reduce_oper<op::reduce_min>);
        add_mem_op("ReduceProd", &onnx_parser::parse_reduce_oper<op::reduce_prod>);
        add_mem_op("ReduceSum", &onnx_parser::parse_reduce_oper<op::reduce_sum>);
        add_mem_op("ReduceSumSquare", &onnx_parser::parse_reduce_sum_square);
        add_mem_op("Reshape", &onnx_parser::parse_reshape);
        add_mem_op("RNN", &onnx_parser::parse_rnn);
        add_mem_op("Pad", &onnx_parser::parse_pad);
        add_mem_op("Shape", &onnx_parser::parse_shape);
        add_mem_op("Slice", &onnx_parser::parse_slice);
        add_mem_op("Softmax", &onnx_parser::parse_softmax<op::softmax>);
        add_mem_op("Split", &onnx_parser::parse_split);
        add_mem_op("Squeeze", &onnx_parser::parse_squeeze);
        add_mem_op("Transpose", &onnx_parser::parse_transpose);
        add_mem_op("Unsqueeze", &onnx_parser::parse_unsqueeze);
        add_mem_op("LSTM", &onnx_parser::parse_lstm);

        // init the activation function map
        init_actv_func();
    }

    void init_actv_func()
    {
        // Support name format of all lower case or the first letter capital
        map_actv_funcs.insert(std::make_pair("tanh", op::tanh{}));
        map_actv_funcs.insert(std::make_pair("relu", op::relu{}));
        map_actv_funcs.insert(std::make_pair("sigmoid", op::sigmoid{}));
        map_actv_funcs.insert(std::make_pair("leakyrelu", op::leaky_relu{}));
        map_actv_funcs.insert(std::make_pair("elu", op::elu{}));
    }

    template <class F>
    void add_op(std::string name, F f)
    {
        ops.emplace(name, [=](auto&&... xs) {
            return std::vector<instruction_ref>{f(std::forward<decltype(xs)>(xs)...)};
        });
    }

    // Multi output op
    template <class F>
    void add_multi_op(std::string name, F f)
    {
        ops.emplace(name, f);
    }

    template <class F>
    void add_mem_op(std::string name, F f)
    {
        add_op(name, [=](auto&&... xs) {
            return std::mem_fn(f)(*this, name, std::forward<decltype(xs)>(xs)...);
        });
    }

    template <class T>
    void add_binary_op(std::string name, T x)
    {
        add_op(name, [this, x](node_info info, std::vector<instruction_ref> args) {
            if(args.size() != 2)
                MIGRAPHX_THROW("binary operators should have 2 operands");
            if(contains(info.attributes, "broadcast") and contains(info.attributes, "axis"))
            {
                uint64_t broadcasted = parse_value(info.attributes.at("broadcast")).at<uint64_t>();
                if(broadcasted != 0)
                {
                    uint64_t axis = parse_value(info.attributes.at("axis")).at<uint64_t>();
                    auto l = prog.add_instruction(op::broadcast{axis, args[0]->get_shape().lens()},
                                                  args[1]);
                    return prog.add_instruction(x, args[0], l);
                }
                return prog.add_instruction(x, args);
            }
            else
            {
                return add_broadcastable_binary_op(args[0], args[1], x);
            }
        });
    }

    std::vector<std::size_t> compute_broadcasted_lens(std::vector<std::size_t> s0,
                                                      std::vector<std::size_t> s1)
    {
        // Example:
        // s0 = (3,2,4,5) and s1 = (2,1,1)
        //
        // In this case we need to broadcast (:,1,1) portion of
        // s1 plus broadcast the 1st dimension of s1
        // giving output_lens = (3,2,4,5)
        //
        // Another example:
        // s0 = (3,2,1,5) and s1 = (2,7,5)
        // In this case we need to broadcast the (:,:,1:,:) axis
        // of s0 plus the 1st dimension of s1 giving
        // output_lens = (3,2,7,5)
        if(s0.size() > s1.size())
        {
            s0.swap(s1);
        }

        std::vector<std::size_t> out_lens(s1);
        auto offset = s1.size() - s0.size();
        std::transform(s0.begin(),
                       s0.end(),
                       s1.begin() + offset,
                       out_lens.begin() + offset,
                       [&](auto a, auto b) {
                           if(a != b and a != 1 and b != 1)
                           {
                               MIGRAPHX_THROW("COMPUTE_BROADCASTLEN: shape {" +
                                              to_string_range(s0) + "} and {" +
                                              to_string_range(s1) + "} mismatch!");
                           }
                           return std::max(a, b);
                       });

        return out_lens;
    }

    instruction_ref make_contiguous(instruction_ref ins)
    {
        if(ins->get_shape().standard())
        {
            return ins;
        }

        return prog.add_instruction(op::contiguous{}, ins);
    }

    template <class T>
    instruction_ref add_broadcastable_binary_op(instruction_ref arg0, instruction_ref arg1, T x)
    {
        if(arg0->get_shape().lens() != arg1->get_shape().lens())
        {
            // Get lengths for both arguments
            auto s0       = arg0->get_shape().lens();
            auto s1       = arg1->get_shape().lens();
            auto out_lens = compute_broadcasted_lens(s0, s1);

            auto l0 = arg0;
            if(arg0->get_shape().lens() != out_lens)
                l0 = prog.add_instruction(op::multibroadcast{out_lens}, arg0);

            auto l1 = arg1;
            if(arg1->get_shape().lens() != out_lens)
                l1 = prog.add_instruction(op::multibroadcast{out_lens}, arg1);

            return prog.add_instruction(x, l0, l1);
        }
        else
        {
            return prog.add_instruction(x, {arg0, arg1});
        }
    }

    template <class T>
    void add_generic_op(std::string name, T x)
    {
        add_op(name, [this, x](const node_info&, std::vector<instruction_ref> args) {
            return prog.add_instruction(x, args);
        });
    }

    template <class T>
    void add_variadic_op(std::string name, T x)
    {
        add_op(name, [this, x](const node_info&, std::vector<instruction_ref> args) {
            return std::accumulate(std::next(args.begin()),
                                   args.end(),
                                   args.front(),
                                   [this, x](instruction_ref a, instruction_ref b) {
                                       return add_broadcastable_binary_op(a, b, x);
                                   });
        });
    }

    template <class T>
    std::vector<int64_t> to_int64_vector(const std::vector<T>& input_vector)
    {
        std::vector<int64_t> output_vector(input_vector.begin(), input_vector.end());
        return output_vector;
    }

    instruction_ref
    add_bias(const std::vector<instruction_ref>& args, instruction_ref curr_ins, uint64_t axis)
    {
        if(args.size() == 3)
        {
            auto bias_bcast =
                prog.add_instruction(op::broadcast{axis, curr_ins->get_shape().lens()}, args[2]);
            return prog.add_instruction(op::add{}, curr_ins, bias_bcast);
        }
        return curr_ins;
    }

    template <class Op>
    void check_asym_padding(instruction_ref& ins,
                            std::vector<int64_t>& padding,
                            Op& op,
                            float pad_val = 0)
    {
        if(padding[0] != padding[2] || padding[1] != padding[3])
        {
            padding = {0, 0, padding[0], padding[1], 0, 0, padding[2], padding[3]};
            ins     = prog.add_instruction(op::pad{padding, pad_val}, ins);
        }
        else
        {
            op.padding[0] = padding[0];
            op.padding[1] = padding[1];
        }
    }

    instruction_ref
    parse_clip(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        auto input_lens = args[0]->get_shape().lens();
        instruction_ref min_arg;
        instruction_ref max_arg;
        bool min_used = false;
        bool max_used = false;

        if(args.size() == 3)
        {
            min_arg  = args[1];
            max_arg  = args[2];
            min_used = true;
            max_used = true;
        }
        else if(args.size() == 2)
        {
            min_arg  = args[1];
            min_used = true;
        }
        // if using previous opset for attributes
        else if(contains(info.attributes, "min") and contains(info.attributes, "max"))
        {

            float min_val = parse_value(info.attributes.at("min")).at<float>();
            float max_val = parse_value(info.attributes.at("max")).at<float>();
            min_arg       = prog.add_literal(min_val);
            max_arg       = prog.add_literal(max_val);
            min_used      = true;
            max_used      = true;
        }

        if(min_used)
            min_arg = prog.add_instruction(op::multibroadcast{input_lens}, min_arg);

        if(max_used)
            max_arg = prog.add_instruction(op::multibroadcast{input_lens}, max_arg);

        if(min_used and max_used)
            return prog.add_instruction(op::clip{}, args[0], min_arg, max_arg);
        if(min_used)
            return prog.add_instruction(op::max{}, args[0], min_arg);

        return prog.add_instruction(op::identity{}, args[0]);
    }

    template <class Op>
    instruction_ref
    parse_softmax(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        int64_t axis = 1;
        if(contains(info.attributes, "axis"))
        {
            axis = parse_value(info.attributes.at("axis")).at<int>();
        }

        return prog.add_instruction(Op{axis}, std::move(args));
    }

    template <class Op>
    instruction_ref
    parse_arg_op(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        int64_t axis = 0;
        if(contains(info.attributes, "axis"))
        {
            axis = static_cast<int64_t>(parse_value(info.attributes.at("axis")).at<int>());
        }

        int keep_dims = 1;
        if(contains(info.attributes, "keepdims"))
        {
            keep_dims = parse_value(info.attributes.at("keepdims")).at<int>();
        }

        if(keep_dims == 0)
        {
            auto ins = prog.add_instruction(Op{axis}, std::move(args));
            return prog.add_instruction(op::squeeze{{axis}}, ins);
        }
        else
        {
            return prog.add_instruction(Op{axis}, std::move(args));
        }
    }

    template <class Op>
    instruction_ref process_auto_pad_attribute(instruction_ref ins,
                                               node_info info,
                                               Op& op,
                                               const std::vector<std::size_t>& in_lens)
    {
        if(!contains(info.attributes, "auto_pad"))
        {
            return ins;
        }

        auto auto_pad = info.attributes["auto_pad"].s();
        if(auto_pad.find("SAME") != std::string::npos)
        {
            // calculate the padding
            std::array<std::size_t, 2> out_lens;
            out_lens[0] = (in_lens[2] + op.stride[0] - 1) / op.stride[0];
            out_lens[1] = (in_lens[3] + op.stride[1] - 1) / op.stride[1];

            std::array<std::size_t, 2> explicit_pads;
            explicit_pads[0] = (out_lens[0] - 1) * op.stride[0] + op.lengths[0] - in_lens[2];
            explicit_pads[1] = (out_lens[1] - 1) * op.stride[1] + op.lengths[1] - in_lens[3];
            op.padding[0]    = explicit_pads[0] / 2;
            op.padding[1]    = explicit_pads[1] / 2;
            explicit_pads[0] -= 2 * op.padding[0];
            explicit_pads[1] -= 2 * op.padding[1];
            std::vector<std::int64_t> pads(8, 0);
            if(explicit_pads[0] != 0 or explicit_pads[1] != 0)
            {
                if(auto_pad == "SAME_UPPER")
                {
                    pads[6] = explicit_pads[0];
                    pads[7] = explicit_pads[1];
                }
                else if(auto_pad == "SAME_LOWER")
                {
                    pads[2] = explicit_pads[0];
                    pads[3] = explicit_pads[1];
                }

                // MaxPool
                if(op.mode == "max")
                {
                    ins = prog.add_instruction(op::pad{pads, std::numeric_limits<float>::lowest()},
                                               ins);
                }
                // AveragePool
                else
                {
                    ins = prog.add_instruction(op::pad{pads}, ins);
                }
            }

            op.padding_mode = op::padding_mode_t::same;
        }

        return ins;
    }

    template <class Op>
    instruction_ref
    parse_conv(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        Op op;
        auto l0      = args[0];
        auto weights = args[1];
        if(contains(info.attributes, "pads"))
        {
            if(contains(info.attributes, "auto_pad"))
            {
                auto s = info.attributes["auto_pad"].s();
                if(contains(info.attributes, "pads") and to_upper(s) != "NOTSET")
                {
                    MIGRAPHX_THROW("auto_pad and padding cannot be specified simultaneously");
                }
            }
            std::vector<std::int64_t> padding;
            copy(info.attributes["pads"].ints(), std::back_inserter(padding));
            if(padding.size() != 4)
            {
                MIGRAPHX_THROW("padding should have 4 values");
            }
            check_asym_padding(l0, padding, op);
        }
        if(contains(info.attributes, "strides"))
        {
            copy(info.attributes["strides"].ints(), op.stride.begin());
        }
        if(contains(info.attributes, "dilations"))
        {
            copy(info.attributes["dilations"].ints(), op.dilation.begin());
        }
        if(contains(info.attributes, "auto_pad"))
        {
            auto s = info.attributes["auto_pad"].s();
            if(contains(info.attributes, "pads") and to_upper(s) != "NOTSET")
            {
                MIGRAPHX_THROW("auto_pad and padding cannot be specified simultaneously");
            }

            if(s.find("SAME") != std::string::npos)
            {
                op.padding_mode                 = op::padding_mode_t::same;
                std::vector<size_t> weight_dims = weights->get_shape().lens();
                size_t weight_h                 = weight_dims[2];
                size_t weight_w                 = weight_dims[3];

                auto input_dims = l0->get_shape().lens();
                std::vector<int64_t> padding(input_dims.size());
                calculate_padding(
                    0, padding, input_dims[2], op.stride[0], op.dilation[0], weight_h);
                calculate_padding(
                    1, padding, input_dims[3], op.stride[1], op.dilation[1], weight_w);

                check_asym_padding(l0, padding, op);
            }
        }
        if(contains(info.attributes, "group"))
        {
            op.group = parse_value(info.attributes.at("group")).at<int>();
        }

        auto l1 = prog.add_instruction(op, l0, args[1]);
        return add_bias(args, l1, 1);
    }

    instruction_ref
    parse_conv_transpose(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        op::deconvolution op;
        auto l0 = args[0];
        std::vector<std::int64_t> padding;
        bool asymm_padding = false;
        if(contains(info.attributes, "pads"))
        {
            if(contains(info.attributes, "auto_pad"))
            {
                auto s = info.attributes["auto_pad"].s();
                if(contains(info.attributes, "pads") and to_upper(s) != "NOTSET")
                {
                    MIGRAPHX_THROW("auto_pad and padding cannot be specified simultaneously");
                }
            }
            copy(info.attributes["pads"].ints(), std::back_inserter(padding));
            if(padding.size() != 4)
            {
                MIGRAPHX_THROW("padding should have 4 values");
            }
            if(padding[0] != padding[2] || padding[1] != padding[3])
            {
                asymm_padding = true;
            }
            else
            {
                op.padding[0] = padding[0];
                op.padding[1] = padding[1];
            }
        }
        if(contains(info.attributes, "strides"))
        {
            copy(info.attributes["strides"].ints(), op.stride.begin());
        }
        if(contains(info.attributes, "dilations"))
        {
            copy(info.attributes["dilations"].ints(), op.dilation.begin());
        }
        if(contains(info.attributes, "auto_pad"))
        {
            auto s = info.attributes["auto_pad"].s();
            if(contains(info.attributes, "pads") and to_upper(s) != "NOTSET")
            {
                MIGRAPHX_THROW("auto_pad and padding cannot be specified simultaneously");
            }

            if(s.find("SAME") != std::string::npos)
            {
                op.padding_mode = op::padding_mode_t::same;
            }
        }

        if(contains(info.attributes, "group"))
        {
            op.group = parse_value(info.attributes.at("group")).at<int>();
        }

        auto l1                   = prog.add_instruction(op, l0, args[1]);
        std::vector<int64_t> dims = to_int64_vector(l1->get_shape().lens());
        std::vector<int64_t> curr_shape{dims[2], dims[3]};
        if(asymm_padding)
        {
            op::slice slice_op;
            slice_op.axes   = {0, 1, 2, 3};
            slice_op.starts = {0, 0, 0 + padding[0], 0 + padding[1]};
            slice_op.ends   = {
                dims[0], dims[1], curr_shape[0] - padding[2], curr_shape[1] - padding[3]};

            l1 = prog.add_instruction(slice_op, l1);
        }

        if(contains(info.attributes, "output_padding"))
        {
            std::vector<int64_t> output_padding;
            copy(info.attributes["output_padding"].ints(), std::back_inserter(output_padding));
            output_padding = {0, 0, 0, 0, 0, 0, output_padding[0], output_padding[1]};
            l1             = prog.add_instruction(op::pad{output_padding}, l1);
        }

        if(contains(info.attributes, "output_shape"))
        {
            std::vector<int64_t> output_shape;
            copy(info.attributes["output_shape"].ints(), std::back_inserter(output_shape));
            dims       = to_int64_vector(l1->get_shape().lens());
            curr_shape = {dims[2], dims[3]};
            if(curr_shape != output_shape)
            {
                std::vector<int64_t> target_padding = {0,
                                                       0,
                                                       0,
                                                       0,
                                                       0,
                                                       0,
                                                       output_shape[0] - curr_shape[0],
                                                       output_shape[1] - curr_shape[1]};
                l1 = prog.add_instruction(op::pad{target_padding}, l1);
            }
        }

        return add_bias(args, l1, 1);
    }

    instruction_ref
    parse_pooling(const std::string& name, node_info info, std::vector<instruction_ref> args)
    {
        op::pooling op{ends_with(name, "MaxPool") ? "max" : "average"};
        auto l0 = args[0];
        if(starts_with(name, "Global"))
        {
            auto lens  = args.front()->get_shape().lens();
            op.lengths = {lens[2], lens[3]};
        }

        if(contains(info.attributes, "pads"))
        {
            if(contains(info.attributes, "auto_pad"))
            {
                auto s = info.attributes["auto_pad"].s();
                if(to_upper(s) != "NOTSET")
                {
                    MIGRAPHX_THROW(
                        "PARSE_POOLING: auto_pad and padding cannot be specified simultaneously");
                }
            }

            std::vector<std::int64_t> padding;
            copy(info.attributes["pads"].ints(), std::back_inserter(padding));
            if(padding.size() != 4)
            {
                MIGRAPHX_THROW("PARSE_POOLING: padding should have 4 values");
            }
            float pad_val = 0;
            if(op.mode == "max")
                pad_val = std::numeric_limits<float>::lowest();
            check_asym_padding(l0, padding, op, pad_val);
        }

        if(contains(info.attributes, "strides"))
        {
            copy(info.attributes["strides"].ints(), op.stride.begin());
        }
        if(contains(info.attributes, "kernel_shape"))
        {
            copy(info.attributes["kernel_shape"].ints(), op.lengths.begin());
        }

        if(contains(info.attributes, "auto_pad"))
        {
            auto in_lens = args[0]->get_shape().lens();
            l0           = process_auto_pad_attribute(l0, info, op, in_lens);
        }

        return prog.add_instruction(op, l0);
    }

    instruction_ref
    parse_reshape(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        op::reshape op;
        if(args.size() == 1)
        {
            literal s = parse_value(info.attributes.at("shape"));
            s.visit([&](auto v) { copy(v, std::back_inserter(op.dims)); });
        }
        if(args.size() == 2)
        {
            auto s = args[1]->eval();
            check_arg_empty(s, "Reshape: dynamic shape is not supported");
            s.visit([&](auto v) { copy(v, std::back_inserter(op.dims)); });
        }

        return prog.add_instruction(op, make_contiguous(args[0]));
    }

    instruction_ref
    parse_flatten(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        int64_t axis = 1;
        if(contains(info.attributes, "axis"))
        {
            axis = parse_value(info.attributes.at("axis")).at<int>();
        }
        return prog.add_instruction(op::flatten{axis}, args[0]);
    }

    instruction_ref
    parse_squeeze(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        op::squeeze op;
        literal s = parse_value(info.attributes.at("axes"));
        s.visit([&](auto v) { copy(v, std::back_inserter(op.axes)); });
        return prog.add_instruction(op, make_contiguous(args[0]));
    }

    instruction_ref
    parse_unsqueeze(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        op::unsqueeze op;
        literal s = parse_value(info.attributes.at("axes"));
        s.visit([&](auto v) { copy(v, std::back_inserter(op.axes)); });
        return prog.add_instruction(op, make_contiguous(args[0]));
    }

    instruction_ref
    parse_concat(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        // change to hande axis to be negative values
        if(!contains(info.attributes, "axis"))
        {
            MIGRAPHX_THROW("PARSE_CONCAT: attribute axis is required!");
        }

        int axis = parse_value(info.attributes.at("axis")).at<int>();
        op::concat op{axis};
        return prog.add_instruction(op, std::move(args));
    }

    instruction_ref
    parse_gather(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        int axis = 0;
        if(contains(info.attributes, "axis"))
        {
            axis = parse_value(info.attributes.at("axis")).at<int>();
        }

        op::gather op{axis};
        return prog.add_instruction(op, make_contiguous(args[0]), make_contiguous(args[1]));
    }

    instruction_ref
    parse_slice(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        op::slice op;

        // slice can have up to 5 inputs, we first check the 5th one
        // to decide whether MIGRAPHX can handle this slice
        if(args.size() == 5)
        {
            migraphx::argument step_arg = args.back()->eval();
            check_arg_empty(step_arg, "PARSE_SLICE: cannot handle variable steps for slice");
            std::vector<int> steps;
            step_arg.visit([&](auto s) { steps.assign(s.begin(), s.end()); });
            if(!std::all_of(steps.begin(), steps.end(), [](auto s) { return s == 1; }))
            {
                MIGRAPHX_THROW("PARSE_SLICE: cannot handle step other than 1");
            }
        }

        if(args.size() >= 4)
        {
            migraphx::argument axes_arg = args.at(3)->eval();
            check_arg_empty(axes_arg, "PARSE_SLICE: cannot handle variable axes for slice");
            axes_arg.visit([&](auto s) { op.axes.assign(s.begin(), s.end()); });
        }
        else if(contains(info.attributes, "axes"))
        {
            literal s = parse_value(info.attributes.at("axes"));
            s.visit([&](auto v) { copy(v, std::back_inserter(op.axes)); });
        }

        if(args.size() >= 3)
        {
            migraphx::argument end_arg = args.at(2)->eval();
            check_arg_empty(end_arg, "PARSE_SLICE: cannot handle variable ends for slice");
            end_arg.visit([&](auto s) { op.ends.assign(s.begin(), s.end()); });
        }
        else if(contains(info.attributes, "ends"))
        {
            op.ends = get_indices(info.attributes.at("ends"));
        }

        if(args.size() >= 2)
        {
            migraphx::argument start_arg = args.at(1)->eval();
            check_arg_empty(start_arg, "PARSE_SLICE: cannot handle variable starts for slice");
            start_arg.visit([&](auto s) { op.starts.assign(s.begin(), s.end()); });
        }
        else if(contains(info.attributes, "starts"))
        {
            literal s = parse_value(info.attributes.at("starts"));
            s.visit([&](auto v) { copy(v, std::back_inserter(op.starts)); });
        }

        return prog.add_instruction(op, args[0]);
    }

    instruction_ref
    parse_constant(const std::string&, node_info info, const std::vector<instruction_ref>&)
    {
        literal v = parse_value(info.attributes.at("value"));
        // return empty literal
        if(v.get_shape().elements() == 0)
        {
            return prog.add_literal(literal{});
        }

        auto dim_size = info.attributes.at("value").t().dims_size();
        // if dim_size is 0, it is a scalar
        if(dim_size == 0)
        {
            migraphx::shape scalar_shape{v.get_shape().type()};
            return prog.add_literal(migraphx::literal{scalar_shape, v.data()});
        }

        return prog.add_literal(v);
    }

    instruction_ref
    parse_gemm(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        float alpha = 1.0f;
        float beta  = 1.0f;
        bool transa = false;
        bool transb = false;
        if(contains(info.attributes, "alpha"))
        {
            alpha = parse_value(info.attributes.at("alpha")).at<float>();
        }
        if(contains(info.attributes, "beta"))
        {
            beta = parse_value(info.attributes.at("beta")).at<float>();
        }
        if(contains(info.attributes, "transA"))
        {
            transa = parse_value(info.attributes.at("transA")).at<bool>();
        }
        if(contains(info.attributes, "transB"))
        {
            transb = parse_value(info.attributes.at("transB")).at<bool>();
        }

        std::vector<int64_t> perm(args[0]->get_shape().lens().size());
        std::iota(perm.begin(), perm.end(), int64_t{0});
        // swap the last two elements
        std::swap(*perm.rbegin(), *(perm.rbegin() + 1));

        auto l1 = (transa) ? prog.add_instruction(op::transpose{perm}, args[0]) : args[0];
        auto l2 = (transb) ? prog.add_instruction(op::transpose{perm}, args[1]) : args[1];
        if(args.size() == 3)
        {
            if(beta != 0.f && args[2]->get_shape().elements() > 0)
            {
                auto out_lens   = l1->get_shape().lens();
                out_lens.back() = l2->get_shape().lens().back();
                auto l3         = args[2];
                auto l3_lens    = l3->get_shape().lens();
                if(!std::equal(out_lens.begin(), out_lens.end(), l3_lens.begin(), l3_lens.end()))
                {
                    l3 = prog.add_instruction(op::multibroadcast{out_lens}, args[2]);
                }
                return prog.add_instruction(op::dot{alpha, beta}, l1, l2, l3);
            }
        }

        return prog.add_instruction(op::dot{alpha, beta}, l1, l2);
    }

    template <class Op>
    instruction_ref
    parse_matmul(const std::string&, const node_info&, std::vector<instruction_ref> args)
    {
        auto l0      = args[0];
        auto l1      = args[1];
        auto l0_lens = l0->get_shape().lens();
        auto l1_lens = l1->get_shape().lens();

        // args[0] is a vector, prepend 1 to the shape
        bool is_a_prepended = false;
        if(l0_lens.size() == 1)
        {
            is_a_prepended = true;
            l0_lens.insert(l0_lens.begin(), 1);
            l0 = prog.add_instruction(op::unsqueeze{{0}}, args[0]);
        }

        bool is_b_appended = false;
        if(l1_lens.size() == 1)
        {
            is_b_appended = true;
            l1_lens.push_back(1);
            l1 = prog.add_instruction(op::unsqueeze{{1}}, args[1]);
        }

        instruction_ref bl0 = l0;
        instruction_ref bl1 = l1;
        if(!std::equal(l0_lens.rbegin() + 2, l0_lens.rend(), l1_lens.rbegin() + 2, l1_lens.rend()))
        {
            auto l0_it = l0_lens.begin() + l0_lens.size() - 2;
            std::vector<std::size_t> l0_broadcasted_lens(l0_lens.begin(), l0_it);
            auto l1_it = l1_lens.begin() + l1_lens.size() - 2;
            std::vector<std::size_t> l1_broadcasted_lens(l1_lens.begin(), l1_it);
            auto output_lens = compute_broadcasted_lens(l0_broadcasted_lens, l1_broadcasted_lens);
            l0_broadcasted_lens = output_lens;
            l0_broadcasted_lens.insert(l0_broadcasted_lens.end(), l0_it, l0_lens.end());
            l1_broadcasted_lens = output_lens;
            l1_broadcasted_lens.insert(l1_broadcasted_lens.end(), l1_it, l1_lens.end());
            if(l0_lens != l0_broadcasted_lens)
            {
                bl0 = prog.add_instruction(op::multibroadcast{l0_broadcasted_lens}, l0);
            }
            if(l1_lens != l1_broadcasted_lens)
            {
                bl1 = prog.add_instruction(op::multibroadcast{l1_broadcasted_lens}, l1);
            }
        }

        auto dot_res     = prog.add_instruction(Op{1, 0}, bl0, bl1);
        int64_t num_axis = static_cast<int64_t>(dot_res->get_shape().lens().size());
        if(is_a_prepended)
        {
            dot_res = prog.add_instruction(op::squeeze{{num_axis - 2}}, dot_res);
            --num_axis;
        }
        if(is_b_appended)
        {
            dot_res = prog.add_instruction(op::squeeze{{num_axis - 1}}, dot_res);
        }

        return dot_res;
    }

    instruction_ref
    parse_batchnorm(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        float epsilon                                     = 1e-5f;
        float momentum                                    = 0.9f;
        op::batch_norm_inference::bn_infer_mode_t bn_mode = op::batch_norm_inference::spatial;
        if(contains(info.attributes, "epsilon"))
        {
            epsilon = parse_value(info.attributes.at("epsilon")).at<float>();
        }
        if(contains(info.attributes, "momentum"))
        {
            momentum = parse_value(info.attributes.at("momentum")).at<float>();
        }
        if(contains(info.attributes, "spatial"))
        {
            bn_mode = (parse_value(info.attributes.at("spatial")).at<uint64_t>() > 0)
                          ? op::batch_norm_inference::spatial
                          : op::batch_norm_inference::per_activation;
        }
        op::batch_norm_inference op{epsilon, momentum, bn_mode};
        return prog.add_instruction(op, std::move(args));
    }

    instruction_ref
    parse_instancenorm(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        // y = scale * ( x - mean ) / sqrt ( variance + epsilon ) + bias
        // mean = reduce_mean({H, W}, x)
        // variance = reduce_mean({H, W}, (x - mean)^2)

        float epsilon = 1e-5f;
        if(contains(info.attributes, "epsilon"))
        {
            epsilon = parse_value(info.attributes.at("epsilon")).at<float>();
        }
        auto x     = args[0];
        auto scale = args[1];
        auto bias  = args[2];
        auto dims  = x->get_shape().lens();

        auto mean            = prog.add_instruction(op::reduce_mean{{2, 3}}, x);
        auto mean_bcast      = prog.add_instruction(op::multibroadcast{dims}, mean);
        auto l0              = prog.add_instruction(op::sqdiff{}, x, mean_bcast);
        auto variance        = prog.add_instruction(op::reduce_mean{{2, 3}}, l0);
        auto l1              = prog.add_instruction(op::sub{}, x, mean_bcast);
        auto epsilon_literal = prog.add_literal(epsilon);
        auto epsilon_bcast   = prog.add_instruction(op::multibroadcast{dims}, epsilon_literal);
        auto variance_bcast  = prog.add_instruction(op::multibroadcast{dims}, variance);
        auto l2              = prog.add_instruction(op::add{}, variance_bcast, epsilon_bcast);
        auto l3              = prog.add_instruction(op::rsqrt{}, l2);
        auto l4              = prog.add_instruction(op::mul{}, l1, l3);
        auto scale_bcast     = prog.add_instruction(op::broadcast{1, dims}, scale);
        ;
        auto bias_bcast = prog.add_instruction(op::broadcast{1, dims}, bias);
        auto l5         = prog.add_instruction(op::mul{}, l4, scale_bcast);
        return prog.add_instruction(op::add{}, l5, bias_bcast);
    }

    instruction_ref
    parse_leaky_relu(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        float alpha = 0.01; // default alpha val for leaky relu
        if(contains(info.attributes, "alpha"))
        {
            alpha = parse_value(info.attributes.at("alpha")).at<float>();
        }
        op::leaky_relu op{alpha};
        return prog.add_instruction(op, args.front());
    }

    instruction_ref parse_elu(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        float alpha = 1.0; // default alpha val for elu
        if(contains(info.attributes, "alpha"))
        {
            alpha = parse_value(info.attributes.at("alpha")).at<float>();
        }
        op::elu op{alpha};
        return prog.add_instruction(op, args.front());
    }

    instruction_ref parse_lrn(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        float alpha = 0.0001;
        float beta  = 0.75;
        float bias  = 1.0;
        int size    = 1;
        if(contains(info.attributes, "alpha"))
            alpha = parse_value(info.attributes.at("alpha")).at<float>();
        if(contains(info.attributes, "beta"))
            beta = parse_value(info.attributes.at("beta")).at<float>();
        if(contains(info.attributes, "bias"))
            bias = parse_value(info.attributes.at("bias")).at<float>();
        if(contains(info.attributes, "size"))
            size = parse_value(info.attributes.at("size")).at<int>();
        op::lrn op{alpha, beta, bias, size};
        return prog.add_instruction(op, args.front());
    }

    instruction_ref
    parse_imagescaler(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        float scale = 1.0;
        std::vector<float> bias{};
        if(contains(info.attributes, "scale"))
        {
            scale = parse_value(info.attributes.at("scale")).at<float>();
        }

        if(contains(info.attributes, "bias"))
        {
            auto&& bias_floats = info.attributes["bias"].floats();
            bias               = std::vector<float>(bias_floats.begin(), bias_floats.end());
        }
        auto input_shape       = args.front()->get_shape();
        auto const& input_lens = input_shape.lens();
        auto input_type        = input_shape.type();

        auto scale_val = prog.add_literal(literal{shape{input_type}, {scale}});
        auto bias_vals = prog.add_literal(literal{shape{input_type, {bias.size()}}, bias});

        auto scale_tensor = prog.add_instruction(migraphx::op::scalar{input_lens}, scale_val);
        auto img_scaled   = prog.add_instruction(migraphx::op::mul{}, args.front(), scale_tensor);
        auto bias_bcast   = prog.add_instruction(migraphx::op::broadcast{1, input_lens}, bias_vals);
        return prog.add_instruction(migraphx::op::add{}, img_scaled, bias_bcast);
    }

    instruction_ref
    parse_transpose(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        std::vector<int64_t> perm{};
        if(contains(info.attributes, "perm"))
        {
            auto&& perm_vals = info.attributes["perm"].ints();
            perm             = std::vector<int64_t>(perm_vals.begin(), perm_vals.end());
        }
        return prog.add_instruction(migraphx::op::transpose{perm}, args.front());
    }

    instruction_ref parse_pad(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        std::vector<int64_t> pads{};
        float value = 0.0f;
        if(contains(info.attributes, "pads"))
        {
            auto&& pad_vals = info.attributes["pads"].ints();
            pads            = std::vector<int64_t>(pad_vals.begin(), pad_vals.end());
        }
        // check if padding is actually being done (at least one value is nonzero)
        if(std::all_of(pads.begin(), pads.end(), [](const int& i) { return i == 0; }))
        {
            return prog.add_instruction(migraphx::op::identity{}, args.front());
        }
        if(contains(info.attributes, "value"))
        {
            value = parse_value(info.attributes.at("value")).at<float>();
        }
        if(contains(info.attributes, "mode"))
        {
            auto mode = info.attributes.at("mode").s();
            if(mode != "constant")
                MIGRAPHX_THROW("migraphx currently only supports constant padding");
        }
        return prog.add_instruction(migraphx::op::pad{pads, value}, args.front());
    }
    // Use a literal instruction to replace the shape since, output of
    // shape operator are literals in migraphx
    instruction_ref
    parse_shape(const std::string&, const node_info&, std::vector<instruction_ref> args)
    {
        if(args.size() != 1)
            MIGRAPHX_THROW("Shape: operator should have 1 operand");
        std::vector<std::size_t> arg_shape = args[0]->get_shape().lens();
        std::vector<int64_t> vec_shape(arg_shape.size());
        migraphx::shape s(migraphx::shape::int64_type, {arg_shape.size()});
        std::transform(arg_shape.begin(), arg_shape.end(), vec_shape.begin(), [](auto i) {
            return int64_t(i);
        });
        return prog.add_literal(migraphx::literal{s, vec_shape});
    }

    // Use a literal instruction to replace the constantFill operator. In RNN, input shape
    // and value are fixed, so no need to do the actual computation for the constantFill
    // operator
    instruction_ref
    parse_constant_fill(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        int input_as_shape = 0;
        int dtype          = 1;
        float value        = 0.0f;

        if(contains(info.attributes, "dtype"))
        {
            dtype = parse_value(info.attributes.at("dtype")).at<int>();
        }
        shape::type_t type = get_type(dtype);

        if(contains(info.attributes, "input_as_shape"))
        {
            input_as_shape = parse_value(info.attributes.at("input_as_shape")).at<int>();
        }

        if(contains(info.attributes, "value"))
        {
            value = parse_value(info.attributes.at("value")).at<float>();
        }

        if(contains(info.attributes, "extra_shape"))
        {
            MIGRAPHX_THROW("ConstantFill: cannot handle extra shape attribute");
        }

        if(input_as_shape == 1)
        {
            if(args.size() != 1)
            {
                MIGRAPHX_THROW("ConstantFill: need an input argument as output shape");
            }

            if(contains(info.attributes, "shape"))
            {
                MIGRAPHX_THROW("ConstantFill: cannot set the shape argument and pass in an input "
                               "at the same time");
            }

            migraphx::argument in = args[0]->eval();
            check_arg_empty(in, "ConstantFill: dynamic shape is not supported");

            std::vector<std::size_t> dims;
            in.visit([&](auto input) { dims.assign(input.begin(), input.end()); });
            migraphx::shape s(type, dims);
            std::vector<float> values(s.elements(), value);
            return prog.add_literal(migraphx::literal(s, values));
        }
        else if(input_as_shape == 0)
        {
            if(!contains(info.attributes, "shape"))
            {
                MIGRAPHX_THROW("ConstantFill: attribute output shape is needed");
            }

            literal ls = parse_value(info.attributes.at("shape"));
            std::vector<std::size_t> dims;
            ls.visit([&](auto s) { dims.assign(s.begin(), s.end()); });
            migraphx::shape s{type, dims};
            std::vector<float> values(s.elements(), value);
            return prog.add_literal(migraphx::literal(s, values));
        }
        else
        {
            MIGRAPHX_THROW("ConstantFill: wrong value of attribute input_as_shape");
        }
    }

    instruction_ref
    parse_constant_of_shape(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        literal l_val{};
        if(contains(info.attributes, "value"))
        {
            l_val = parse_value(info.attributes.at("value"));
            if(l_val.get_shape().elements() != 1)
            {
                MIGRAPHX_THROW("ConstantOfShape: attribute value can contain only 1 elements!");
            }
        }
        else
        {
            l_val = literal({shape::float_type, {1}, {0}}, {0.0f});
        }

        // input is empty, output is a scalar
        auto type = l_val.get_shape().type();

        if(args.empty())
        {
            MIGRAPHX_THROW("ConstantOfShape : must have 1 input!");
        }
        else
        {
            migraphx::shape s;
            // empty input tensor, output is a scalar
            if(args[0]->get_shape().elements() == 0)
            {
                s = migraphx::shape{type, {1}, {0}};
            }
            else
            {
                migraphx::argument in = args[0]->eval();
                check_arg_empty(in, "ConstantOfShape: dynamic shape is not supported");

                std::vector<std::size_t> dims;
                in.visit([&](auto input) { dims.assign(input.begin(), input.end()); });
                s = migraphx::shape{type, dims};
            }

            literal l_out{};
            l_val.visit([&](auto val) {
                using val_type = std::remove_cv_t<typename decltype(val)::value_type>;
                // l_val contains only one element
                std::vector<val_type> out_vec(s.elements(), val.front());
                l_out = literal(s, out_vec);
            });

            return prog.add_literal(l_out);
        }
    }

    instruction_ref
    parse_expand(const std::string&, const node_info&, std::vector<instruction_ref> args)
    {
        auto in_lens             = args[0]->get_shape().lens();
        migraphx::argument arg_s = args[1]->eval();
        check_arg_empty(arg_s, "Expand: dynamic shape is not supported");
        std::vector<std::size_t> dims;
        arg_s.visit([&](auto input) { dims.assign(input.begin(), input.end()); });
        auto out_lens = compute_broadcasted_lens(in_lens, dims);
        return prog.add_instruction(op::multibroadcast{out_lens}, args[0]);
    }

    std::vector<instruction_ref>
    parse_rnn(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        migraphx::shape input_shape = args[0]->get_shape();
        std::size_t hidden_size     = args[1]->get_shape().lens()[1];

        if(contains(info.attributes, "hidden_size"))
        {
            std::size_t hidden_size_att = parse_value(info.attributes.at("hidden_size")).at<int>();
            if(hidden_size != hidden_size_att)
            {
                MIGRAPHX_THROW("RNN: hidden size mismatch in input and attribute");
            }
        }

        // Handling of direction to be added later
        std::string direction{"forward"};
        if(contains(info.attributes, "direction"))
        {
            direction = info.attributes.at("direction").s();
        }

        op::rnn_direction dirct = op::rnn_direction::forward;
        if(direction == "bidirectional")
        {
            dirct = op::rnn_direction::bidirectional;
        }
        else if(direction == "reverse")
        {
            dirct = op::rnn_direction::reverse;
        }

        std::vector<std::string> vec_names{"tanh"};
        if(contains(info.attributes, "activations"))
        {
            auto names = info.attributes.at("activations").strings();
            vec_names.clear();
            vec_names.resize(names.size());
            std::transform(names.begin(), names.end(), vec_names.begin(), [](auto name) {
                return to_lower(name);
            });
        }

        auto name_it = std::find_if(vec_names.begin(), vec_names.end(), [&](auto& name) {
            return (map_actv_funcs.count(name) == 0);
        });
        if(name_it != vec_names.end())
        {
            MIGRAPHX_THROW("RNN: activation function " + std::string(*name_it) + " not supported");
        }

        // bidirectional case should have two activation functions.
        // one is for forward, and the other is for reverse.
        // if only one actv function is provided, we use it in both
        // forward and reverse direction
        if(dirct == op::rnn_direction::bidirectional)
        {
            if(vec_names.size() == 1)
            {
                vec_names.push_back(vec_names.at(0));
            }
        }

        std::vector<operation> vec_actv_funcs(vec_names.size());
        std::transform(vec_names.begin(),
                       vec_names.end(),
                       vec_actv_funcs.begin(),
                       [&](const auto& fn) { return map_actv_funcs[fn]; });

        // To be added later
        float clip = 0.0;
        if(contains(info.attributes, "clip"))
        {
            clip = parse_value(info.attributes.at("clip")).at<float>();
        }

        // if the number of arguments is less than 6, append
        // undefined operator to have 6 arguments
        if(args.size() < 6)
        {
            auto ins = prog.add_instruction(op::undefined{});
            args.insert(args.end(), (6 - args.size()), ins);
        }

        // first output for the concatenation of hidden states
        auto hidden_states = prog.add_instruction(op::rnn{hidden_size, vec_actv_funcs, dirct, clip},
                                                  std::move(args));

        // second output for the last hidden state
        auto last_output = prog.add_instruction(op::rnn_last_output{}, hidden_states);

        return {hidden_states, last_output};
    }

    std::vector<instruction_ref>
    parse_gru(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        migraphx::shape input_shape = args[0]->get_shape();
        std::size_t hidden_size     = args[2]->get_shape().lens()[2];

        if(contains(info.attributes, "hidden_size"))
        {
            std::size_t hidden_size_att = parse_value(info.attributes.at("hidden_size")).at<int>();
            if(hidden_size != hidden_size_att)
            {
                MIGRAPHX_THROW("GRU: hidden size mismatch in input and attribute");
            }
        }

        // Handling of direction to be added later
        std::string direction{"forward"};
        if(contains(info.attributes, "direction"))
        {
            direction = info.attributes.at("direction").s();
        }

        op::rnn_direction dirct = op::rnn_direction::forward;
        if(direction == "bidirectional")
        {
            dirct = op::rnn_direction::bidirectional;
        }
        else if(direction == "reverse")
        {
            dirct = op::rnn_direction::reverse;
        }

        std::vector<std::string> vec_names = {"sigmoid", "tanh"};
        if(contains(info.attributes, "activations"))
        {
            auto names = info.attributes.at("activations").strings();
            vec_names.clear();
            vec_names.resize(names.size());
            std::transform(names.begin(), names.end(), vec_names.begin(), [](auto name) {
                return to_lower(name);
            });
        }

        // need 4 activation functions
        if(dirct == op::rnn_direction::bidirectional)
        {
            // 4 activation functions are used in the bidirectional
            // scenario. No spec is provided in onnx::operator. we
            // use the algorithm that: if 1 actv function is provided,
            // repeat 1 four times. If 2 actv functins are provided,
            // assume forward and reverse use the same pair of actv
            // functions. For the case of 3 actv functions provided,
            // assume the 3rd one is repeated once and used by the
            // reverse direction.
            // This may need change later
            if(vec_names.size() == 1)
            {
                vec_names.insert(vec_names.end(), 3, vec_names.at(0));
            }
            else if(vec_names.size() == 2)
            {
                // repeat the activation functions
                vec_names.push_back(vec_names.at(0));
                vec_names.push_back(vec_names.at(1));
            }
            else if(vec_names.size() == 3)
            {
                vec_names.push_back(vec_names.at(2));
            }
        }
        else
        {
            if(vec_names.size() == 1)
            {
                vec_names.push_back(vec_names.at(0));
            }
        }

        auto name_it = std::find_if(vec_names.begin(), vec_names.end(), [&](auto& name) {
            return (map_actv_funcs.count(name) == 0);
        });
        if(name_it != vec_names.end())
        {
            MIGRAPHX_THROW("GRU: activation function " + std::string(*name_it) + " not supported");
        }

        std::vector<operation> vec_actv_funcs(vec_names.size());
        std::transform(vec_names.begin(),
                       vec_names.end(),
                       vec_actv_funcs.begin(),
                       [&](const auto& name) { return map_actv_funcs[name]; });

        float clip = 0.0;
        if(contains(info.attributes, "clip"))
        {
            clip = parse_value(info.attributes.at("clip")).at<float>();
        }

        int linear_before_reset = 0;
        if(contains(info.attributes, "linear_before_reset"))
        {
            linear_before_reset = parse_value(info.attributes.at("linear_before_reset")).at<int>();
        }

        // append undefined opeator to make 6 arguments
        if(args.size() < 6)
        {
            auto ins = prog.add_instruction(op::undefined{});
            args.insert(args.end(), 6 - args.size(), ins);
        }

        // first output for concatenation of hidden states
        auto hidden_states = prog.add_instruction(
            op::gru{hidden_size, vec_actv_funcs, dirct, clip, linear_before_reset},
            std::move(args));

        // second output for last gru output
        auto last_output = prog.add_instruction(op::rnn_last_output{}, hidden_states);

        return {hidden_states, last_output};
    }

    std::vector<instruction_ref>
    parse_lstm(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        migraphx::shape input_shape = args[0]->get_shape();
        std::size_t hidden_size     = args[2]->get_shape().lens()[2];

        if(contains(info.attributes, "hidden_size"))
        {
            std::size_t hidden_size_att = parse_value(info.attributes.at("hidden_size")).at<int>();
            if(hidden_size != hidden_size_att)
            {
                MIGRAPHX_THROW("LSTM: hidden size mismatch in input and attribute");
            }
        }

        // Handling of direction to be added later
        std::string direction{"forward"};
        if(contains(info.attributes, "direction"))
        {
            direction = info.attributes.at("direction").s();
        }

        op::rnn_direction dirct = op::rnn_direction::forward;
        if(direction == "bidirectional")
        {
            dirct = op::rnn_direction::bidirectional;
        }
        else if(direction == "reverse")
        {
            dirct = op::rnn_direction::reverse;
        }
        else if(direction == "forward")
        {
            dirct = op::rnn_direction::forward;
        }
        else
        {
            MIGRAPHX_THROW("LSTM: incorrect direction attribute");
        }

        std::vector<std::string> vec_names = {"sigmoid", "tanh", "tanh"};
        if(contains(info.attributes, "activations"))
        {
            auto names = info.attributes.at("activations").strings();
            vec_names.clear();
            vec_names.resize(names.size());
            std::transform(names.begin(), names.end(), vec_names.begin(), [](auto name) {
                return to_lower(name);
            });
        }

        // need 6 activation functions for bidirectional directions
        if(dirct == op::rnn_direction::bidirectional)
        {
            // 6 activation functions are used in the bidirectional
            // scenario. No spec is provided in onnx::operator. we
            // use the algorithm that: if 1 actv function is provided,
            // repeat 1st six times. If 2 actv functins are provided,
            // repeat 2nd once, then repeat all three once
            // if 3 actv funcs are provide, repeat all three once.
            // the same algorithm is used for 4, 5, and 6 actv funcions
            // provided. This may need change later
            switch(vec_names.size())
            {
            case 1:
                vec_names = {vec_names.at(0),
                             vec_names.at(0),
                             vec_names.at(0),
                             vec_names.at(0),
                             vec_names.at(0),
                             vec_names.at(0)};
                break;

            case 2:
                // repeat the 2nd actv func once, then repeat all three another time
                vec_names = {vec_names.at(0),
                             vec_names.at(1),
                             vec_names.at(1),
                             vec_names.at(0),
                             vec_names.at(1),
                             vec_names.at(1)};
                break;

            case 3:
                // repeat all three actv funcs once
                vec_names = {vec_names.at(0),
                             vec_names.at(1),
                             vec_names.at(2),
                             vec_names.at(0),
                             vec_names.at(1),
                             vec_names.at(2)};
                break;

            case 4:
                vec_names = {vec_names.at(0),
                             vec_names.at(1),
                             vec_names.at(2),
                             vec_names.at(3),
                             vec_names.at(3),
                             vec_names.at(3)};
                break;

            case 5:
                vec_names = {vec_names.at(0),
                             vec_names.at(1),
                             vec_names.at(2),
                             vec_names.at(3),
                             vec_names.at(4),
                             vec_names.at(4)};
                break;

            default: break;
            }
        }
        else
        {
            switch(vec_names.size())
            {
            case 1: vec_names = {vec_names.at(0), vec_names.at(0), vec_names.at(0)}; break;

            case 2:
                // repeat the 2nd actv func once, so we have 3 actv funcs
                vec_names = {vec_names.at(0), vec_names.at(1), vec_names.at(1)};
                break;

            default: break;
            }
        }

        auto name_it = std::find_if(vec_names.begin(), vec_names.end(), [&](auto& name) {
            return (map_actv_funcs.count(name) == 0);
        });
        if(name_it != vec_names.end())
        {
            MIGRAPHX_THROW("LSTM: activation function " + std::string(*name_it) + " not supported");
        }

        std::vector<operation> vec_actv_funcs(vec_names.size());
        std::transform(vec_names.begin(),
                       vec_names.end(),
                       vec_actv_funcs.begin(),
                       [&](const auto& name) { return map_actv_funcs[name]; });

        float clip = 0.0;
        if(contains(info.attributes, "clip"))
        {
            clip = parse_value(info.attributes.at("clip")).at<float>();
        }

        int input_forget = 0;
        if(contains(info.attributes, "input_forget"))
        {
            input_forget = parse_value(info.attributes.at("input_forget")).at<int>();
        }

        // append undefined opeator to make 6 arguments
        if(args.size() < 8)
        {
            auto ins = prog.add_instruction(op::undefined{});
            args.insert(args.end(), 8 - args.size(), ins);
        }

        // first output for concatenation of hidden states
        auto hidden_states = prog.add_instruction(
            op::lstm{hidden_size, vec_actv_funcs, dirct, clip, input_forget}, std::move(args));

        // second output for last lstm output
        auto last_output = prog.add_instruction(op::rnn_last_output{}, hidden_states);

        // third output for last cell output
        auto last_cell_output = prog.add_instruction(op::lstm_last_cell_output{}, hidden_states);

        return {hidden_states, last_output, last_cell_output};
    }

    template <class T>
    instruction_ref
    parse_reduce_oper(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        std::size_t n_dim = args.front()->get_shape().lens().size();

        // default to reduce over all dimensions
        std::vector<int64_t> axes(n_dim);
        std::iota(axes.begin(), axes.end(), 0);
        if(contains(info.attributes, "axes"))
        {
            axes.clear();
            auto&& attr_axes = info.attributes["axes"].ints();
            axes             = std::vector<int64_t>(attr_axes.begin(), attr_axes.end());
        }

        int keep_dims = 1;
        if(contains(info.attributes, "keepdims"))
        {
            keep_dims = parse_value(info.attributes.at("keepdims")).at<int>();
        }

        if(keep_dims == 1)
        {
            return prog.add_instruction(T{axes}, std::move(args));
        }
        else
        {
            auto ins = prog.add_instruction(T{axes}, std::move(args));
            return prog.add_instruction(op::squeeze{axes}, ins);
        }
    }

    instruction_ref
    parse_reduce_l1(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        auto abs_ins = prog.add_instruction(op::abs{}, args[0]);
        return parse_reduce_oper<op::reduce_sum>({}, std::move(info), {abs_ins});
    }

    instruction_ref
    parse_reduce_l2(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        auto square_ins = prog.add_instruction(op::mul{}, args[0], args[0]);
        auto sum_ins    = parse_reduce_oper<op::reduce_sum>({}, std::move(info), {square_ins});
        return prog.add_instruction(op::sqrt{}, sum_ins);
    }

    instruction_ref
    parse_reduce_log_sum(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        auto sum_ins = parse_reduce_oper<op::reduce_sum>({}, std::move(info), std::move(args));
        return prog.add_instruction(op::log{}, sum_ins);
    }

    instruction_ref
    parse_reduce_log_sum_exp(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        auto exp_ins = prog.add_instruction(op::exp{}, args[0]);
        auto sum_ins = parse_reduce_oper<op::reduce_sum>({}, std::move(info), {exp_ins});
        return prog.add_instruction(op::log{}, sum_ins);
    }

    instruction_ref
    parse_reduce_sum_square(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        auto square_ins = prog.add_instruction(op::mul{}, args[0], args[0]);
        return parse_reduce_oper<op::reduce_sum>({}, std::move(info), {square_ins});
    }

    instruction_ref
    parse_cast(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        if(!contains(info.attributes, "to"))
        {
            MIGRAPHX_THROW("PARSE_CAST: missing to type attribute!");
        }

        int to_type        = parse_value(info.attributes.at("to")).at<int>();
        shape::type_t type = get_type(to_type);
        return prog.add_instruction(op::convert{type}, std::move(args));
    }

    std::vector<instruction_ref>
    parse_split(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        int64_t axis = 0;
        if(contains(info.attributes, "axis"))
        {
            axis = parse_value(info.attributes.at("axis")).at<int>();
        }

        auto lens      = args[0]->get_shape().lens();
        int64_t n_rank = static_cast<int64_t>(lens.size());
        if((axis < -n_rank) || (axis >= n_rank))
        {
            MIGRAPHX_THROW("PARSE_SPLIT: axis attribute out of rank!");
        }
        int64_t tuned_axis = (axis < 0) ? axis + n_rank : axis;

        std::vector<int64_t> vec_splits;
        if(contains(info.attributes, "split"))
        {
            literal s = parse_value(info.attributes.at("split"));
            s.visit([&](auto v) { vec_splits.assign(v.begin(), v.end()); });

            if(std::accumulate(vec_splits.begin(), vec_splits.end(), int64_t(0)) !=
               static_cast<int64_t>(lens[tuned_axis]))
            {
                MIGRAPHX_THROW("PARSE_SPLIT: sum of split attribute unequal to dim size of axis!");
            }
        }
        // no split attribute, input is equally divided
        else
        {
            if((lens[tuned_axis] % info.num_outputs) != 0)
            {
                MIGRAPHX_THROW("PARSE_SPLIT: input cannot be equally divided into " +
                               to_string(info.num_outputs) + " splits!");
            }
            auto dl = lens[tuned_axis] / info.num_outputs;
            vec_splits.resize(info.num_outputs, dl);
        }

        std::vector<instruction_ref> ret_ins;
        int64_t start = 0;
        for(auto sl : vec_splits)
        {
            ret_ins.push_back(
                prog.add_instruction(op::slice{{axis}, {start}, {start + sl}}, args[0]));
            start += sl;
        }

        return ret_ins;
    }

    void parse_from(std::istream& is)
    {
        onnx::ModelProto model;
        if(model.ParseFromIstream(&is))
        {
            if(model.has_graph())
            {
                this->parse_graph(model.graph());
            }
        }
        else
        {
            MIGRAPHX_THROW("Failed reading onnx file.");
        }
    }

    void parse_from(const void* data, std::size_t size)
    {
        onnx::ModelProto model;
        if(model.ParseFromArray(data, size))
        {
            if(model.has_graph())
            {
                this->parse_graph(model.graph());
            }
        }
        else
        {
            MIGRAPHX_THROW("Failed reading onnx file.");
        }
    }

    void parse_graph(const onnx::GraphProto& graph)
    {
        nodes = get_nodes(graph);
        for(auto&& f : graph.initializer())
            instructions[f.name()] = prog.add_literal(parse_tensor(f));

        for(auto&& input : graph.input())
        {
            const std::string& name = input.name();
            // input not in initializer_data, so it is a real input
            if(!contains(instructions, name))
            {
                // TODO: Get shape of input parameter
                shape s            = parse_type(input.type(), batch_size);
                instructions[name] = prog.add_parameter(name, s);
            }
        }
        for(auto&& output : graph.output())
        {
            this->parse_node(output.name());
        }

        // Find instructions corresponding to the output
        auto prog_output = graph.output();
        std::vector<std::string> all_output_names;
        std::vector<std::string> prog_output_names;
        std::transform(prog_output.begin(),
                       prog_output.end(),
                       std::back_inserter(all_output_names),
                       [](auto& node) { return node.name(); });
        std::copy_if(
            all_output_names.begin(),
            all_output_names.end(),
            std::back_inserter(prog_output_names),
            [&](const auto& name) { return !(name.empty() or instructions.count(name) == 0); });

        std::vector<instruction_ref> output_ins;
        std::transform(prog_output_names.begin(),
                       prog_output_names.end(),
                       std::back_inserter(output_ins),
                       [&](const auto& name) { return instructions[name]; });

        // add the return instuction
        prog.add_return(output_ins);
    }

    void parse_undefined(const std::string& name)
    {
        auto ins           = prog.add_instruction(op::undefined{});
        instructions[name] = ins;
    }

    void parse_node(const std::string& name)
    {
        if(name.empty())
            MIGRAPHX_THROW("Onnx node must have a name");
        if(instructions.count(name) == 0)
        {
            auto&& node = nodes.at(name);
            std::vector<instruction_ref> args;
            for(auto&& input : node.input())
            {
                if(input.empty())
                {
                    this->parse_undefined(input);
                }
                else if(nodes.count(input) > 0)
                {
                    assert(name != input);
                    this->parse_node(input);
                }
                args.push_back(instructions.at(input));
            }
            std::vector<instruction_ref> result;
            if(ops.count(node.op_type()) == 0)
            {
                if(skip_unknown_operators)
                    result.push_back(prog.add_instruction(op::unknown{node.op_type()}, args));
                else
                    MIGRAPHX_THROW("Unknown operator: " + node.op_type());
            }
            else
            {
                std::size_t output_num = static_cast<std::size_t>(node.output().size());
                result = ops[node.op_type()]({get_attributes(node), output_num}, args);
            }
            // Even no output nodes produce output in migraphx
            if(node.output().empty() and result.size() == 1)
            {
                instructions[name] = result.front();
            }
            else
            {
                auto output_num = std::min<std::size_t>(node.output().size(), result.size());
                std::transform(node.output().begin(),
                               node.output().begin() + output_num,
                               result.begin(),
                               std::inserter(instructions, instructions.end()),
                               [](auto&& x, auto&& y) { return std::make_pair(x, y); });
            }
        }
    }

    static attribute_map get_attributes(const onnx::NodeProto& node)
    {
        std::unordered_map<std::string, onnx::AttributeProto> result;
        for(auto&& attr : node.attribute())
        {
            result[attr.name()] = attr;
        }
        return result;
    }

    static node_map get_nodes(const onnx::GraphProto& graph)
    {
        std::unordered_map<std::string, onnx::NodeProto> result;
        std::size_t n = 0;
        for(auto&& node : graph.node())
        {
            if(node.output().empty())
            {
                if(node.name().empty())
                {
                    result["migraphx_unamed_node_" + std::to_string(n)] = node;
                    n++;
                }
                else
                {
                    result[node.name()] = node;
                }
            }
            for(auto&& output : node.output())
            {
                result[output] = node;
            }
        }
        return result;
    }

    static std::vector<int64_t> get_indices(const onnx::AttributeProto& attr)
    {
        std::vector<int64_t> result;
        literal s = parse_value(attr);
        s.visit([&](auto v) { copy(v, std::back_inserter(result)); });
        // Clamp large indices to -1
        std::replace_if(
            result.begin(),
            result.end(),
            [](auto x) { return x > int64_t{std::numeric_limits<std::int32_t>::max()} / 2; },
            -1);
        return result;
    }

    template <class T>
    static literal from_repeated(shape::type_t t, const T& r)
    {
        std::size_t size = r.size();
        return literal{{t, {size}}, r.begin(), r.end()};
    }

    static literal parse_value(const onnx::AttributeProto& attr)
    {
        switch(attr.type())
        {
        case onnx::AttributeProto::FLOAT: return literal{attr.f()};
        case onnx::AttributeProto::INT: return literal{attr.i()};
        case onnx::AttributeProto::TENSOR: return parse_tensor(attr.t());
        case onnx::AttributeProto::FLOATS: return from_repeated(shape::float_type, attr.floats());
        case onnx::AttributeProto::INTS: return from_repeated(shape::int64_type, attr.ints());
        case onnx::AttributeProto::UNDEFINED:
        case onnx::AttributeProto::GRAPH:
        case onnx::AttributeProto::STRING:
        case onnx::AttributeProto::STRINGS:
        case onnx::AttributeProto::TENSORS:
        case onnx::AttributeProto::SPARSE_TENSOR:
        case onnx::AttributeProto::SPARSE_TENSORS:
        case onnx::AttributeProto::GRAPHS: return {};
        }
        MIGRAPHX_THROW("Invalid attribute type");
    }

    static literal parse_tensor(const onnx::TensorProto& t)
    {
        std::vector<std::size_t> dims(t.dims().begin(), t.dims().end());
        if(t.has_raw_data())
        {
            const std::string& s = t.raw_data();
            switch(t.data_type())
            {
            case onnx::TensorProto::FLOAT: return create_literal(shape::float_type, dims, s.data());
            case onnx::TensorProto::FLOAT16:
                return create_literal(shape::half_type, dims, s.data());
            case onnx::TensorProto::DOUBLE:
                return create_literal(shape::double_type, dims, s.data());
            case onnx::TensorProto::INT64: return create_literal(shape::int64_type, dims, s.data());
            case onnx::TensorProto::INT8:
            case onnx::TensorProto::UINT16:
            case onnx::TensorProto::INT16:
            case onnx::TensorProto::INT32:
            case onnx::TensorProto::BOOL: return create_literal(shape::int32_type, dims, s.data());
            case onnx::TensorProto::UINT8:
            case onnx::TensorProto::STRING:
            case onnx::TensorProto::UNDEFINED:
            case onnx::TensorProto::UINT32:
            case onnx::TensorProto::UINT64:
            case onnx::TensorProto::COMPLEX64:
            case onnx::TensorProto::COMPLEX128: throw std::runtime_error("");
            }
            MIGRAPHX_THROW("Invalid tensor type");
        }
        switch(t.data_type())
        {
        case onnx::TensorProto::INT8:
        case onnx::TensorProto::UINT16:
        case onnx::TensorProto::INT16:
        case onnx::TensorProto::INT32:
        case onnx::TensorProto::BOOL:
            return create_literal(shape::int32_type, dims, t.int32_data());
        case onnx::TensorProto::INT64:
            return create_literal(shape::int64_type, dims, t.int64_data());
        case onnx::TensorProto::DOUBLE:
            return create_literal(shape::double_type, dims, t.double_data());
        case onnx::TensorProto::FLOAT:
            return create_literal(shape::float_type, dims, t.float_data());
        case onnx::TensorProto::FLOAT16:
        {
            std::vector<uint16_t> data_uint16(t.int32_data().begin(), t.int32_data().end());
            std::vector<half> data_half;
            std::transform(data_uint16.begin(),
                           data_uint16.end(),
                           std::back_inserter(data_half),
                           [](uint16_t raw_val) { return *reinterpret_cast<half*>(&raw_val); });
            return create_literal(shape::half_type, dims, data_half);
        }
        case onnx::TensorProto::UNDEFINED:
        case onnx::TensorProto::UINT8:
        case onnx::TensorProto::STRING:
        case onnx::TensorProto::UINT32:
        case onnx::TensorProto::UINT64:
        case onnx::TensorProto::COMPLEX64:
        case onnx::TensorProto::COMPLEX128: throw std::runtime_error("");
        }
        MIGRAPHX_THROW("Invalid tensor type");
    }

    static literal
    create_literal(shape::type_t shape_type, const std::vector<size_t>& dims, const char* data)
    {
        // in case of scalar constants in onnx file, use dims=1 to fill initializer data
        if(dims.empty())
            return literal{{shape_type}, data};
        return literal{{shape_type, dims}, data};
    }

    template <class T, MIGRAPHX_REQUIRES(not std::is_pointer<T>{})>
    static literal create_literal(shape::type_t shape_type, const std::vector<size_t>& dims, T data)
    {
        if(dims.empty())
            return literal{{shape_type}, data.begin(), data.end()};
        return literal{{shape_type, dims}, data.begin(), data.end()};
    }

    static shape parse_type(const onnx::TypeProto& t, const unsigned int batch_size)
    {
        shape::type_t shape_type{};
        switch(t.tensor_type().elem_type())
        {
        case onnx::TensorProto::FLOAT: shape_type = shape::float_type; break;
        case onnx::TensorProto::INT8: shape_type = shape::int8_type; break;
        case onnx::TensorProto::UINT16: shape_type = shape::uint16_type; break;
        case onnx::TensorProto::INT16: shape_type = shape::int16_type; break;
        case onnx::TensorProto::INT32: shape_type = shape::int32_type; break;
        case onnx::TensorProto::INT64: shape_type = shape::int64_type; break;
        case onnx::TensorProto::FLOAT16: shape_type = shape::half_type; break;
        case onnx::TensorProto::DOUBLE: shape_type = shape::double_type; break;
        case onnx::TensorProto::UINT32: shape_type = shape::uint32_type; break;
        case onnx::TensorProto::UINT64: shape_type = shape::uint64_type; break;
        case onnx::TensorProto::UINT8:
        case onnx::TensorProto::STRING:
        case onnx::TensorProto::BOOL:
        case onnx::TensorProto::UNDEFINED:
        case onnx::TensorProto::COMPLEX64:
        case onnx::TensorProto::COMPLEX128:
            break; // throw std::runtime_error("Unsupported type");
        }
        std::vector<std::size_t> dims;
        auto&& tensor_dims = t.tensor_type().shape().dim();
        std::transform(tensor_dims.begin(),
                       tensor_dims.end(),
                       std::back_inserter(dims),
                       [&](auto&& d) -> std::size_t {
                           if(d.has_dim_value())
                           {
                               if(static_cast<int>(d.dim_value()) <= 0)
                                   return batch_size;
                               return d.dim_value();
                           }
                           return batch_size;
                       });
        if(dims.empty())
            return {shape_type};

        return {shape_type, dims};
    }

    shape::type_t get_type(int dtype)
    {
        switch(dtype)
        {
        case 1: return shape::float_type;
        case 2: return shape::uint8_type;
        case 3: return shape::int8_type;
        case 4: return shape::uint16_type;
        case 5: return shape::int16_type;
        case 6: return shape::int32_type;
        case 7: return shape::int64_type;
        case 10: return shape::half_type;
        case 11: return shape::double_type;
        case 12: return shape::uint32_type;
        case 13: return shape::uint64_type;
        default:
        {
            MIGRAPHX_THROW("Prototensor data type " + std::to_string(dtype) + " not supported");
        }
        }
    }

    void check_arg_empty(const argument& arg, const std::string& msg)
    {
        if(arg.empty())
        {
            MIGRAPHX_THROW(msg);
        }
    }
};

template <class... Ts>
program parse_onnx_from(onnx_options options, Ts&&... xs)
{
    onnx_parser parser;
    parser.batch_size             = options.batch_size;
    parser.skip_unknown_operators = options.skip_unknown_operators;
    if(options.print_program_on_error)
    {
        // Log the program when it can't be parsed
        try
        {
            parser.parse_from(std::forward<Ts>(xs)...);
        }
        catch(...)
        {
            std::cerr << parser.prog << std::endl;
            throw;
        }
    }
    else
    {
        parser.parse_from(std::forward<Ts>(xs)...);
    }
    return std::move(parser.prog);
}

program parse_onnx(const std::string& name, onnx_options options)
{
    std::fstream input(name.c_str(), std::ios::in | std::ios::binary);
    return parse_onnx_from(options, input);
}

program parse_onnx_buffer(const std::string& buffer, onnx_options options)
{
    return parse_onnx_from(options, buffer.data(), buffer.size());
}

program parse_onnx_buffer(const void* data, std::size_t size, onnx_options options)
{
    return parse_onnx_from(options, data, size);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
