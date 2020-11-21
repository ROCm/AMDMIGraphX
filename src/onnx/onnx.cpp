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
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/config.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/pad_calc.hpp>
#include <migraphx/type_traits.hpp>
#include <migraphx/float_equal.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/filesystem.hpp>

#include <migraphx/op/as_shape.hpp>
#include <migraphx/op/batch_norm_inference.hpp>
#include <migraphx/op/broadcast.hpp>
#include <migraphx/op/concat.hpp>
#include <migraphx/op/convert.hpp>
#include <migraphx/op/gather.hpp>
#include <migraphx/op/gru.hpp>
#include <migraphx/op/lrn.hpp>
#include <migraphx/op/lstm.hpp>
#include <migraphx/op/multibroadcast.hpp>
#include <migraphx/op/pad.hpp>
#include <migraphx/op/reshape.hpp>
#include <migraphx/op/rnn.hpp>
#include <migraphx/op/rnn_last_cell_output.hpp>
#include <migraphx/op/rnn_last_hs_output.hpp>
#include <migraphx/op/rnn_variable_seq_lens.hpp>
#include <migraphx/op/rnn_var_sl_last_output.hpp>
#include <migraphx/op/scalar.hpp>
#include <migraphx/op/slice.hpp>
#include <migraphx/op/squeeze.hpp>
#include <migraphx/op/transpose.hpp>
#include <migraphx/op/undefined.hpp>
#include <migraphx/op/unknown.hpp>
#include <migraphx/op/unsqueeze.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace onnx = onnx_for_migraphx;

struct onnx_parser
{
    std::string filename;
    std::string path    = ".";
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
    program prog                  = program();
    module* mm                    = prog.get_main_module();
    bool is_pytorch               = false;
    std::size_t default_dim_value = 1;
    std::unordered_map<std::string, std::vector<std::size_t>> map_input_dims;
    bool skip_unknown_operators = false;

    std::unordered_map<std::string, op_func> ops;
    std::unordered_map<std::string, operation> map_actv_funcs;

    onnx_parser()
    {
        // sort onnx operator alphabetically through name
        add_generic_op("Abs", "abs");
        add_generic_op("Acos", "acos");
        add_generic_op("Acosh", "acosh");
        add_generic_op("Asin", "asin");
        add_generic_op("Asinh", "asinh");
        add_generic_op("Atan", "atan");
        add_generic_op("Atanh", "atanh");
        add_generic_op("Ceil", "ceil");
        add_generic_op("Concat", "concat");
        add_generic_op("Cos", "cos");
        add_generic_op("Cosh", "cosh");
        add_generic_op("Erf", "erf");
        add_generic_op("Exp", "exp");
        add_generic_op("Flatten", "flatten");
        add_generic_op("Floor", "floor");
        add_generic_op("Gather", "gather", true);
        add_generic_op("Identity", "identity");
        add_generic_op("Log", "log");
        add_generic_op("LogSoftmax", "logsoftmax");
        add_generic_op("Neg", "neg");
        add_generic_op("Reciprocal", "recip");
        add_generic_op("Relu", "relu");
        add_generic_op("Round", "round");
        add_generic_op("Sigmoid", "sigmoid");
        add_generic_op("Sign", "sign");
        add_generic_op("Sin", "sin");
        add_generic_op("Sinh", "sinh");
        add_generic_op("Softmax", "softmax");
        add_generic_op("Sqrt", "sqrt");
        add_generic_op("Squeeze", "squeeze", true);
        add_generic_op("Tan", "tan");
        add_generic_op("Tanh", "tanh");
        add_generic_op("Unsqueeze", "unsqueeze", true);

        add_binary_op("Add", "add");
        add_binary_op("Div", "div");
        add_binary_op("Mul", "mul");
        add_binary_op("Pow", "pow");
        add_binary_op("PRelu", "prelu");
        add_binary_op("Sub", "sub");

        add_variadic_op("Sum", "add");
        add_variadic_op("Max", "max");
        add_variadic_op("Min", "min");

        add_mem_op("ATen", &onnx_parser::parse_aten);
        add_mem_op("AveragePool", &onnx_parser::parse_pooling);
        add_mem_op("ArgMax", "argmax", &onnx_parser::parse_arg_op);
        add_mem_op("ArgMin", "argmin", &onnx_parser::parse_arg_op);
        add_mem_op("BatchNormalization", &onnx_parser::parse_batchnorm);
        add_mem_op("Cast", &onnx_parser::parse_cast);
        add_mem_op("Clip", &onnx_parser::parse_clip);
        add_mem_op("Constant", &onnx_parser::parse_constant);
        add_mem_op("ConstantFill", &onnx_parser::parse_constant_fill);
        add_mem_op("ConstantOfShape", &onnx_parser::parse_constant_of_shape);
        add_mem_op("Conv", "convolution", &onnx_parser::parse_conv);
        add_mem_op("ConvInteger", "quant_convolution", &onnx_parser::parse_conv);
        add_mem_op("ConvTranspose", &onnx_parser::parse_conv_transpose);
        add_mem_op("Dropout", &onnx_parser::parse_dropout);
        add_mem_op("Elu", &onnx_parser::parse_elu);
        add_mem_op("Equal", "equal", &onnx_parser::parse_compare_op);
        add_mem_op("Expand", &onnx_parser::parse_expand);
        add_mem_op("GatherElements", &onnx_parser::parse_gather_elements);
        add_mem_op("Gemm", &onnx_parser::parse_gemm);
        add_mem_op("GlobalAveragePool", &onnx_parser::parse_pooling);
        add_mem_op("GlobalMaxPool", &onnx_parser::parse_pooling);
        add_mem_op("Greater", "greater", &onnx_parser::parse_compare_op);
        add_mem_op("GRU", &onnx_parser::parse_gru);
        add_mem_op("ImageScaler", &onnx_parser::parse_imagescaler);
        add_mem_op("InstanceNormalization", &onnx_parser::parse_instancenorm);
        add_mem_op("LeakyRelu", &onnx_parser::parse_leaky_relu);
        add_mem_op("Less", "less", &onnx_parser::parse_compare_op);
        add_mem_op("LRN", &onnx_parser::parse_lrn);
        add_mem_op("LSTM", &onnx_parser::parse_lstm);
        add_mem_op("MatMul", "dot", &onnx_parser::parse_matmul);
        add_mem_op("MatMulInteger", "quant_dot", &onnx_parser::parse_matmul);
        add_mem_op("MaxPool", &onnx_parser::parse_pooling);
        add_mem_op("NonZero", &onnx_parser::parse_nonzero);
        add_mem_op("OneHot", &onnx_parser::parse_onehot);
        add_mem_op("Pad", &onnx_parser::parse_pad);
        add_mem_op("Range", &onnx_parser::parse_range);
        add_mem_op("ReduceL1", &onnx_parser::parse_reduce_l1);
        add_mem_op("ReduceL2", &onnx_parser::parse_reduce_l2);
        add_mem_op("ReduceLogSum", &onnx_parser::parse_reduce_log_sum);
        add_mem_op("ReduceLogSumExp", &onnx_parser::parse_reduce_log_sum_exp);
        add_mem_op("ReduceMax", "reduce_max", &onnx_parser::parse_reduce_oper);
        add_mem_op("ReduceMean", "reduce_mean", &onnx_parser::parse_reduce_oper);
        add_mem_op("ReduceMin", "reduce_min", &onnx_parser::parse_reduce_oper);
        add_mem_op("ReduceProd", "reduce_prod", &onnx_parser::parse_reduce_oper);
        add_mem_op("ReduceSum", "reduce_sum", &onnx_parser::parse_reduce_oper);
        add_mem_op("ReduceSumSquare", &onnx_parser::parse_reduce_sum_square);
        add_mem_op("Reshape", &onnx_parser::parse_reshape);
        add_mem_op("Resize", &onnx_parser::parse_resize);
        add_mem_op("RNN", &onnx_parser::parse_rnn);
        add_mem_op("Selu", &onnx_parser::parse_selu);
        add_mem_op("Shape", &onnx_parser::parse_shape);
        add_mem_op("Slice", &onnx_parser::parse_slice);
        add_mem_op("Split", &onnx_parser::parse_split);
        add_mem_op("Tile", &onnx_parser::parse_tile);
        add_mem_op("Transpose", &onnx_parser::parse_transpose);
        add_mem_op("Upsample", &onnx_parser::parse_upsample);
        add_mem_op("Where", &onnx_parser::parse_where);

        // init the activation function map
        init_actv_func();
    }

    void init_actv_func()
    {
        // Support name format of all lower case or the first letter capital
        map_actv_funcs.insert(std::make_pair("tanh", make_op("tanh")));
        map_actv_funcs.insert(std::make_pair("relu", make_op("relu")));
        map_actv_funcs.insert(std::make_pair("sigmoid", make_op("sigmoid")));
        map_actv_funcs.insert(std::make_pair("leakyrelu", make_op("leaky_relu")));
        map_actv_funcs.insert(std::make_pair("elu", make_op("elu")));
    }

    operation load(const std::string& name, const node_info& info) const
    {
        auto op = make_op(name);
        auto v  = op.to_value();
        for(auto&& x : v)
        {
            if(info.attributes.count(x.get_key()) == 0)
                continue;
            literal s = parse_value(info.attributes.at(x.get_key()));
            if(x.is_array())
            {
                std::vector<value> values;
                s.visit([&](auto y) {
                    std::transform(y.begin(), y.end(), std::back_inserter(values), [](auto z) {
                        return value(z);
                    });
                });
                x = values;
            }
            else
            {
                s.visit([&](auto y) { x = y.front(); });
            }
        }
        op.from_value(v);
        return op;
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
    void add_mem_op(const std::string& name, F f)
    {
        add_op(name, [=](auto&&... xs) {
            return std::mem_fn(f)(*this, name, std::forward<decltype(xs)>(xs)...);
        });
    }

    template <class F>
    void add_mem_op(const std::string& onnx_name, const std::string& op_name, F f)
    {
        add_op(onnx_name, [=](auto&&... xs) {
            return std::mem_fn(f)(*this, onnx_name, op_name, std::forward<decltype(xs)>(xs)...);
        });
    }

    void add_binary_op(const std::string& onnx_name, const std::string& op_name)
    {
        add_op(onnx_name, [this, op_name](node_info info, std::vector<instruction_ref> args) {
            if(args.size() != 2)
                MIGRAPHX_THROW("binary operators should have 2 operands");
            if(contains(info.attributes, "broadcast") and contains(info.attributes, "axis"))
            {
                uint64_t broadcasted = parse_value(info.attributes.at("broadcast")).at<uint64_t>();
                if(broadcasted != 0)
                {
                    uint64_t axis = parse_value(info.attributes.at("axis")).at<uint64_t>();
                    auto l = mm->add_instruction(op::broadcast{axis, args[0]->get_shape().lens()},
                                                 args[1]);
                    return mm->add_instruction(make_op(op_name), args[0], l);
                }
                return mm->add_instruction(make_op(op_name), args);
            }
            else
            {
                return add_broadcastable_binary_op(args[0], args[1], op_name);
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

    instruction_ref make_contiguous(instruction_ref ins) const
    {
        if(ins->get_shape().standard())
        {
            return ins;
        }

        return mm->add_instruction(make_op("contiguous"), ins);
    }

    instruction_ref
    add_broadcastable_binary_op(instruction_ref arg0, instruction_ref arg1, const std::string& name)
    {
        if(arg0->get_shape().lens() != arg1->get_shape().lens())
        {
            // Get lengths for both arguments
            auto s0       = arg0->get_shape().lens();
            auto s1       = arg1->get_shape().lens();
            auto out_lens = compute_broadcasted_lens(s0, s1);

            auto l0 = arg0;
            if(arg0->get_shape().lens() != out_lens)
                l0 = mm->add_instruction(op::multibroadcast{out_lens}, arg0);

            auto l1 = arg1;
            if(arg1->get_shape().lens() != out_lens)
                l1 = mm->add_instruction(op::multibroadcast{out_lens}, arg1);

            return mm->add_instruction(make_op(name), l0, l1);
        }
        else
        {
            return mm->add_instruction(make_op(name), {arg0, arg1});
        }
    }

    void add_generic_op(const std::string& onnx_name,
                        const std::string& op_name,
                        bool contiguous = false)
    {
        add_op(
            onnx_name,
            [this, op_name, contiguous](const node_info& info, std::vector<instruction_ref> args) {
                auto op = load(op_name, info);
                if(contiguous)
                {
                    std::transform(args.begin(), args.end(), args.begin(), [&](auto arg) {
                        return this->make_contiguous(arg);
                    });
                }
                return mm->add_instruction(op, args);
            });
    }

    void add_variadic_op(const std::string& onnx_name, const std::string& op_name)
    {
        add_op(onnx_name, [this, op_name](const node_info&, std::vector<instruction_ref> args) {
            return std::accumulate(std::next(args.begin()),
                                   args.end(),
                                   args.front(),
                                   [this, op_name](instruction_ref a, instruction_ref b) {
                                       return add_broadcastable_binary_op(a, b, op_name);
                                   });
        });
    }

    template <class T>
    std::vector<int64_t> to_int64_vector(const std::vector<T>& input_vector)
    {
        std::vector<int64_t> output_vector(input_vector.begin(), input_vector.end());
        return output_vector;
    }

    instruction_ref add_bias(const std::vector<instruction_ref>& args,
                             instruction_ref curr_ins,
                             uint64_t axis) const
    {
        if(args.size() == 3)
        {
            auto bias_bcast =
                mm->add_instruction(op::broadcast{axis, curr_ins->get_shape().lens()}, args[2]);
            return mm->add_instruction(make_op("add"), curr_ins, bias_bcast);
        }
        return curr_ins;
    }

    static bool is_asym_padding(const std::vector<int64_t>& padding)
    {
        assert(padding.size() % 2 == 0);
        size_t pad_ndims = padding.size() / 2;

        for(size_t i = 0; i < pad_ndims; i++)
        {
            if(padding[i] != padding[i + pad_ndims])
            {
                return true;
            }
        }
        return false;
    }

    void check_asym_padding(instruction_ref& ins,
                            const std::vector<int64_t>& padding,
                            value& v,
                            int count_include_pad = 0,
                            float pad_val         = 0) const
    {
        size_t pad_ndims  = padding.size() / 2;
        auto left_pad_it  = padding.begin();
        auto right_pad_it = left_pad_it + pad_ndims;

        if(is_asym_padding(padding) or count_include_pad == 1)
        {
            std::vector<int64_t> asym_pads{0, 0, 0, 0}; // don't pad N and C
            // add left pads
            asym_pads.insert(asym_pads.begin() + 2, left_pad_it, right_pad_it);
            // add right pads
            asym_pads.insert(asym_pads.begin() + pad_ndims + 4, right_pad_it, padding.end());
            ins = mm->add_instruction(op::pad{asym_pads, pad_val}, ins);
        }
        else
        {
            v["padding"] = std::vector<size_t>(left_pad_it, right_pad_it);
        }
    }

    instruction_ref
    parse_clip(const std::string&, node_info info, std::vector<instruction_ref> args) const
    {
        auto input_lens = args[0]->get_shape().lens();
        instruction_ref min_arg;
        instruction_ref max_arg;
        bool min_used = false;
        bool max_used = false;

        if(args.size() == 3 and args[2]->name() != "undefined")
        {
            max_arg  = args[2];
            max_used = true;
        }

        if(args.size() >= 2 and args[1]->name() != "undefined")
        {
            min_arg  = args[1];
            min_used = true;
        }
        // if using previous opset for attributes
        else if(contains(info.attributes, "min") and contains(info.attributes, "max"))
        {

            float min_val = parse_value(info.attributes.at("min")).at<float>();
            float max_val = parse_value(info.attributes.at("max")).at<float>();
            min_arg       = mm->add_literal(min_val);
            max_arg       = mm->add_literal(max_val);
            min_used      = true;
            max_used      = true;
        }

        if(min_used)
        {
            min_arg = mm->add_instruction(op::multibroadcast{input_lens}, min_arg);
        }

        if(max_used)
        {
            max_arg = mm->add_instruction(op::multibroadcast{input_lens}, max_arg);
        }

        if(min_used and max_used)
        {
            return mm->add_instruction(make_op("clip"), args[0], min_arg, max_arg);
        }
        else if(max_used)
        {
            return mm->add_instruction(make_op("min"), args[0], max_arg);
        }
        else if(min_used)
        {
            return mm->add_instruction(make_op("max"), args[0], min_arg);
        }
        else
        {
            return mm->add_instruction(make_op("identity"), args[0]);
        }
    }

    instruction_ref parse_arg_op(const std::string&,
                                 const std::string& op_name,
                                 node_info info,
                                 std::vector<instruction_ref> args) const
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
            auto ins = mm->add_instruction(make_op(op_name, {{"axis", axis}}), std::move(args));
            return mm->add_instruction(op::squeeze{{axis}}, ins);
        }
        else
        {
            return mm->add_instruction(make_op(op_name, {{"axis", axis}}), std::move(args));
        }
    }

    void calc_reflect_indices(std::vector<int>& indices, const int64_t num_dims)
    {
        int k         = 0;
        bool reversed = false;
        // in reflect padding, if the num_pads > num_dims,
        // compute the extra pad indices periodically, ex. ( 1, 2, 3, 2, 1, 0)
        for(int& idx : indices)
        {
            if(k == num_dims - 1)
                reversed = true;
            if(k == 0)
                reversed = false;
            if(reversed)
                k--;
            else
                k++;
            idx = k;
        }
    }

    instruction_ref reflect_pad(const std::vector<int64_t>& pads, instruction_ref input)
    {
        size_t num_dims = pads.size() / 2;
        std::vector<int> ldims(pads.begin(), pads.begin() + num_dims);
        std::vector<int> rdims(pads.begin() + num_dims, pads.end());
        assert(ldims.size() == rdims.size());

        std::vector<int64_t> axes(num_dims);
        std::iota(axes.begin(), axes.end(), int64_t{0});

        // iterate over dimensions, starting from lowest dimension
        for(int64_t i = num_dims - 1; i >= 0; i--)
        {
            auto axis   = i;
            auto lcount = ldims.at(i);
            auto rcount = rdims.at(i);
            if(lcount == 0 and rcount == 0) // no padding for current dim
                continue;

            // calculate starts and ends for each iteration since shape may change
            std::vector<size_t> dims = input->get_shape().lens();
            std::vector<int64_t> starts(axes.size(), 0);
            std::vector<int64_t> ends(dims.begin(), dims.end());
            std::vector<instruction_ref> slices;

            auto starts_it = starts.begin() + i;
            auto ends_it   = ends.begin() + i;
            auto dims_it   = dims.begin() + i;

            std::vector<int> l_indices(lcount);
            std::vector<int> r_indices(rcount);

            // compute slice indices in a periodic fashion
            calc_reflect_indices(l_indices, *dims_it);
            calc_reflect_indices(r_indices, *dims_it);

            for(int idx : l_indices)
            {
                *starts_it = idx;
                *ends_it   = *starts_it + 1;
                slices.push_back(mm->add_instruction(op::slice{axes, starts, ends}, input));
            }
            // when padding on the left side, the outermost pad should be at the beginning
            std::reverse(slices.begin(), slices.end());
            slices.push_back(input);
            for(int idx : r_indices)
            {
                *starts_it = *dims_it - idx - 1;
                *ends_it   = *starts_it + 1;
                slices.push_back(mm->add_instruction(op::slice{axes, starts, ends}, input));
            }
            input = mm->add_instruction(op::concat{axis}, slices);
        }
        return input;
    }

    void check_attr_sizes(size_t kdims, size_t attr_size, const std::string& error_msg)
    {
        if(kdims != attr_size)
        {
            MIGRAPHX_THROW(error_msg + " k-dims: " + to_string(kdims) +
                           " attribute size: " + to_string(attr_size));
        }
    }

    void recalc_conv_attributes(value& v, size_t kdims)
    {
        if(v["padding"].size() != kdims)
        {
            v["padding"].resize(kdims);
            std::fill_n(v["padding"].begin(), kdims, 0);
        }
        if(v["stride"].size() != kdims)
        {
            v["stride"].resize(kdims);
            std::fill_n(v["stride"].begin(), kdims, 1);
        }
        if(v["dilation"].size() != kdims)
        {
            v["dilation"].resize(kdims);
            std::fill_n(v["dilation"].begin(), kdims, 1);
        }
    }

    static void cal_auto_padding_size(node_info info,
                                      value& v,
                                      const std::vector<std::size_t>& k_lens,
                                      const std::vector<std::size_t>& dilation,
                                      const std::vector<std::size_t>& in_lens,
                                      std::vector<int64_t>& paddings)
    {
        size_t kdims = in_lens.size() - 2;
        assert(k_lens.size() == kdims and dilation.size() == kdims);

        if(!contains(info.attributes, "auto_pad"))
        {
            return;
        }

        auto auto_pad = info.attributes["auto_pad"].s();
        if(auto_pad.find("SAME") != std::string::npos)
        {
            bool is_same_upper = (auto_pad.find("SAME_UPPER") != std::string::npos);
            paddings.resize(2 * kdims);

            for(size_t i = 0; i < paddings.size() / 2; i++)
            {
                calculate_padding(i,
                                  paddings,
                                  in_lens[i + 2],
                                  v["stride"][i].to<int64_t>(),
                                  dilation[i],
                                  k_lens[i],
                                  is_same_upper);
            }
        }
    }

    static void check_padding_mode(node_info info, const std::string& op_name)
    {
        // ensure pads availabe only when auto_pad is "NOT_SET"
        if(contains(info.attributes, "pads") and contains(info.attributes, "auto_pad"))
        {
            auto s = info.attributes["auto_pad"].s();
            if(to_upper(s) != "NOTSET")
            {
                MIGRAPHX_THROW("PARSE_" + op_name +
                               ": auto_pad and padding cannot be specified simultaneously");
            }
        }
    }

    instruction_ref parse_conv(const std::string&,
                               const std::string& op_name,
                               node_info info,
                               std::vector<instruction_ref> args)
    {
        auto op      = make_op(op_name);
        auto values  = op.to_value();
        auto l0      = args[0];
        auto weights = args[1];
        auto in_lens = l0->get_shape().lens();
        assert(in_lens.size() > 2);
        auto kdims = in_lens.size() - 2;

        // ensure pads availabe only when auto_pad is "NOT_SET"
        check_padding_mode(info, "CONV");

        if(contains(info.attributes, "strides"))
        {
            values["stride"].clear();
            copy(info.attributes["strides"].ints(), std::back_inserter(values["stride"]));
            check_attr_sizes(kdims, values["stride"].size(), "PARSE_CONV: inconsistent strides");
        }
        if(contains(info.attributes, "dilations"))
        {
            values["dilation"].clear();
            copy(info.attributes["dilations"].ints(), std::back_inserter(values["dilation"]));
            check_attr_sizes(
                kdims, values["dilation"].size(), "PARSE_CONV: inconsistent dilations");
        }

        std::vector<int64_t> padding;
        if(contains(info.attributes, "pads"))
        {
            values["padding"].clear();
            copy(info.attributes["pads"].ints(), std::back_inserter(padding));
            check_attr_sizes(kdims, padding.size() / 2, "PARSE_CONV: inconsistent paddings");
        }

        if(contains(info.attributes, "auto_pad"))
        {
            auto weight_lens = weights->get_shape().lens();
            std::vector<std::size_t> k_lens(weight_lens.begin() + 2, weight_lens.end());
            cal_auto_padding_size(info,
                                  values,
                                  k_lens,
                                  values["dilation"].to_vector<std::size_t>(),
                                  in_lens,
                                  padding);
            auto auto_pad = info.attributes["auto_pad"].s();
            if(auto_pad.find("SAME") != std::string::npos)
            {
                values["padding_mode"] = to_value(op::padding_mode_t::same);
            }
        }
        check_asym_padding(l0, padding, values);

        if(contains(info.attributes, "group"))
        {
            values["group"] = parse_value(info.attributes.at("group")).at<int>();
        }

        recalc_conv_attributes(values, kdims);

        op.from_value(values);
        auto l1 = mm->add_instruction(op, l0, args[1]);
        return add_bias(args, l1, 1);
    }

    instruction_ref
    parse_conv_transpose(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        operation op = make_op("deconvolution");
        value values = op.to_value();
        // op::deconvolution op;
        auto l0 = args[0];
        std::vector<std::int64_t> padding;
        bool asym_padding = false;
        auto in_lens      = l0->get_shape().lens();
        assert(in_lens.size() > 2);
        auto kdims = in_lens.size() - 2;

        // ensure pads availabe only when auto_pad is "NOT_SET"
        check_padding_mode(info, "CONV_TRANSPOSE");

        if(contains(info.attributes, "pads"))
        {
            copy(info.attributes["pads"].ints(), std::back_inserter(padding));

            asym_padding = is_asym_padding(padding);

            if(not asym_padding)
            {
                size_t pad_ndims = padding.size() / 2;
                check_attr_sizes(kdims, pad_ndims, "PARSE_CONV_TRANSPOSE: inconsistent paddings");
                values["padding"].clear();
                std::transform(padding.begin(),
                               padding.begin() + pad_ndims,
                               std::back_inserter(values["padding"]),
                               [](auto pad_val) { return pad_val; });
            }
        }
        if(contains(info.attributes, "strides"))
        {
            values["stride"].clear();
            copy(info.attributes["strides"].ints(), std::back_inserter(values["stride"]));
            check_attr_sizes(
                kdims, values["stride"].size(), "PARSE_CONV_TRANSPOSE: inconsistent strides");
        }
        if(contains(info.attributes, "dilations"))
        {
            values["dilation"].clear();
            copy(info.attributes["dilations"].ints(), std::back_inserter(values["dilation"]));
            check_attr_sizes(
                kdims, values["dilation"].size(), "PARSE_CONV_TRANSPOSE: inconsistent dilations");
        }
        if(contains(info.attributes, "auto_pad"))
        {
            auto s = info.attributes["auto_pad"].s();
            if(contains(info.attributes, "pads") and to_upper(s) != "NOTSET")
            {
                MIGRAPHX_THROW("PARSE_CONV_TRANSPOSE: auto_pad and padding cannot be specified "
                               "simultaneously");
            }

            if(s.find("SAME") != std::string::npos)
            {
                values["padding_mode"] = to_value(op::padding_mode_t::same);
            }
        }

        if(contains(info.attributes, "group"))
        {
            values["group"] = parse_value(info.attributes.at("group")).at<int>();
        }

        recalc_conv_attributes(values, kdims);

        op.from_value(values);
        auto l1                   = mm->add_instruction(op, l0, args[1]);
        std::vector<int64_t> dims = to_int64_vector(l1->get_shape().lens());
        std::vector<int64_t> curr_shape(dims.begin() + 2, dims.end());
        if(asym_padding)
        {
            std::vector<int64_t> axes(kdims);
            std::iota(axes.begin(), axes.end(), 2); // ignore first 2 dims

            auto pad_kdim_start = padding.begin() + kdims;
            std::vector<int64_t> starts(padding.begin(), pad_kdim_start);

            std::vector<int64_t> ends{};
            std::transform(curr_shape.begin(),
                           curr_shape.end(),
                           pad_kdim_start,
                           std::back_inserter(ends),
                           [](auto curr_dim, auto pad_dim) { return curr_dim - pad_dim; });

            l1 = mm->add_instruction(op::slice{axes, starts, ends}, l1);
        }

        if(contains(info.attributes, "output_padding"))
        {
            size_t non_kdims = dims.size() * 2 - kdims;
            std::vector<int64_t> output_padding(non_kdims, 0);
            copy(info.attributes["output_padding"].ints(), std::back_inserter(output_padding));
            check_attr_sizes(kdims,
                             output_padding.size() - non_kdims,
                             "PARSE_CONV_TRANSPOSE: inconsistent output padding");
            l1 = mm->add_instruction(op::pad{output_padding}, l1);
        }

        if(contains(info.attributes, "output_shape"))
        {
            std::vector<int64_t> output_shape;
            copy(info.attributes["output_shape"].ints(), std::back_inserter(output_shape));
            check_attr_sizes(
                kdims, output_shape.size(), "PARSE_CONV_TRANSPOSE: inconsistent output shape");
            dims = to_int64_vector(l1->get_shape().lens());
            copy(dims.begin() + 2, dims.end(), curr_shape.begin());
            if(curr_shape != output_shape)
            {
                std::vector<int64_t> target_padding(dims.size() * 2 - kdims, 0);
                std::transform(output_shape.begin(),
                               output_shape.end(),
                               curr_shape.begin(),
                               std::back_inserter(target_padding),
                               [](auto out_dim, auto curr_dim) { return out_dim - curr_dim; });
                l1 = mm->add_instruction(op::pad{target_padding}, l1);
            }
        }

        return add_bias(args, l1, 1);
    }

    static void
    tune_padding_to_symmetric(int64_t& left, int64_t& right, const int stride, int64_t& s_start)
    {
        s_start = 0;
        if(left > right)
        {
            right = left;
        }
        else if(left < right)
        {
            auto diff = right - left;
            s_start   = (diff + stride - 1) / stride;
            left      = left + s_start * stride;
            right     = left;
        }
    }

    static void tune_padding_size(const value& v,
                                  std::vector<int64_t>& padding,
                                  int count_include_pad,
                                  std::vector<int64_t>& s_start)
    {
        // maxpooling or count_include_pad is 1, no change is required.
        if(v.at("mode").to<std::string>() == "max" or count_include_pad == 1)
        {
            return;
        }

        // if padding is symmetric, return directly
        if(!is_asym_padding(padding))
        {
            return;
        }

        // asymmetric padding, make it symmetric
        std::size_t n_dims = padding.size() / 2;
        s_start.resize(n_dims);
        for(std::size_t i = 0; i < n_dims; ++i)
        {
            tune_padding_to_symmetric(
                padding[i], padding[i + n_dims], v.at("stride")[i].to<int64_t>(), s_start[i]);
        }
    }

    instruction_ref
    parse_pooling(const std::string& name, node_info info, std::vector<instruction_ref> args)
    {
        std::string mode = ends_with(name, "MaxPool") ? "max" : "average";
        operation op     = make_op("pooling", {{"mode", mode}});
        value values     = op.to_value();
        auto l0          = args[0];
        auto in_lens     = l0->get_shape().lens();
        assert(in_lens.size() > 2);
        auto kdims = in_lens.size() - 2;

        if(starts_with(name, "Global"))
        {
            values["lengths"] = std::vector<size_t>(in_lens.begin() + 2, in_lens.end());
        }

        // does not support ceil_mode
        if(contains(info.attributes, "ceil_mode"))
        {
            values["ceil_mode"] = static_cast<bool>(info.attributes.at("ceil_mode").i());
        }

        // count include padding, if count include pad is 1, we always use
        // explicit pad
        int count_include_pad = 0;
        if(contains(info.attributes, "count_include_pad"))
        {
            count_include_pad = info.attributes.at("count_include_pad").i();
        }

        if(contains(info.attributes, "strides"))
        {
            values["stride"].clear();
            copy(info.attributes["strides"].ints(), std::back_inserter(values["stride"]));
            check_attr_sizes(kdims, values["stride"].size(), "PARSE_POOLING: inconsistent strides");
        }
        if(contains(info.attributes, "kernel_shape"))
        {
            values["lengths"].clear();
            copy(info.attributes["kernel_shape"].ints(), std::back_inserter(values["lengths"]));
            check_attr_sizes(
                kdims, values["lengths"].size(), "PARSE_POOLING: inconsistent lengths");
        }

        // ensure pads availabe only when auto_pad is "NOT_SET"
        check_padding_mode(info, "POOLING");

        std::vector<int64_t> paddings;
        float pad_val = ((mode == "max") ? std::numeric_limits<float>::lowest() : 0.0f);
        if(contains(info.attributes, "pads"))
        {
            values["padding"].clear();
            copy(info.attributes["pads"].ints(), std::back_inserter(paddings));
            check_attr_sizes(
                kdims, paddings.size() / 2, "PARSE_POOLING: inconsistent explicit paddings");
        }

        if(contains(info.attributes, "auto_pad"))
        {
            values["padding"].clear();
            // return paddings could be empty, then setting to 0 for no padding
            cal_auto_padding_size(info,
                                  values,
                                  values["lengths"].to_vector<std::size_t>(),
                                  {1, 1},
                                  in_lens,
                                  paddings);
        }

        if(paddings.size() != 2 * kdims)
        {
            paddings.resize(kdims * 2);
            std::fill_n(paddings.begin(), 2 * kdims, 0);
        }

        if(values["padding"].size() != kdims)
        {
            values["padding"].resize(kdims);
            std::fill_n(values["padding"].begin(), kdims, 0);
        }

        if(values["stride"].size() != kdims)
        {
            values["stride"].resize(kdims);
            std::fill_n(values["stride"].begin(), kdims, 1);
        }
        // used to calculate the supposed output shape
        std::vector<int64_t> orig_padding(paddings.begin(), paddings.end());

        std::vector<int64_t> slice_start;
        std::vector<int64_t> slice_end;
        tune_padding_size(values, paddings, count_include_pad, slice_start);

        if(!slice_start.empty())
        {
            // calculate expected output shape
            orig_padding.insert(orig_padding.begin() + kdims, 2, 0);
            orig_padding.insert(orig_padding.begin(), 2, 0);
            op::pad pad{orig_padding, 0.0f};
            shape padded_shape = pad.compute_shape({l0->get_shape()});
            auto out_lens      = make_op("pooling", values).compute_shape({padded_shape}).lens();

            // compute slice_end information
            slice_end.resize(slice_start.size());
            std::transform(out_lens.begin() + 2,
                           out_lens.end(),
                           slice_start.begin(),
                           slice_end.begin(),
                           [](auto i, auto j) { return i + j; });
        }

        check_asym_padding(l0, paddings, values, count_include_pad, pad_val);
        in_lens = l0->get_shape().lens();
        for(size_t i = 0; i < kdims; i++)
        {
            if(values["lengths"][i].to<int64_t>() >
               in_lens[i + 2] + 2 * values["padding"][i].to<int64_t>())
            {
                MIGRAPHX_THROW("PARSE_POOLING: kernel shape is too large");
            }
        }
        op.from_value(values);
        auto l1 = mm->add_instruction(op, l0);
        if(!slice_start.empty())
        {
            std::vector<int64_t> axes(kdims);
            std::iota(axes.begin(), axes.end(), 2);
            l1 = mm->add_instruction(op::slice{axes, slice_start, slice_end}, l1);
        }

        return l1;
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

        return mm->add_instruction(op, make_contiguous(args[0]));
    }

    static const auto& get_nearest_op(const std::string& mode)
    {
        using nearest_op = std::function<std::size_t(std::size_t, double)>;
        static std::unordered_map<std::string, nearest_op> const nearest_ops = {
            {"round_prefer_floor",
             [=](std::size_t d_in, double val) {
                 val = std::max(0.0, std::min(d_in - 1.0, val));
                 return static_cast<std::size_t>(std::ceil((val - 0.5)));
             }},
            {"round_prefer_ceil",
             [=](std::size_t d_in, double val) {
                 val = std::max(0.0, std::min(d_in - 1.0, val));
                 return static_cast<std::size_t>(std::round((val)));
             }},
            {"floor",
             [=](std::size_t d_in, double val) {
                 val = std::max(0.0, std::min(d_in - 1.0, val));
                 return static_cast<std::size_t>(std::floor((val)));
             }},
            {"ceil", [=](std::size_t d_in, double val) {
                 val = std::max(0.0, std::min(d_in - 1.0, val));
                 return static_cast<std::size_t>(std::ceil((val)));
             }}};

        if(!contains(nearest_ops, mode))
        {
            MIGRAPHX_THROW("PARSE_RESIZE: nearest_mode " + mode + " not supported!");
        }

        return nearest_ops.at(mode);
    }

    static const auto& get_original_idx_op(const std::string& mode)
    {
        using original_idx_op =
            std::function<double(std::size_t, std::size_t, std::size_t, double)>;
        static std::unordered_map<std::string, original_idx_op> const idx_ops = {
            {"half_pixel",
             [=](std::size_t, std::size_t, std::size_t idx, double scale) {
                 return (idx + 0.5) / scale - 0.5;
             }},
            {"pytorch_half_pixel",
             [=](std::size_t, std::size_t l_out, std::size_t idx, double scale) {
                 return l_out > 1 ? (idx + 0.5) / scale - 0.5 : 0.0;
             }},
            {"align_corners",
             [=](std::size_t l_in, std::size_t l_out, std::size_t idx, double) {
                 return 1.0 * idx * (l_in - 1.0) / (l_out - 1.0);
             }},
            {"asymmetric",
             [=](std::size_t, std::size_t, std::size_t idx, double scale) { return idx / scale; }},
            {"tf_half_pixel_for_nn", [=](std::size_t, std::size_t, std::size_t idx, double scale) {
                 return (idx + 0.5) / scale;
             }}};

        if(!contains(idx_ops, mode))
        {
            MIGRAPHX_THROW("PARSE_RESIZE: coordinate_transformation_mode " + mode +
                           " not supported!");
        }

        return idx_ops.at(mode);
    }

    instruction_ref
    parse_resize(const std::string&, const node_info& info, std::vector<instruction_ref> args)
    {
        std::string coord_trans_mode = "half_pixel";
        if(contains(info.attributes, "coordinate_transformation_mode"))
        {
            coord_trans_mode = info.attributes.at("coordinate_transformation_mode").s();
            // does not support transformation mode "tf_crop_and_resize"
            if(coord_trans_mode == "tf_crop_and_resize")
            {
                MIGRAPHX_THROW("PARSE_RESIZE: \"tf_crop_and_resize\" mode is not supported!");
            }
        }

        // mode: only nearest mode is supported for now
        if(contains(info.attributes, "mode"))
        {
            auto mode = info.attributes.at("mode").s();
            if(mode != "nearest")
            {
                MIGRAPHX_THROW("PARSE_RESIZE: only nearest mode is supported!");
            }
        }

        // nearest mode
        std::string nearest_mode = "round_prefer_floor";
        if(contains(info.attributes, "nearest_mode"))
        {
            nearest_mode = info.attributes.at("nearest_mode").s();
        }

        // check exclude_outside, only support 0
        if(contains(info.attributes, "exclude_outside"))
        {
            int exclude_outside = info.attributes.at("exclude_outside").i();
            if(exclude_outside == 1)
            {
                MIGRAPHX_THROW("PARSE_RESIZE: exclude_outside 1 is not supported!");
            }
        }

        // input data shape info
        auto in_s    = args[0]->get_shape();
        auto in_lens = in_s.lens();

        // output shape is explicitly specified
        std::vector<std::size_t> out_lens(in_lens.size());

        // scale
        std::vector<double> vec_scale;

        // output size is specified in input, so use it as output size
        if(args.size() == 4 and args.back()->name() != "undefined")
        {
            auto arg_out_s = args[3]->eval();
            check_arg_empty(arg_out_s, "PARSE_RESIZE: dynamic output size is not supported!");
            arg_out_s.visit([&](auto ol) { out_lens.assign(ol.begin(), ol.end()); });

            if(out_lens.size() != in_lens.size())
            {
                MIGRAPHX_THROW("PARSE_RESIZE: specified output size does not match input size");
            }

            // compute the scale
            vec_scale.resize(in_lens.size());
            std::transform(in_lens.begin(),
                           in_lens.end(),
                           out_lens.begin(),
                           vec_scale.begin(),
                           [](auto iss, auto oss) { return 1.0 * oss / iss; });
        }
        // need to compute the output lens from input
        else
        {
            auto arg_scale = args[2]->eval();
            check_arg_empty(arg_scale, "PARSE_RESIZE: dynamic input scale is not supported!");

            arg_scale.visit([&](auto v) { vec_scale.assign(v.begin(), v.end()); });
            if(in_lens.size() != vec_scale.size())
            {
                MIGRAPHX_THROW("PARSE_RESIZE: ranks of input and scale are different!");
            }

            std::transform(
                in_lens.begin(),
                in_lens.end(),
                vec_scale.begin(),
                out_lens.begin(),
                [&](auto idx, auto scale) { return static_cast<std::size_t>(idx * scale); });
        }

        shape out_s{in_s.type(), out_lens};
        std::vector<int> ind(out_s.elements());

        // map out_idx to in_idx
        auto nearest_op = get_nearest_op(nearest_mode);
        auto idx_op     = get_original_idx_op(coord_trans_mode);

        shape_for_each(out_s, [&](auto idx) {
            auto in_idx = idx;
            for(auto ii = 0; ii < in_lens.size(); ++ii)
            {
                auto idx_val = idx_op(in_lens[ii], out_lens[ii], in_idx[ii], vec_scale[ii]);
                in_idx[ii]   = nearest_op(in_lens[ii], idx_val);
            }

            ind[out_s.index(idx)] = static_cast<int64_t>(in_s.index(in_idx));
        });

        // reshape input to one-dimension
        std::vector<int64_t> rsp_lens = {static_cast<int64_t>(in_s.elements())};
        shape ind_s{shape::int32_type, out_lens};
        auto rsp     = mm->add_instruction(make_op("reshape", {{"dims", rsp_lens}}), args[0]);
        auto ins_ind = mm->add_literal(literal(ind_s, ind));
        return mm->add_instruction(make_op("gather", {{"axis", 0}}), rsp, ins_ind);
    }

    instruction_ref
    parse_gather_elements(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        int axis = 0;
        if(contains(info.attributes, "axis"))
        {
            axis = parse_value(info.attributes.at("axis")).at<int>();
        }

        // standardize input data and index
        auto arg_data = make_contiguous(args[0]);
        auto arg_ind  = make_contiguous(args[1]);

        auto data_s = arg_data->get_shape();
        auto ind_s  = arg_ind->get_shape();

        if(data_s.lens().size() != ind_s.lens().size())
        {
            MIGRAPHX_THROW("PARSE_GATHER_ELEMENTS: input data and index must have the same rank!");
        }

        int n_rank     = static_cast<int>(data_s.lens().size());
        int tuned_axis = (axis < 0) ? (axis + n_rank) : axis;

        auto axis_stride      = data_s.strides()[tuned_axis];
        int64_t data_elem_num = static_cast<int64_t>(data_s.elements());
        // reshape the input data as one dimension and used as input data
        // to the gather operator
        arg_data = mm->add_instruction(op::reshape{{data_elem_num}}, arg_data);

        std::size_t elem_num = ind_s.elements();
        std::vector<int> ind_index(elem_num);
        std::iota(ind_index.begin(), ind_index.end(), 0);

        // convert index in input indices to that in input data
        std::vector<int> data_indices(elem_num);
        std::transform(ind_index.begin(), ind_index.end(), data_indices.begin(), [&](auto i) {
            return data_s.index(ind_s.multi(i));
        });

        std::vector<int> vec_axis_ind(elem_num);
        std::transform(ind_index.begin(), ind_index.end(), vec_axis_ind.begin(), [&](auto i) {
            return ind_s.multi(i)[tuned_axis];
        });

        auto l_shape_idx =
            mm->add_literal(literal(ind_s, data_indices.begin(), data_indices.end()));
        auto l_dim_idx = mm->add_literal(literal(ind_s, vec_axis_ind.begin(), vec_axis_ind.end()));
        auto l_stride  = mm->add_literal(literal{{ind_s.type(), {1}}, {axis_stride}});
        l_stride       = mm->add_instruction(op::multibroadcast{ind_s.lens()}, l_stride);
        auto dim_diff  = mm->add_instruction(make_op("sub"), arg_ind, l_dim_idx);
        auto delta     = mm->add_instruction(make_op("mul"), dim_diff, l_stride);
        auto ind       = mm->add_instruction(make_op("add"), l_shape_idx, delta);

        op::gather op{0};
        return mm->add_instruction(op, arg_data, ind);
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
            literal s = parse_value(info.attributes.at("ends"));
            s.visit([&](auto v) { copy(v, std::back_inserter(op.ends)); });
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

        if(op.axes.empty())
        {
            std::vector<int64_t> axes(args[0]->get_shape().lens().size());
            std::iota(axes.begin(), axes.end(), int64_t{0});
            op.axes = axes;
        }

        return mm->add_instruction(op, args[0]);
    }

    instruction_ref
    parse_constant(const std::string&, node_info info, const std::vector<instruction_ref>&) const
    {
        literal v = parse_value(info.attributes.at("value"));
        // return empty literal
        if(v.get_shape().elements() == 0)
        {
            return mm->add_literal(literal{});
        }

        auto dim_size = info.attributes.at("value").t().dims_size();
        // if dim_size is 0, it is a scalar
        if(dim_size == 0)
        {
            migraphx::shape scalar_shape{v.get_shape().type()};
            return mm->add_literal(migraphx::literal{scalar_shape, v.data()});
        }

        return mm->add_literal(v);
    }

    instruction_ref
    parse_gemm(const std::string&, node_info info, std::vector<instruction_ref> args) const
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

        auto l1 = (transa) ? mm->add_instruction(op::transpose{perm}, args[0]) : args[0];
        auto l2 = (transb) ? mm->add_instruction(op::transpose{perm}, args[1]) : args[1];
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
                    l3 = mm->add_instruction(op::multibroadcast{out_lens}, args[2]);
                }
                return mm->add_instruction(
                    make_op("dot", {{"alpha", alpha}, {"beta", beta}}), l1, l2, l3);
            }
        }

        return mm->add_instruction(make_op("dot", {{"alpha", alpha}, {"beta", beta}}), l1, l2);
    }

    instruction_ref parse_matmul(const std::string&,
                                 const std::string& op_name,
                                 const node_info&,
                                 std::vector<instruction_ref> args)
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
            l0 = mm->add_instruction(op::unsqueeze{{0}}, args[0]);
        }

        bool is_b_appended = false;
        if(l1_lens.size() == 1)
        {
            is_b_appended = true;
            l1_lens.push_back(1);
            l1 = mm->add_instruction(op::unsqueeze{{1}}, args[1]);
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
                bl0 = mm->add_instruction(op::multibroadcast{l0_broadcasted_lens}, l0);
            }
            if(l1_lens != l1_broadcasted_lens)
            {
                bl1 = mm->add_instruction(op::multibroadcast{l1_broadcasted_lens}, l1);
            }
        }

        auto dot_res = mm->add_instruction(make_op(op_name, {{"alpha", 1}, {"beta", 0}}), bl0, bl1);
        int64_t num_axis = static_cast<int64_t>(dot_res->get_shape().lens().size());
        if(is_a_prepended)
        {
            dot_res = mm->add_instruction(op::squeeze{{num_axis - 2}}, dot_res);
            --num_axis;
        }
        if(is_b_appended)
        {
            dot_res = mm->add_instruction(op::squeeze{{num_axis - 1}}, dot_res);
        }

        return dot_res;
    }

    instruction_ref
    parse_batchnorm(const std::string&, node_info info, std::vector<instruction_ref> args) const
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
        return mm->add_instruction(op, std::move(args));
    }

    instruction_ref
    parse_instancenorm(const std::string&, node_info info, std::vector<instruction_ref> args) const
    {
        // y = scale * ( x - mean ) / sqrt ( variance + epsilon ) + bias
        // mean = reduce_mean({D1, D2, ... Dk}, x)
        // variance = reduce_mean({D1, D2, ... Dk}, (x - mean)^2)

        float epsilon = 1e-5f;
        if(contains(info.attributes, "epsilon"))
        {
            epsilon = parse_value(info.attributes.at("epsilon")).at<float>();
        }
        auto x     = args[0];
        auto scale = args[1];
        auto bias  = args[2];
        auto dims  = x->get_shape().lens();
        auto ndims = dims.size();
        assert(ndims >= 2);
        auto kdims = ndims - 2;

        std::vector<int64_t> axes(kdims);
        std::iota(axes.begin(), axes.end(), 2);

        auto mean            = mm->add_instruction(make_op("reduce_mean", {{"axes", axes}}), x);
        auto mean_bcast      = mm->add_instruction(op::multibroadcast{dims}, mean);
        auto l0              = mm->add_instruction(make_op("sqdiff"), x, mean_bcast);
        auto variance        = mm->add_instruction(make_op("reduce_mean", {{"axes", axes}}), l0);
        auto l1              = mm->add_instruction(make_op("sub"), x, mean_bcast);
        auto epsilon_literal = mm->add_literal(epsilon);
        auto epsilon_bcast   = mm->add_instruction(op::multibroadcast{dims}, epsilon_literal);
        auto variance_bcast  = mm->add_instruction(op::multibroadcast{dims}, variance);
        auto l2              = mm->add_instruction(make_op("add"), variance_bcast, epsilon_bcast);
        auto l3              = mm->add_instruction(make_op("rsqrt"), l2);
        auto l4              = mm->add_instruction(make_op("mul"), l1, l3);
        auto scale_bcast     = mm->add_instruction(op::broadcast{1, dims}, scale);
        ;
        auto bias_bcast = mm->add_instruction(op::broadcast{1, dims}, bias);
        auto l5         = mm->add_instruction(make_op("mul"), l4, scale_bcast);
        return mm->add_instruction(make_op("add"), l5, bias_bcast);
    }

    instruction_ref
    parse_leaky_relu(const std::string&, node_info info, std::vector<instruction_ref> args) const
    {
        float alpha = 0.01; // default alpha val for leaky relu
        if(contains(info.attributes, "alpha"))
        {
            alpha = parse_value(info.attributes.at("alpha")).at<float>();
        }
        auto op = make_op("leaky_relu", {{"alpha", alpha}});
        return mm->add_instruction(op, args.front());
    }

    instruction_ref
    parse_elu(const std::string&, node_info info, std::vector<instruction_ref> args) const
    {
        float alpha = 1.0; // default alpha val for elu
        if(contains(info.attributes, "alpha"))
        {
            alpha = parse_value(info.attributes.at("alpha")).at<float>();
        }
        auto op = make_op("elu", {{"alpha", alpha}});
        return mm->add_instruction(op, args.front());
    }

    instruction_ref
    parse_lrn(const std::string&, node_info info, std::vector<instruction_ref> args) const
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
        return mm->add_instruction(op, args.front());
    }

    instruction_ref
    parse_imagescaler(const std::string&, node_info info, std::vector<instruction_ref> args) const
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

        auto scale_val = mm->add_literal(literal{shape{input_type}, {scale}});
        auto bias_vals = mm->add_literal(literal{shape{input_type, {bias.size()}}, bias});

        auto scale_tensor = mm->add_instruction(migraphx::op::scalar{input_lens}, scale_val);
        auto img_scaled = mm->add_instruction(migraphx::make_op("mul"), args.front(), scale_tensor);
        auto bias_bcast = mm->add_instruction(migraphx::op::broadcast{1, input_lens}, bias_vals);
        return mm->add_instruction(migraphx::make_op("add"), img_scaled, bias_bcast);
    }

    instruction_ref
    parse_transpose(const std::string&, node_info info, std::vector<instruction_ref> args) const
    {
        std::vector<int64_t> perm{};
        if(contains(info.attributes, "perm"))
        {
            auto&& perm_vals = info.attributes["perm"].ints();
            perm             = std::vector<int64_t>(perm_vals.begin(), perm_vals.end());
        }
        return mm->add_instruction(migraphx::op::transpose{perm}, args.front());
    }

    instruction_ref parse_pad(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        std::vector<int64_t> pads{};
        if(args.size() >= 2)
        {
            auto pad_arg = args.at(1)->eval();
            check_arg_empty(pad_arg, "PARSE_PAD: pad input must be constant");
            pad_arg.visit([&](auto v) { pads.assign(v.begin(), v.end()); });
        }
        else if(contains(info.attributes, "pads"))
        {
            auto&& pad_vals = info.attributes["pads"].ints();
            pads            = std::vector<int64_t>(pad_vals.begin(), pad_vals.end());
        }
        else
        {
            MIGRAPHX_THROW("PARSE_PAD: pad must be available");
        }

        // check if padding is actually being done (at least one value is nonzero)
        if(std::all_of(pads.begin(), pads.end(), [](const int& i) { return i == 0; }))
        {
            return mm->add_instruction(make_op("identity"), args.front());
        }

        if(contains(info.attributes, "mode"))
        {
            auto mode = info.attributes.at("mode").s();
            if(mode == "reflect")
                return reflect_pad(pads, args.front());
            if(mode != "constant")
            {
                MIGRAPHX_THROW(
                    "PARSE_PAD: migraphx currently only supports constant and reflect padding");
            }
        }

        float value = 0.0f;
        // third input is the value
        if(args.size() == 3)
        {
            auto val_ins = args.at(2);
            if(!val_ins->can_eval())
            {
                MIGRAPHX_THROW("PARSE_PAD: input value must be constant");
            }
            auto val_arg = val_ins->eval();
            if(val_arg.get_shape().elements() != 1)
            {
                MIGRAPHX_THROW("PARSE_PAD: value should contain only one element");
            }
            value = val_arg.at<float>();
        }
        else if(contains(info.attributes, "value"))
        {
            value = parse_value(info.attributes.at("value")).at<float>();
        }

        return mm->add_instruction(migraphx::op::pad{pads, value}, args.front());
    }

    instruction_ref
    parse_selu(const std::string&, const node_info& info, std::vector<instruction_ref> args) const
    {
        auto type   = args[0]->get_shape().type();
        auto lens   = args[0]->get_shape().lens();
        float alpha = 1.67326f;
        if(contains(info.attributes, "alpha"))
        {
            alpha = info.attributes.at("alpha").f();
        }

        float gamma = 1.0507f;
        if(contains(info.attributes, "gamma"))
        {
            gamma = info.attributes.at("gamma").f();
        }

        auto l_alpha = mm->add_literal({{type, {1}}, {alpha}});
        auto l_gamma = mm->add_literal({{type, {1}}, {gamma / 2.0f}});
        if(lens != std::vector<std::size_t>{1})
        {
            l_alpha =
                mm->add_instruction(make_op("multibroadcast", {{"output_lens", lens}}), l_alpha);
            l_gamma =
                mm->add_instruction(make_op("multibroadcast", {{"output_lens", lens}}), l_gamma);
        }

        auto sign_x = mm->add_instruction(make_op("sign"), args[0]);
        auto exp_x  = mm->add_instruction(make_op("exp"), args[0]);

        auto alpha_ex  = mm->add_instruction(make_op("mul"), l_alpha, exp_x);
        auto aex_alpha = mm->add_instruction(make_op("sub"), alpha_ex, l_alpha);

        auto ins1 = mm->add_instruction(make_op("add"), aex_alpha, args[0]);
        auto ins2 = mm->add_instruction(make_op("sub"), aex_alpha, args[0]);

        auto sign2   = mm->add_instruction(make_op("mul"), sign_x, ins2);
        auto ins_sub = mm->add_instruction(make_op("sub"), ins1, sign2);

        return mm->add_instruction(make_op("mul"), ins_sub, l_gamma);
    }

    // Use a literal instruction to replace the shape since, output of
    // shape operator are literals in migraphx
    instruction_ref
    parse_shape(const std::string&, const node_info&, std::vector<instruction_ref> args) const
    {
        if(args.size() != 1)
            MIGRAPHX_THROW("Shape: operator should have 1 operand");
        std::vector<std::size_t> arg_shape = args[0]->get_shape().lens();
        std::vector<int64_t> vec_shape(arg_shape.size());
        migraphx::shape s(migraphx::shape::int64_type, {arg_shape.size()});
        std::transform(arg_shape.begin(), arg_shape.end(), vec_shape.begin(), [](auto i) {
            return int64_t(i);
        });
        return mm->add_literal(migraphx::literal{s, vec_shape});
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
            return mm->add_literal(migraphx::literal(s, values));
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
            return mm->add_literal(migraphx::literal(s, values));
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

            return mm->add_literal(l_out);
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
        return mm->add_instruction(op::multibroadcast{out_lens}, args[0]);
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
            auto ins = mm->add_instruction(op::undefined{});
            args.insert(args.end(), (6 - args.size()), ins);
        }

        // first output for the concatenation of hidden states
        auto hidden_states =
            mm->add_instruction(op::rnn{hidden_size, vec_actv_funcs, dirct, clip}, std::move(args));

        // second output for the last hidden state
        auto last_output = mm->add_instruction(op::rnn_last_hs_output{}, hidden_states);

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
            auto ins = mm->add_instruction(op::undefined{});
            args.insert(args.end(), 6 - args.size(), ins);
        }

        // first output for concatenation of hidden states
        auto hidden_states = mm->add_instruction(
            op::gru{hidden_size, vec_actv_funcs, dirct, clip, linear_before_reset},
            std::move(args));

        // second output for last gru output
        auto last_output = mm->add_instruction(op::rnn_last_hs_output{}, hidden_states);

        return {hidden_states, last_output};
    }

    void lstm_actv_functions(op::rnn_direction dirct, std::vector<std::string>& actv_func_names)
    {
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
            switch(actv_func_names.size())
            {
            case 1:
                actv_func_names = {actv_func_names.at(0),
                                   actv_func_names.at(0),
                                   actv_func_names.at(0),
                                   actv_func_names.at(0),
                                   actv_func_names.at(0),
                                   actv_func_names.at(0)};
                break;

            case 2:
                // repeat the 2nd actv func once, then repeat all three another time
                actv_func_names = {actv_func_names.at(0),
                                   actv_func_names.at(1),
                                   actv_func_names.at(1),
                                   actv_func_names.at(0),
                                   actv_func_names.at(1),
                                   actv_func_names.at(1)};
                break;

            case 3:
                // repeat all three actv funcs once
                actv_func_names = {actv_func_names.at(0),
                                   actv_func_names.at(1),
                                   actv_func_names.at(2),
                                   actv_func_names.at(0),
                                   actv_func_names.at(1),
                                   actv_func_names.at(2)};
                break;

            case 4:
                actv_func_names = {actv_func_names.at(0),
                                   actv_func_names.at(1),
                                   actv_func_names.at(2),
                                   actv_func_names.at(3),
                                   actv_func_names.at(3),
                                   actv_func_names.at(3)};
                break;

            case 5:
                actv_func_names = {actv_func_names.at(0),
                                   actv_func_names.at(1),
                                   actv_func_names.at(2),
                                   actv_func_names.at(3),
                                   actv_func_names.at(4),
                                   actv_func_names.at(4)};
                break;

            default: break;
            }
        }
        else
        {
            switch(actv_func_names.size())
            {
            case 1:
                actv_func_names = {
                    actv_func_names.at(0), actv_func_names.at(0), actv_func_names.at(0)};
                break;

            case 2:
                // repeat the 2nd actv func once, so we have 3 actv funcs
                actv_func_names = {
                    actv_func_names.at(0), actv_func_names.at(1), actv_func_names.at(1)};
                break;

            default: break;
            }
        }
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

        lstm_actv_functions(dirct, vec_names);

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
            auto ins = mm->add_instruction(op::undefined{});
            args.insert(args.end(), 8 - args.size(), ins);
        }

        // first output for concatenation of hidden states
        auto hidden_states = mm->add_instruction(
            op::lstm{hidden_size, vec_actv_funcs, dirct, clip, input_forget}, std::move(args));

        auto last_output = mm->add_instruction(op::rnn_last_hs_output{}, hidden_states);

        // third output for last cell output
        auto last_cell_output = mm->add_instruction(op::rnn_last_cell_output{}, hidden_states);

        return {hidden_states, last_output, last_cell_output};
    }

    instruction_ref parse_reduce_oper(const std::string&,
                                      const std::string& op_name,
                                      node_info info,
                                      std::vector<instruction_ref> args) const
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
            return mm->add_instruction(make_op(op_name, {{"axes", axes}}), std::move(args));
        }
        else
        {
            auto ins = mm->add_instruction(make_op(op_name, {{"axes", axes}}), std::move(args));
            return mm->add_instruction(op::squeeze{axes}, ins);
        }
    }

    instruction_ref
    parse_reduce_l1(const std::string&, node_info info, std::vector<instruction_ref> args) const
    {
        auto abs_ins = mm->add_instruction(make_op("abs"), args[0]);
        return parse_reduce_oper({}, "reduce_sum", std::move(info), {abs_ins});
    }

    instruction_ref
    parse_reduce_l2(const std::string&, node_info info, std::vector<instruction_ref> args) const
    {
        auto square_ins = mm->add_instruction(make_op("mul"), args[0], args[0]);
        auto sum_ins    = parse_reduce_oper({}, "reduce_sum", std::move(info), {square_ins});
        return mm->add_instruction(make_op("sqrt"), sum_ins);
    }

    instruction_ref parse_reduce_log_sum(const std::string&,
                                         node_info info,
                                         std::vector<instruction_ref> args) const
    {
        auto sum_ins = parse_reduce_oper({}, "reduce_sum", std::move(info), std::move(args));
        return mm->add_instruction(make_op("log"), sum_ins);
    }

    instruction_ref parse_reduce_log_sum_exp(const std::string&,
                                             node_info info,
                                             std::vector<instruction_ref> args) const
    {
        auto exp_ins = mm->add_instruction(make_op("exp"), args[0]);
        auto sum_ins = parse_reduce_oper({}, "reduce_sum", std::move(info), {exp_ins});
        return mm->add_instruction(make_op("log"), sum_ins);
    }

    instruction_ref parse_reduce_sum_square(const std::string&,
                                            node_info info,
                                            std::vector<instruction_ref> args) const
    {
        auto square_ins = mm->add_instruction(make_op("mul"), args[0], args[0]);
        return parse_reduce_oper({}, "reduce_sum", std::move(info), {square_ins});
    }

    instruction_ref
    parse_cast(const std::string&, node_info info, std::vector<instruction_ref> args) const
    {
        if(!contains(info.attributes, "to"))
        {
            MIGRAPHX_THROW("PARSE_CAST: missing to type attribute!");
        }

        int to_type        = parse_value(info.attributes.at("to")).at<int>();
        shape::type_t type = get_type(to_type);
        return mm->add_instruction(make_op("convert", {{"target_type", type}}), std::move(args));
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
                mm->add_instruction(op::slice{{axis}, {start}, {start + sl}}, args[0]));
            start += sl;
        }

        return ret_ins;
    }

    instruction_ref
    parse_onehot(const std::string&, node_info info, std::vector<instruction_ref> args)
    {
        migraphx::argument depth_arg = args[1]->eval();
        check_arg_empty(depth_arg, "PARSE_ONEHOT: depth - dynamic shape not supported");
        size_t depth = depth_arg.at<size_t>();

        int64_t axis = -1;
        if(contains(info.attributes, "axis"))
        {
            axis = info.attributes.at("axis").i();
        }

        std::vector<float> depth_input(depth * depth, 0.0f);
        for(int i = 0; i < depth; i++)
        {
            depth_input[depth * i + i] = 1.0f;
        }

        auto type = args[2]->get_shape().type();
        shape s{type, {depth, depth}};
        auto l_val      = mm->add_literal({s, depth_input});
        auto gather_out = mm->add_instruction(op::gather{0}, {l_val, args[0]});

        // Finally, we need a transpose to move the inner most dim to the axis dim
        int n_rank = gather_out->get_shape().lens().size();
        if(axis < -n_rank or axis >= n_rank)
        {
            MIGRAPHX_THROW("PARSE_ONEHOT: axis out of range");
        }
        int64_t tuned_axis = (axis < 0) ? axis + n_rank : axis;
        std::vector<int64_t> perm(n_rank - 1);
        std::iota(perm.begin(), perm.end(), 0);
        perm.insert(perm.begin() + tuned_axis, n_rank - 1);
        auto tr_out = mm->add_instruction(op::transpose{perm}, gather_out);
        auto lens   = tr_out->get_shape().lens();

        auto off_val       = mm->add_instruction(op::slice{{0}, {0}, {1}}, args[2]);
        auto on_val        = mm->add_instruction(op::slice{{0}, {1}, {2}}, args[2]);
        auto diff          = mm->add_instruction(make_op("sub"), on_val, off_val);
        auto unsq_off_val  = mm->add_instruction(op::multibroadcast{lens}, off_val);
        auto unsq_diff_val = mm->add_instruction(op::multibroadcast{lens}, diff);
        auto l_mul         = mm->add_instruction(make_op("mul"), tr_out, unsq_diff_val);
        return mm->add_instruction(make_op("add"), l_mul, unsq_off_val);
    }

    instruction_ref
    parse_tile(const std::string&, const node_info&, std::vector<instruction_ref> args)
    {
        migraphx::argument arg_s = args[1]->eval();
        check_arg_empty(arg_s, "PARSE_TILE: dynamic shape is not supported");
        std::vector<std::int64_t> repeats;
        arg_s.visit([&](auto input) { repeats.assign(input.begin(), input.end()); });

        auto l0 = args[0];
        for(int i = 0; i < repeats.size(); i++)
        {
            auto l1 = l0;
            for(int j = 1; j < repeats[i]; j++)
            {
                l0 = mm->add_instruction(op::concat{i}, l0, l1);
            }
        }
        return l0;
    }

    instruction_ref
    parse_range(const std::string&, const node_info&, std::vector<instruction_ref> args)
    {

        auto start_arg = args[0]->eval();
        check_arg_empty(start_arg, "PARSE_RANGE: start arg dynamic shape is not supported");
        auto limit_arg = args[1]->eval();
        check_arg_empty(limit_arg, "PARSE_RANGE: limit arg dynamic shape is not supported");
        auto delta_arg = args[2]->eval();
        check_arg_empty(delta_arg, "PARSE_RANGE: delta arg dynamic shape is not supported");

        assert(args[0]->get_shape().elements() == 1 and args[1]->get_shape().elements() == 1 and
               args[2]->get_shape().elements() == 1);

        instruction_ref l0;

        visit_all(start_arg, limit_arg, delta_arg)([&](auto start, auto limit, auto delta) {
            auto start_val = start.front();
            auto limit_val = limit.front();
            auto delta_val = delta.front();

            size_t num_elements = static_cast<size_t>(
                ceil(static_cast<double>(limit_val - start_val) / static_cast<double>(delta_val)));

            assert(num_elements > 0);

            using type = decltype(start_val);

            std::vector<type> range_vals(num_elements);

            std::generate(range_vals.begin(), range_vals.end(), [&]() {
                auto result = start_val;
                start_val += delta_val;
                return result;
            });

            l0 = mm->add_literal({shape{args[0]->get_shape().type(), {num_elements}}, range_vals});
        });
        return l0;
    }

    enum class reduce_mode_t
    {
        sum  = 0,
        mean = 1,
        max  = 2
    };

    instruction_ref parse_embedding_bag(const node_info& info,
                                        std::vector<instruction_ref> args) const
    {
        if(args[2]->get_shape().elements() != 1)
            MIGRAPHX_THROW("PARSE_EMBEDDING_BAG: MIGraphX only supports offsets of size 1");
        reduce_mode_t reduce_mode = reduce_mode_t::sum;
        if(contains(info.attributes, "mode"))
        {
            reduce_mode = static_cast<reduce_mode_t>(info.attributes.at("mode").i());
        }

        auto l0 = mm->add_instruction(op::gather{}, args[0], args[1]);
        switch(reduce_mode)
        {
        case reduce_mode_t::sum:
            l0 = mm->add_instruction(make_op("reduce_sum", {{"axes", {0}}}), l0);
            break;
        case reduce_mode_t::mean:
            l0 = mm->add_instruction(make_op("reduce_mean", {{"axes", {0}}}), l0);
            break;
        case reduce_mode_t::max:
            l0 = mm->add_instruction(make_op("reduce_max", {{"axes", {0}}}), l0);
            break;
        }
        return l0;
    }

    instruction_ref
    parse_aten(const std::string&, const node_info& info, std::vector<instruction_ref> args) const
    {
        if(contains(info.attributes, "operator"))
        {
            auto op_name = info.attributes.at("operator").s();
            if(op_name.find("embedding_bag") != std::string::npos)
            {
                return parse_embedding_bag(info, std::move(args));
            }
        }
        MIGRAPHX_THROW("PARSE_ATEN: unsupported custom operator");
    }

    std::vector<instruction_ref>
    parse_dropout(const std::string&, const node_info&, std::vector<instruction_ref> args) const
    {
        auto out = mm->add_instruction(make_op("identity"), args[0]);
        auto s   = args[0]->get_shape();
        std::vector<int8_t> vec(s.elements(), 1);
        shape mask_s{shape::bool_type, s.lens()};
        auto mask = mm->add_literal(literal(mask_s, vec));

        return {out, mask};
    }

    template <class T>
    std::vector<std::size_t> nonzero_indices(const std::vector<T>& data)
    {
        std::vector<std::size_t> indices;
        for(std::size_t i = 0; i < data.size(); ++i)
        {
            if(!float_equal(data[i], 0))
                indices.push_back(i);
        }

        return indices;
    }

    instruction_ref
    parse_nonzero(const std::string&, const node_info&, std::vector<instruction_ref> args)
    {
        migraphx::argument data_arg = args.back()->eval();
        check_arg_empty(data_arg, "PARSE_NONZERO: cannot support non-constant input!");

        std::vector<std::size_t> indices;
        data_arg.visit([&](auto val) {
            using val_type = std::remove_cv_t<typename decltype(val)::value_type>;
            std::vector<val_type> vec_data;
            vec_data.assign(val.begin(), val.end());
            indices = this->nonzero_indices(vec_data);
        });

        shape in_s = args[0]->get_shape();
        shape out_s{shape::int64_type, {in_s.lens().size(), indices.size()}};

        std::vector<int64_t> out_data(out_s.elements());
        for(std::size_t i = 0; i < indices.size(); ++i)
        {
            auto idx = in_s.multi(indices[i]);
            for(std::size_t j = 0; j < in_s.lens().size(); ++j)
            {
                out_data[out_s.index({j, i})] = idx[j];
            }
        }

        return mm->add_literal(literal(out_s, out_data));
    }

    instruction_ref parse_compare_op(const std::string&,
                                     const std::string& op_name,
                                     const node_info&,
                                     std::vector<instruction_ref> args)
    {
        auto l = add_broadcastable_binary_op(args[0], args[1], op_name);
        if(l->get_shape().type() != shape::bool_type)
        {
            l = mm->add_instruction(make_op("convert", {{"target_type", shape::bool_type}}), l);
        }
        return l;
    }

    instruction_ref
    parse_upsample(const std::string&, const node_info& info, std::vector<instruction_ref> args)
    {
        if(contains(info.attributes, "mode"))
        {
            auto mode = info.attributes.at("mode").s();
            if(mode != "nearest")
            {
                MIGRAPHX_THROW("PARSE_UPSAMPLE: only nearest mode is supported!");
            }
        }

        auto arg_scale = args[1]->eval();
        check_arg_empty(arg_scale, "PARSE_UPSAMPLE: only constant scale is supported!");
        std::vector<float> vec_scale;
        arg_scale.visit([&](auto v) { vec_scale.assign(v.begin(), v.end()); });

        auto in_s    = args[0]->get_shape();
        auto in_lens = in_s.lens();
        if(in_lens.size() != vec_scale.size())
        {
            MIGRAPHX_THROW("PARSE_UPSAMPLE: ranks of input and scale are different!");
        }

        std::vector<std::size_t> out_lens(in_lens.size());
        std::transform(in_lens.begin(),
                       in_lens.end(),
                       vec_scale.begin(),
                       out_lens.begin(),
                       [&](auto idx, auto scale) { return static_cast<std::size_t>(idx * scale); });

        std::vector<float> idx_scale(in_lens.size());
        std::transform(
            out_lens.begin(),
            out_lens.end(),
            in_lens.begin(),
            idx_scale.begin(),
            [](auto od, auto id) { return (od == id) ? 1.0f : (id - 1.0f) / (od - 1.0f); });

        shape out_s{in_s.type(), out_lens};
        std::vector<int> ind(out_s.elements());

        // map out_idx to in_idx
        shape_for_each(out_s, [&](auto idx) {
            auto in_idx = idx;
            std::transform(idx.begin(),
                           idx.end(),
                           idx_scale.begin(),
                           in_idx.begin(),
                           // nearest mode
                           [](auto index, auto scale) {
                               return static_cast<std::size_t>(std::round(index * scale));
                           });

            ind[out_s.index(idx)] = static_cast<int64_t>(in_s.index(in_idx));
        });

        // reshape input to one-dimension
        std::vector<int64_t> rsp_lens = {static_cast<int64_t>(in_s.elements())};
        shape ind_s{shape::int32_type, out_lens};
        auto rsp     = mm->add_instruction(make_op("reshape", {{"dims", rsp_lens}}), args[0]);
        auto ins_ind = mm->add_literal(literal(ind_s, ind));
        return mm->add_instruction(make_op("gather", {{"axis", 0}}), rsp, ins_ind);
    }

    instruction_ref
    parse_where(const std::string&, const node_info&, std::vector<instruction_ref> args)
    {
        auto cond =
            mm->add_instruction(make_op("convert", {{"target_type", shape::int32_type}}), args[0]);
        auto lens = compute_broadcasted_lens(cond->get_shape().lens(), args[1]->get_shape().lens());
        lens      = compute_broadcasted_lens(lens, args[2]->get_shape().lens());
        if(cond->get_shape().lens() != lens)
        {
            cond = mm->add_instruction(make_op("multibroadcast", {{"output_lens", lens}}), cond);
        }

        if(args[1]->get_shape().lens() != lens)
        {
            args[1] =
                mm->add_instruction(make_op("multibroadcast", {{"output_lens", lens}}), args[1]);
        }

        if(args[2]->get_shape().lens() != lens)
        {
            args[2] =
                mm->add_instruction(make_op("multibroadcast", {{"output_lens", lens}}), args[2]);
        }

        // compute index
        auto elem_num = args[1]->get_shape().elements();

        // concatenation of input data
        auto concat_data = mm->add_instruction(make_op("concat", {{"axis", 0}}), args[2], args[1]);
        std::vector<int64_t> dims = {static_cast<int64_t>(2 * elem_num)};
        auto rsp_data = mm->add_instruction(make_op("reshape", {{"dims", dims}}), concat_data);

        std::vector<int> ind(elem_num);
        std::iota(ind.begin(), ind.end(), 0);
        shape ind_s{shape::int32_type, lens};
        auto l_ind = mm->add_literal(literal(ind_s, ind));
        std::vector<int> offset(elem_num, elem_num);
        auto l_offset   = mm->add_literal(literal({shape::int32_type, lens}, offset));
        auto ins_offset = mm->add_instruction(make_op("mul"), l_offset, cond);
        auto ins_ind    = mm->add_instruction(make_op("add"), ins_offset, l_ind);

        return mm->add_instruction(make_op("gather", {{"axis", 0}}), rsp_data, ins_ind);
    }

    void parse_from(std::istream& is, std::string name = "")
    {
        this->filename   = std::move(name);
        auto parent_path = fs::path(this->filename).parent_path();
        if(not parent_path.empty())
            this->path = parent_path;

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
        for(auto&& f : graph.initializer())
        {
            instructions[f.name()] = mm->add_literal(parse_tensor(f));
        }

        for(auto&& input : graph.input())
        {
            const std::string& name = input.name();
            // input not in initializer_data, so it is a real input
            if(!contains(instructions, name))
            {
                std::vector<std::size_t> dims;
                if(map_input_dims.count(name) > 0)
                {
                    dims = map_input_dims.at(name);
                }

                shape s            = parse_type(input.type(), dims);
                instructions[name] = mm->add_parameter(name, s);
            }
        }

        for(auto&& node : graph.node())
        {
            std::vector<instruction_ref> args;
            for(auto&& input : node.input())
            {
                if(input.empty())
                {
                    this->parse_undefined(input);
                }
                if(instructions.count(input) == 0)
                {
                    MIGRAPHX_THROW("PARSE_GRAPH: invalid onnx file. Input \"" + input +
                                   "\" is unavailable due to unordered nodes!");
                }
                args.push_back(instructions.at(input));
            }

            std::vector<instruction_ref> result;
            std::size_t output_num = static_cast<std::size_t>(node.output().size());
            if(ops.count(node.op_type()) == 0)
            {
                if(skip_unknown_operators)
                    result.push_back(mm->add_instruction(op::unknown{node.op_type()}, args));
                else
                    MIGRAPHX_THROW("Unknown operator: " + node.op_type());
            }
            else
            {
                result = ops[node.op_type()]({get_attributes(node), output_num}, args);
            }

            output_num = std::min<std::size_t>(output_num, result.size());
            std::transform(node.output().begin(),
                           node.output().begin() + output_num,
                           result.begin(),
                           std::inserter(instructions, instructions.end()),
                           [](auto&& x, auto&& y) { return std::make_pair(x, y); });
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
        mm->add_return(output_ins);
    }

    void parse_undefined(const std::string& name)
    {
        if(!contains(instructions, name))
        {
            auto ins           = mm->add_instruction(op::undefined{});
            instructions[name] = ins;
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

    static shape::type_t get_type(int dtype)
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
        case 9: return shape::bool_type;
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

    template <class T>
    static literal from_repeated(shape::type_t t, const T& r)
    {
        std::size_t size = r.size();
        return literal{{t, {size}}, r.begin(), r.end()};
    }

    literal parse_value(const onnx::AttributeProto& attr) const
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
        MIGRAPHX_THROW("PARSE_VALUE: Invalid attribute type " + std::to_string(attr.type()));
    }

    literal parse_tensor(const onnx::TensorProto& t) const
    {
        std::vector<std::size_t> dims(t.dims().begin(), t.dims().end());
        if(not t.external_data().empty())
        {
            const std::string& data_file = t.external_data().at(0).value();
            auto raw_buffer              = read_buffer(path + "/" + data_file);
            std::string s(raw_buffer.begin(), raw_buffer.end());
            auto type = get_type(t.data_type());
            return create_literal(type, dims, s.data());
        }
        if(t.has_raw_data())
        {
            const std::string& s = t.raw_data();
            auto type            = get_type(t.data_type());
            return create_literal(type, dims, s.data());
        }

        switch(t.data_type())
        {
        case onnx::TensorProto::BOOL: return create_literal(shape::bool_type, dims, t.int32_data());
        case onnx::TensorProto::INT8: return create_literal(shape::int8_type, dims, t.int32_data());
        case onnx::TensorProto::UINT8:
            return create_literal(shape::uint8_type, dims, t.int32_data());
        case onnx::TensorProto::INT16:
            return create_literal(shape::int16_type, dims, t.int32_data());
        case onnx::TensorProto::UINT16:
            return create_literal(shape::uint16_type, dims, t.int32_data());
        case onnx::TensorProto::INT32:
            return create_literal(shape::int32_type, dims, t.int32_data());
        case onnx::TensorProto::UINT32:
            return create_literal(shape::uint32_type, dims, t.uint64_data());
        case onnx::TensorProto::INT64:
            return create_literal(shape::int64_type, dims, t.int64_data());
        case onnx::TensorProto::UINT64:
            return create_literal(shape::uint64_type, dims, t.uint64_data());
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
        case onnx::TensorProto::DOUBLE:
            return create_literal(shape::double_type, dims, t.double_data());
        case onnx::TensorProto::FLOAT:
            return create_literal(shape::float_type, dims, t.float_data());
        case onnx::TensorProto::UNDEFINED:
        case onnx::TensorProto::STRING:
        case onnx::TensorProto::COMPLEX64:
        case onnx::TensorProto::COMPLEX128: throw std::runtime_error("");
        }
        MIGRAPHX_THROW("PARSE_TENSOR: Invalid tensor type");
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

    shape parse_type(const onnx::TypeProto& t, const std::vector<std::size_t>& input_dims) const
    {
        shape::type_t shape_type = get_type(t.tensor_type().elem_type());
        if(!input_dims.empty())
        {
            return {shape_type, input_dims};
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
                               {
                                   return default_dim_value;
                               }
                               return d.dim_value();
                           }
                           else
                           {
                               return default_dim_value;
                           }
                       });

        if(dims.empty())
            return {shape_type};

        return {shape_type, dims};
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
program parse_onnx_from(const onnx_options& options, Ts&&... xs)
{
    onnx_parser parser;
    parser.map_input_dims         = options.map_input_dims;
    parser.default_dim_value      = options.default_dim_value;
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

program parse_onnx(const std::string& name, const onnx_options& options)
{
    std::fstream input(name.c_str(), std::ios::in | std::ios::binary);
    return parse_onnx_from(options, input, name);
}

program parse_onnx_buffer(const std::string& buffer, const onnx_options& options)
{
    return parse_onnx_from(options, buffer.data(), buffer.size());
}

program parse_onnx_buffer(const void* data, std::size_t size, const onnx_options& options)
{
    return parse_onnx_from(options, data, size);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
