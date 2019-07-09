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

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct onnx_parser
{
    using attribute_map = std::unordered_map<std::string, onnx::AttributeProto>;
    using node_map      = std::unordered_map<std::string, onnx::NodeProto>;
    using op_func =
        std::function<std::vector<instruction_ref>(attribute_map, std::vector<instruction_ref>)>;
    node_map nodes;
    std::unordered_map<std::string, instruction_ref> instructions;
    program prog    = program();
    bool is_pytorch = false;

    std::unordered_map<std::string, op_func> ops;
    std::unordered_map<std::string, operation> map_actv_funcs;

    onnx_parser()
    {
        add_generic_op("Relu", op::relu{});
        add_generic_op("Sigmoid", op::sigmoid{});
        add_generic_op("Abs", op::abs{});
        add_generic_op("Exp", op::exp{});
        add_generic_op("Erf", op::erf{});
        add_generic_op("Log", op::log{});
        // disable dropout for inference
        add_generic_op("Dropout", op::identity{});
        add_generic_op("Identity", op::identity{});
        add_generic_op("Sin", op::sin{});
        add_generic_op("Cos", op::cos{});
        add_generic_op("Tan", op::tan{});
        add_generic_op("Sinh", op::sinh{});
        add_generic_op("Cosh", op::cosh{});
        add_generic_op("Tanh", op::tanh{});
        add_generic_op("Asin", op::asin{});
        add_generic_op("Acos", op::acos{});
        add_generic_op("Atan", op::atan{});

        add_binary_op("Add", op::add{});
        add_binary_op("Div", op::div{});
        add_binary_op("Mul", op::mul{});
        add_binary_op("Sub", op::sub{});
        add_binary_op("Pow", op::pow{});

        add_variadic_op("Sum", op::add{});
        add_variadic_op("Max", op::max{});
        add_variadic_op("Min", op::min{});

        add_mem_op("ArgMax", &onnx_parser::parse_argmax);
        add_mem_op("ArgMin", &onnx_parser::parse_argmin);
        add_mem_op("Cast", &onnx_parser::parse_cast);
        add_mem_op("Clip", &onnx_parser::parse_clip);
        add_mem_op("LRN", &onnx_parser::parse_lrn);
        add_mem_op("ImageScaler", &onnx_parser::parse_imagescaler);
        add_mem_op("LeakyRelu", &onnx_parser::parse_leaky_relu);
        add_mem_op("Elu", &onnx_parser::parse_elu);
        add_mem_op("Expand", &onnx_parser::parse_expand);
        add_mem_op("Constant", &onnx_parser::parse_constant);
        add_mem_op("Conv", &onnx_parser::parse_conv);
        add_mem_op("MaxPool", &onnx_parser::parse_pooling);
        add_mem_op("AveragePool", &onnx_parser::parse_pooling);
        add_mem_op("GlobalMaxPool", &onnx_parser::parse_pooling);
        add_mem_op("GlobalAveragePool", &onnx_parser::parse_pooling);
        add_mem_op("Reshape", &onnx_parser::parse_reshape);
        add_mem_op("Flatten", &onnx_parser::parse_flatten);
        add_mem_op("Gemm", &onnx_parser::parse_gemm);
        add_mem_op("MatMul", &onnx_parser::parse_matmul);
        add_mem_op("BatchNormalization", &onnx_parser::parse_batchnorm);
        add_mem_op("Softmax", &onnx_parser::parse_softmax);
        add_mem_op("LogSoftmax", &onnx_parser::parse_logsoftmax);
        add_mem_op("Squeeze", &onnx_parser::parse_squeeze);
        add_mem_op("Unsqueeze", &onnx_parser::parse_unsqueeze);
        add_mem_op("Slice", &onnx_parser::parse_slice);
        add_mem_op("Concat", &onnx_parser::parse_concat);
        add_mem_op("Gather", &onnx_parser::parse_gather);
        add_mem_op("Shape", &onnx_parser::parse_shape);
        add_mem_op("ConstantFill", &onnx_parser::parse_constant_fill);
        add_mem_op("ConstantOfShape", &onnx_parser::parse_constant_of_shape);
        add_mem_op("Transpose", &onnx_parser::parse_transpose);
        add_mem_op("RNN", &onnx_parser::parse_rnn);
        add_mem_op("GRU", &onnx_parser::parse_gru);
        add_mem_op("LSTM", &onnx_parser::parse_lstm);
        add_mem_op("Pad", &onnx_parser::parse_pad);
        add_mem_op("ReduceSum", &onnx_parser::parse_reduce_oper<op::reduce_sum>);
        add_mem_op("ReduceMean", &onnx_parser::parse_reduce_oper<op::reduce_mean>);

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
        add_op(name, [this, x](attribute_map attributes, std::vector<instruction_ref> args) {
            if(args.size() != 2)
                MIGRAPHX_THROW("binary operators should have 2 operands");
            if(contains(attributes, "broadcast") and contains(attributes, "axis"))
            {
                uint64_t broadcasted = parse_value(attributes.at("broadcast")).at<uint64_t>();
                if(broadcasted != 0)
                {
                    uint64_t axis = parse_value(attributes.at("axis")).at<uint64_t>();
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

    template <class T>
    instruction_ref add_broadcastable_binary_op(instruction_ref arg0, instruction_ref arg1, T x)
    {
        if(arg0->get_shape().lens() != arg1->get_shape().lens())
        {
            // Get lengths for both arguments
            auto s0       = arg0->get_shape().lens();
            auto s1       = arg1->get_shape().lens();
            auto out_lens = compute_broadcasted_lens(s0, s1);
            auto l0       = prog.add_instruction(op::multibroadcast{out_lens}, arg0);
            auto l1       = prog.add_instruction(op::multibroadcast{out_lens}, arg1);
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
        add_op(name, [this, x](const attribute_map&, std::vector<instruction_ref> args) {
            return prog.add_instruction(x, args);
        });
    }

    template <class T>
    void add_variadic_op(std::string name, T x)
    {
        add_op(name, [this, x](const attribute_map&, std::vector<instruction_ref> args) {
            return std::accumulate(std::next(args.begin()),
                                   args.end(),
                                   args.front(),
                                   [this, x](instruction_ref a, instruction_ref b) {
                                       return add_broadcastable_binary_op(a, b, x);
                                   });
        });
    }

    instruction_ref parse_clip(const std::string&,
                               const attribute_map& attributes,
                               std::vector<instruction_ref> args)
    {
        op::clip op;
        if(contains(attributes, "max"))
        {
            op.max_val = parse_value(attributes.at("max")).at<float>();
        }
        if(contains(attributes, "min"))
        {
            op.min_val = parse_value(attributes.at("min")).at<float>();
        }
        return prog.add_instruction(op, std::move(args));
    }

//    instruction_ref
//    parse_softmax(const std::string&, const attribute_map&, std::vector<instruction_ref> args)
//    {
//        auto dims = args.front()->get_shape().lens();
//        auto r =
//            prog.add_instruction(op::reshape{{long(dims[0]), long(dims[1]), 1, 1}}, args.front());
//        auto s = prog.add_instruction(op::softmax{}, r);
//        return prog.add_instruction(op::reshape{{long(dims[0]), long(dims[1])}}, s);
//    }

    instruction_ref parse_softmax(const std::string&,
                                     const attribute_map& attributes,
                                     std::vector<instruction_ref> args)
    {
        int axis = 1;
        if(contains(attributes, "axis"))
        {
            axis = parse_value(attributes.at("axis")).at<int>();
        }

        return prog.add_instruction(op::softmax{axis}, std::move(args));
    }

    instruction_ref parse_logsoftmax(const std::string&,
                                     const attribute_map& attributes,
                                     std::vector<instruction_ref> args)
    {
        int axis = 1;
        if(contains(attributes, "axis"))
        {
            axis = parse_value(attributes.at("axis")).at<int>();
        }

        return prog.add_instruction(op::logsoftmax{axis}, std::move(args));
    }

    instruction_ref parse_argmax(const std::string&,
                                 const attribute_map& attributes,
                                 std::vector<instruction_ref> args)
    {
        int64_t axis = 0;
        if(contains(attributes, "axis"))
        {
            axis = static_cast<int64_t>(parse_value(attributes.at("axis")).at<int>());
        }

        int keep_dims = 1;
        if(contains(attributes, "keepdims"))
        {
            keep_dims = parse_value(attributes.at("keepdims")).at<int>();
        }

        if(keep_dims == 0)
        {
            auto ins = prog.add_instruction(op::argmax{axis}, std::move(args));
            return prog.add_instruction(op::squeeze{{axis}}, ins);
        }
        else
        {
            return prog.add_instruction(op::argmax{axis}, std::move(args));
        }
    }

    instruction_ref parse_argmin(const std::string&,
                                 const attribute_map& attributes,
                                 std::vector<instruction_ref> args)
    {
        int64_t axis = 0;
        if(contains(attributes, "axis"))
        {
            axis = static_cast<int64_t>(parse_value(attributes.at("axis")).at<int>());
        }

        int keep_dims = 1;
        if(contains(attributes, "keepdims"))
        {
            keep_dims = parse_value(attributes.at("keepdims")).at<int>();
        }

        if(keep_dims == 0)
        {
            auto ins = prog.add_instruction(op::argmin{axis}, std::move(args));
            return prog.add_instruction(op::squeeze{{axis}}, ins);
        }
        else
        {
            return prog.add_instruction(op::argmin{axis}, std::move(args));
        }
    }

    instruction_ref
    parse_conv(const std::string&, attribute_map attributes, std::vector<instruction_ref> args)
    {
        op::convolution op;
        auto l0 = args[0];
        if(contains(attributes, "pads"))
        {
            if(contains(attributes, "auto_pad"))
            {
                MIGRAPHX_THROW("auto_pad and padding cannot be specified simultaneously");
            }
            std::vector<std::int64_t> padding;
            copy(attributes["pads"].ints(), std::back_inserter(padding));
            if(padding.size() != 4)
            {
                MIGRAPHX_THROW("padding should have 4 values");
            }
            if(padding[0] != padding[2] || padding[1] != padding[3])
            {
                // insert zeros for pad op (args[0] has 4 dims)
                padding = {0, 0, padding[0], padding[1], 0, 0, padding[2], padding[3]};
                l0      = prog.add_instruction(op::pad{padding}, l0);
            }
            else
            {
                op.padding[0] = padding[0];
                op.padding[1] = padding[1];
            }
        }
        if(contains(attributes, "strides"))
        {
            copy(attributes["strides"].ints(), op.stride.begin());
        }
        if(contains(attributes, "dilations"))
        {
            copy(attributes["dilations"].ints(), op.dilation.begin());
        }
        if(contains(attributes, "auto_pad"))
        {
            auto s = attributes["auto_pad"].s();
            if(contains(attributes, "pads") and to_upper(s) != "NOTSET")
            {
                MIGRAPHX_THROW("auto_pad and padding cannot be specified simultaneously");
            }

            if(s.find("SAME") != std::string::npos)
            {
                op.padding_mode = op::padding_mode_t::same;
            }
        }
        if(contains(attributes, "group"))
        {
            op.group = parse_value(attributes.at("group")).at<int>();
        }
        if(args.size() == 3)
        {
            uint64_t axis = 1;
            auto l1       = prog.add_instruction(op, args[0], args[1]);
            auto l2 = prog.add_instruction(op::broadcast{axis, l1->get_shape().lens()}, args[2]);
            return prog.add_instruction(op::add{}, l1, l2);
        }
        return prog.add_instruction(op, l0, args[1]);
    }

    instruction_ref parse_pooling(const std::string& name,
                                  attribute_map attributes,
                                  std::vector<instruction_ref> args)
    {
        op::pooling op{ends_with(name, "MaxPool") ? "max" : "average"};
        auto l0 = args[0];
        if(starts_with(name, "Global"))
        {
            auto lens  = args.front()->get_shape().lens();
            op.lengths = {lens[2], lens[3]};
        }
        if(contains(attributes, "pads"))
        {
            std::vector<std::int64_t> padding;
            copy(attributes["pads"].ints(), std::back_inserter(padding));
            if(padding.size() != 4)
            {
                MIGRAPHX_THROW("padding should have 4 values");
            }
            if(padding[0] != padding[2] || padding[1] != padding[3])
            {
                // insert zeros for pad op (args[0] has 4 dims)
                padding = {0, 0, padding[0], padding[1], 0, 0, padding[2], padding[3]};
                l0 = prog.add_instruction(op::pad{padding, std::numeric_limits<float>::lowest()},
                                          l0);
            }
            else
            {
                op.padding[0] = padding[0];
                op.padding[1] = padding[1];
            }
        }
        if(contains(attributes, "strides"))
        {
            copy(attributes["strides"].ints(), op.stride.begin());
        }
        if(contains(attributes, "kernel_shape"))
        {
            copy(attributes["kernel_shape"].ints(), op.lengths.begin());
        }
        if(contains(attributes, "auto_pad"))
        {
            auto s = attributes["auto_pad"].s();
            if(s.find("SAME_UPPER") == std::string::npos)
            {
                MIGRAPHX_THROW("auto_pad only supports SAME_UPPER for pooling");
            }
            op.padding_mode = op::padding_mode_t::same;
        }

        return prog.add_instruction(op, l0);
    }

    instruction_ref
    parse_reshape(const std::string&, attribute_map attributes, std::vector<instruction_ref> args)
    {
        op::reshape op;
        if(args.size() == 1)
        {
            if (contains(attributes, "shape"))
            {
                literal s = parse_value(attributes.at("shape"));
                s.visit([&](auto v) { copy(v, std::back_inserter(op.dims)); });
            }
            else
            {
                MIGRAPHX_THROW("Parse_reshape: shape attribute is needed when only one argument is provided!");
            }            
        }
        if(args.size() == 2)
        {
            auto s = args[1]->eval();
            if(s.empty())
                MIGRAPHX_THROW("Dynamic shape is not supported.");
            s.visit([&](auto v) { copy(v, std::back_inserter(op.dims)); });
        }

        if (!args[0]->get_shape().standard())
        {
            args[0] = prog.add_instruction(op::contiguous{}, args[0]);
        }

        return prog.add_instruction(op, args[0]);
    }

    instruction_ref
    parse_flatten(const std::string&, attribute_map attributes, std::vector<instruction_ref> args)
    {
        uint64_t axis = 1;
        if(contains(attributes, "axis"))
        {
            axis = parse_value(attributes.at("axis")).at<int>();
        }
        return prog.add_instruction(op::flatten{axis}, args[0]);
    }

    instruction_ref
    parse_squeeze(const std::string&, attribute_map attributes, std::vector<instruction_ref> args)
    {
        op::squeeze op;
        literal s = parse_value(attributes.at("axes"));
        s.visit([&](auto v) { copy(v, std::back_inserter(op.axes)); });
        return prog.add_instruction(op, args[0]);
    }

    instruction_ref
    parse_unsqueeze(const std::string&, attribute_map attributes, std::vector<instruction_ref> args)
    {
        op::unsqueeze op;
        literal s = parse_value(attributes.at("axes"));
        s.visit([&](auto v) { copy(v, std::back_inserter(op.axes)); });
        return prog.add_instruction(op, args[0]);
    }

    instruction_ref
    parse_concat(const std::string&, attribute_map attributes, std::vector<instruction_ref> args)
    {
        std::size_t axis = parse_value(attributes.at("axis")).at<int>();
        op::concat op{axis};
        return prog.add_instruction(op, std::move(args));
    }

    instruction_ref
    parse_gather(const std::string&, attribute_map attributes, std::vector<instruction_ref> args)
    {
        int axis = 0;
        if(contains(attributes, "axis"))
        {
            axis = parse_value(attributes.at("axis")).at<int>();
        }
        op::gather op{axis};
        return prog.add_instruction(op, std::move(args));
    }

    instruction_ref
    parse_slice(const std::string&, attribute_map attributes, std::vector<instruction_ref> args)
    {
        op::slice op;
        if(contains(attributes, "axes"))
        {
            literal s = parse_value(attributes.at("axes"));
            s.visit([&](auto v) { copy(v, std::back_inserter(op.axes)); });
        }
        {
            literal s = parse_value(attributes.at("ends"));
            s.visit([&](auto v) { copy(v, std::back_inserter(op.ends)); });
        }
        {
            literal s = parse_value(attributes.at("starts"));
            s.visit([&](auto v) { copy(v, std::back_inserter(op.starts)); });
        }
        return prog.add_instruction(op, args[0]);
    }

    instruction_ref parse_constant(const std::string&,
                                   attribute_map attributes,
                                   const std::vector<instruction_ref>&)
    {
        literal v = parse_value(attributes.at("value"));
        // return empty literal
        if(v.get_shape().elements() == 0)
        {
            return prog.add_literal(literal{});
        }

        auto dim_size = attributes.at("value").t().dims_size();
        // if dim_size is 0, it is a scalar
        if(dim_size == 0)
        {
            migraphx::shape scalar_shape{v.get_shape().type()};
            return prog.add_literal(migraphx::literal{scalar_shape, v.data()});
        }

        return prog.add_literal(v);
    }

    instruction_ref
    parse_gemm(const std::string&, attribute_map attributes, std::vector<instruction_ref> args)
    {
        float alpha = 1.0f;
        float beta  = 1.0f;
        bool transa = false;
        bool transb = false;
        if(contains(attributes, "alpha"))
        {
            alpha = parse_value(attributes.at("alpha")).at<float>();
        }
        if(contains(attributes, "beta"))
        {
            beta = parse_value(attributes.at("beta")).at<float>();
        }
        if(contains(attributes, "transA"))
        {
            transa = parse_value(attributes.at("transA")).at<bool>();
        }
        if(contains(attributes, "transB"))
        {
            transb = parse_value(attributes.at("transB")).at<bool>();
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

    instruction_ref
    parse_matmul(const std::string&, const attribute_map&, std::vector<instruction_ref> args)
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

        auto dot_res     = prog.add_instruction(op::dot{1.0f, 0.0f}, bl0, bl1);
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
    parse_batchnorm(const std::string&, attribute_map attributes, std::vector<instruction_ref> args)
    {
        float epsilon                                     = 1e-5f;
        float momentum                                    = 0.9f;
        op::batch_norm_inference::bn_infer_mode_t bn_mode = op::batch_norm_inference::spatial;
        bool is_test                                      = false;
        if(contains(attributes, "epsilon"))
        {
            epsilon = parse_value(attributes.at("epsilon")).at<float>();
        }
        if(contains(attributes, "momentum"))
        {
            momentum = parse_value(attributes.at("momentum")).at<float>();
        }
        if(contains(attributes, "is_test"))
        {
            is_test = parse_value(attributes.at("is_test")).at<uint64_t>() > 0;
        }
        if(contains(attributes, "spatial"))
        {
            bn_mode = (parse_value(attributes.at("spatial")).at<uint64_t>() > 0)
                          ? op::batch_norm_inference::spatial
                          : op::batch_norm_inference::per_activation;
        }
        (void)is_test;
        op::batch_norm_inference op{epsilon, momentum, bn_mode};
        return prog.add_instruction(op, std::move(args));
    }

    instruction_ref parse_leaky_relu(const std::string&,
                                     attribute_map attributes,
                                     std::vector<instruction_ref> args)
    {
        float alpha = 0.01; // default alpha val for leaky relu
        if(contains(attributes, "alpha"))
        {
            alpha = parse_value(attributes.at("alpha")).at<float>();
        }
        op::leaky_relu op{alpha};
        return prog.add_instruction(op, args.front());
    }

    instruction_ref
    parse_elu(const std::string&, attribute_map attributes, std::vector<instruction_ref> args)
    {
        float alpha = 1.0; // default alpha val for elu
        if(contains(attributes, "alpha"))
        {
            alpha = parse_value(attributes.at("alpha")).at<float>();
        }
        op::elu op{alpha};
        return prog.add_instruction(op, args.front());
    }

    instruction_ref
    parse_lrn(const std::string&, attribute_map attributes, std::vector<instruction_ref> args)
    {
        float alpha = 0.0001;
        float beta  = 0.75;
        float bias  = 1.0;
        int size    = 1;
        if(contains(attributes, "alpha"))
            alpha = parse_value(attributes.at("alpha")).at<float>();
        if(contains(attributes, "beta"))
            beta = parse_value(attributes.at("beta")).at<float>();
        if(contains(attributes, "bias"))
            bias = parse_value(attributes.at("bias")).at<float>();
        if(contains(attributes, "size"))
            size = parse_value(attributes.at("size")).at<int>();
        op::lrn op{alpha, beta, bias, size};
        return prog.add_instruction(op, args.front());
    }

    instruction_ref parse_imagescaler(const std::string&,
                                      attribute_map attributes,
                                      std::vector<instruction_ref> args)
    {
        float scale = 1.0;
        std::vector<float> bias{};
        if(contains(attributes, "scale"))
        {
            scale = parse_value(attributes.at("scale")).at<float>();
        }

        if(contains(attributes, "bias"))
        {
            auto&& bias_floats = attributes["bias"].floats();
            bias               = std::vector<float>(bias_floats.begin(), bias_floats.end());
        }
        auto input_lens = args.front()->get_shape().lens();

        auto scale_val = prog.add_literal(scale);
        auto bias_vals = prog.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::float_type, {bias.size()}}, bias});

        auto scale_tensor = prog.add_instruction(migraphx::op::scalar{input_lens}, scale_val);
        auto img_scaled   = prog.add_instruction(migraphx::op::mul{}, args.front(), scale_tensor);
        auto bias_bcast   = prog.add_instruction(migraphx::op::broadcast{1, input_lens}, bias_vals);
        return prog.add_instruction(migraphx::op::add{}, img_scaled, bias_bcast);
    }

    instruction_ref
    parse_transpose(const std::string&, attribute_map attributes, std::vector<instruction_ref> args)
    {
        std::vector<int64_t> perm{};
        if(contains(attributes, "perm"))
        {
            auto&& perm_vals = attributes["perm"].ints();
            perm             = std::vector<int64_t>(perm_vals.begin(), perm_vals.end());
        }
        return prog.add_instruction(migraphx::op::transpose{perm}, args.front());
    }

    instruction_ref
    parse_pad(const std::string&, attribute_map attributes, std::vector<instruction_ref> args)
    {
        std::vector<int64_t> pads{};
        float value = 0.0f;
        if(contains(attributes, "pads"))
        {
            auto&& pad_vals = attributes["pads"].ints();
            pads            = std::vector<int64_t>(pad_vals.begin(), pad_vals.end());
        }
        // check if padding is actually being done (at least one value is nonzero)
        if(std::all_of(pads.begin(), pads.end(), [](const int& i) { return i == 0; }))
        {
            return prog.add_instruction(migraphx::op::identity{}, args.front());
        }
        if(contains(attributes, "value"))
        {
            value = parse_value(attributes.at("value")).at<float>();
        }
        if(contains(attributes, "mode"))
        {
            auto mode = attributes.at("mode").s();
            if(mode != "constant")
                MIGRAPHX_THROW("migraphx currently only supports constant padding");
        }
        return prog.add_instruction(migraphx::op::pad{pads, value}, args.front());
    }
    // Use a literal instruction to replace the shape since, output of
    // shape operator are literals in migraphx
    instruction_ref
    parse_shape(const std::string&, const attribute_map&, std::vector<instruction_ref> args)
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
    instruction_ref parse_constant_fill(const std::string&,
                                        attribute_map attributes,
                                        std::vector<instruction_ref> args)
    {
        int input_as_shape = 0;
        int dtype          = 1;
        float value        = 0.0f;

        if(contains(attributes, "dtype"))
        {
            dtype = parse_value(attributes.at("dtype")).at<int>();
        }
        migraphx::shape::type_t type = get_type(dtype);

        if(contains(attributes, "input_as_shape"))
        {
            input_as_shape = parse_value(attributes.at("input_as_shape")).at<int>();
        }

        if(contains(attributes, "value"))
        {
            value = parse_value(attributes.at("value")).at<float>();
        }

        if(contains(attributes, "extra_shape"))
        {
            MIGRAPHX_THROW("ConstantFill: cannot handle extra shape attribute");
        }

        if(input_as_shape == 1)
        {
            if(args.size() != 1)
            {
                MIGRAPHX_THROW("ConstantFill: need an input argument as output shape");
            }

            if(contains(attributes, "shape"))
            {
                MIGRAPHX_THROW("ConstantFill: cannot set the shape argument and pass in an input "
                               "at the same time");
            }

            migraphx::argument in = args[0]->eval();
            if(in.empty())
            {
                MIGRAPHX_THROW("ConstantFill: cannot handle dynamic shape as input");
            }

            std::vector<std::size_t> dims;
            in.visit([&](auto input) { dims.assign(input.begin(), input.end()); });
            migraphx::shape s(type, dims);
            std::vector<float> values(s.elements(), value);
            return prog.add_literal(migraphx::literal(s, values));
        }
        else if(input_as_shape == 0)
        {
            if(!contains(attributes, "shape"))
            {
                MIGRAPHX_THROW("ConstantFill: attribute output shape is needed");
            }

            literal ls = parse_value(attributes.at("shape"));
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

    instruction_ref parse_constant_of_shape(const std::string&,
                                            attribute_map attributes,
                                            std::vector<instruction_ref> args)
    {
        literal l_val{};
        if(contains(attributes, "value"))
        {
            l_val = parse_value(attributes.at("value"));
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
            MIGRAPHX_THROW("Parse ConstantOfShape : must have 1 input!");
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
                if(in.empty())
                {
                    MIGRAPHX_THROW("Parse ConstantOfShape: cannot handle dynamic shape as input");
                }

                std::vector<std::size_t> dims;
                in.visit([&](auto input) { dims.assign(input.begin(), input.end()); });
                s = migraphx::shape{type, dims};
            }

            literal l_out;
            l_val.visit([&](auto val) {
                using type = std::remove_cv_t<typename decltype(val)::value_type>;
                // l_val contains only one element
                std::vector<type> out_vec(s.elements(), *val.begin());
                l_out = literal(s, out_vec);
            });

            return prog.add_literal(l_out);
        }
    }

    instruction_ref
    parse_expand(const std::string&, const attribute_map&, std::vector<instruction_ref> args)
    {
        auto in_lens             = args[0]->get_shape().lens();
        migraphx::argument arg_s = args[1]->eval();
        if(arg_s.empty())
        {
            MIGRAPHX_THROW("Parse Expand: cannot handle dynamic shape as input");
        }
        std::vector<std::size_t> dims;
        arg_s.visit([&](auto input) { dims.assign(input.begin(), input.end()); });
        auto out_lens = compute_broadcasted_lens(in_lens, dims);

        return prog.add_instruction(op::multibroadcast{out_lens}, args[0]);
    }

    std::vector<instruction_ref>
    parse_rnn(const std::string&, attribute_map attributes, std::vector<instruction_ref> args)
    {
        migraphx::shape input_shape = args[0]->get_shape();
        std::size_t hidden_size     = args[1]->get_shape().lens()[1];

        if(contains(attributes, "hidden_size"))
        {
            std::size_t hidden_size_att = parse_value(attributes.at("hidden_size")).at<int>();
            if(hidden_size != hidden_size_att)
            {
                MIGRAPHX_THROW("RNN: hidden size mismatch in input and attribute");
            }
        }

        // Handling of direction to be added later
        std::string direction{"forward"};
        if(contains(attributes, "direction"))
        {
            direction = attributes.at("direction").s();
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
        if(contains(attributes, "activations"))
        {
            auto names = attributes.at("activations").strings();
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
        std::transform(vec_names.begin(), vec_names.end(), vec_actv_funcs.begin(), [&](auto& fn) {
            return map_actv_funcs[fn];
        });

        // To be added later
        float clip = 0.0;
        if(contains(attributes, "clip"))
        {
            clip = parse_value(attributes.at("clip")).at<float>();
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
    parse_gru(const std::string&, attribute_map attributes, std::vector<instruction_ref> args)
    {
        migraphx::shape input_shape = args[0]->get_shape();
        std::size_t hidden_size     = args[2]->get_shape().lens()[2];

        if(contains(attributes, "hidden_size"))
        {
            std::size_t hidden_size_att = parse_value(attributes.at("hidden_size")).at<int>();
            if(hidden_size != hidden_size_att)
            {
                MIGRAPHX_THROW("GRU: hidden size mismatch in input and attribute");
            }
        }

        // Handling of direction to be added later
        std::string direction{"forward"};
        if(contains(attributes, "direction"))
        {
            direction = attributes.at("direction").s();
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
        if(contains(attributes, "activations"))
        {
            auto names = attributes.at("activations").strings();
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
        std::transform(vec_names.begin(), vec_names.end(), vec_actv_funcs.begin(), [&](auto& name) {
            return map_actv_funcs[name];
        });

        float clip = 0.0;
        if(contains(attributes, "clip"))
        {
            clip = parse_value(attributes.at("clip")).at<float>();
        }

        int linear_before_reset = 0;
        if(contains(attributes, "linear_before_reset"))
        {
            linear_before_reset = parse_value(attributes.at("linear_before_reset")).at<int>();
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
    parse_lstm(const std::string&, attribute_map attributes, std::vector<instruction_ref> args)
    {
        migraphx::shape input_shape = args[0]->get_shape();
        std::size_t hidden_size     = args[2]->get_shape().lens()[2];

        if(contains(attributes, "hidden_size"))
        {
            std::size_t hidden_size_att = parse_value(attributes.at("hidden_size")).at<int>();
            if(hidden_size != hidden_size_att)
            {
                MIGRAPHX_THROW("LSTM: hidden size mismatch in input and attribute");
            }
        }

        // Handling of direction to be added later
        std::string direction{"forward"};
        if(contains(attributes, "direction"))
        {
            direction = attributes.at("direction").s();
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
        if(contains(attributes, "activations"))
        {
            auto names = attributes.at("activations").strings();
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
        std::transform(vec_names.begin(), vec_names.end(), vec_actv_funcs.begin(), [&](auto& name) {
            return map_actv_funcs[name];
        });

        float clip = 0.0;
        if(contains(attributes, "clip"))
        {
            clip = parse_value(attributes.at("clip")).at<float>();
        }

        int input_forget = 0;
        if(contains(attributes, "input_forget"))
        {
            input_forget = parse_value(attributes.at("input_forget")).at<int>();
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
    instruction_ref parse_reduce_oper(const std::string&,
                                      attribute_map attributes,
                                      std::vector<instruction_ref> args)
    {
        std::size_t n_dim = args.front()->get_shape().lens().size();

        // default to reduce over all dimensions
        std::vector<int64_t> axes(n_dim);
        std::iota(axes.begin(), axes.end(), 0);
        if(contains(attributes, "axes"))
        {
            axes.clear();
            auto&& attr_axes = attributes["axes"].ints();
            axes             = std::vector<int64_t>(attr_axes.begin(), attr_axes.end());
        }

        int keep_dims = 1;
        if(contains(attributes, "keepdims"))
        {
            keep_dims = parse_value(attributes.at("keepdims")).at<int>();
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
    parse_cast(const std::string&, attribute_map attributes, std::vector<instruction_ref> args)
    {
        if(!contains(attributes, "to"))
        {
            MIGRAPHX_THROW("PARSE_CAST: missing to type attribute!");
        }

        int to_type        = parse_value(attributes.at("to")).at<int>();
        shape::type_t type = get_type(to_type);
        return prog.add_instruction(op::convert{type}, std::move(args));
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

    void parse_graph(const onnx::GraphProto& graph)
    {
        nodes = get_nodes(graph);
        std::unordered_map<std::string, onnx::TensorProto> initializer_data;
        for(auto&& f : graph.initializer())
        {
            initializer_data[f.name()] = f;
        }
        for(auto&& input : graph.input())
        {
            const std::string& name = input.name();
            // Does the input have an initializer?
            if(contains(initializer_data, name))
            {
                auto t             = initializer_data[name];
                instructions[name] = prog.add_literal(parse_tensor(t));
            }
            else
            {
                // TODO: Get shape of input parameter
                shape s            = parse_type(input.type());
                instructions[name] = prog.add_parameter(name, s);
            }
        }
        for(auto&& output : graph.output())
        {
            this->parse_node(output.name());
        }
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
                if(nodes.count(input) > 0)
                {
                    assert(name != input);
                    this->parse_node(input);
                }
                else if(input.empty())
                {
                    this->parse_undefined(input);
                }
                args.push_back(instructions.at(input));
            }
            std::vector<instruction_ref> result;
            if(ops.count(node.op_type()) == 0)
            {
                result.push_back(prog.add_instruction(op::unknown{node.op_type()}, args));
            }
            else
            {
                result = ops[node.op_type()](get_attributes(node), args);
            }
            // Even no output nodes produce output in migraphx
            if(node.output().empty() and result.size() == 1)
            {
                instructions[name] = result.front();
            }
            else
            {
                assert(node.output().size() >= result.size());
                std::transform(result.begin(),
                               result.end(),
                               node.output().begin(),
                               std::inserter(instructions, instructions.end()),
                               [](auto&& x, auto&& y) { return std::make_pair(y, x); });
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
        case onnx::AttributeProto::UNDEFINED: return {};
        case onnx::AttributeProto::FLOAT: return literal{attr.f()};
        case onnx::AttributeProto::INT: return literal{attr.i()};
        case onnx::AttributeProto::STRING: return {};
        case onnx::AttributeProto::TENSOR: return parse_tensor(attr.t());
        case onnx::AttributeProto::GRAPH: return {};
        case onnx::AttributeProto::FLOATS: return from_repeated(shape::float_type, attr.floats());
        case onnx::AttributeProto::INTS: return from_repeated(shape::int64_type, attr.ints());
        case onnx::AttributeProto::STRINGS: return {};
        case onnx::AttributeProto::TENSORS: return {};
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
            case onnx::TensorProto::UNDEFINED: throw std::runtime_error("");
            case onnx::TensorProto::FLOAT: return create_literal(shape::float_type, dims, s.data());
            case onnx::TensorProto::UINT8: throw std::runtime_error("");
            case onnx::TensorProto::INT8: return create_literal(shape::int32_type, dims, s.data());
            case onnx::TensorProto::UINT16:
                return create_literal(shape::int32_type, dims, s.data());
            case onnx::TensorProto::INT16: return create_literal(shape::int32_type, dims, s.data());
            case onnx::TensorProto::INT32: return create_literal(shape::int32_type, dims, s.data());
            case onnx::TensorProto::INT64: return create_literal(shape::int64_type, dims, s.data());
            case onnx::TensorProto::STRING: throw std::runtime_error("");
            case onnx::TensorProto::BOOL: return create_literal(shape::int32_type, dims, s.data());
            case onnx::TensorProto::FLOAT16:
                return create_literal(shape::half_type, dims, s.data());
            case onnx::TensorProto::DOUBLE:
                return create_literal(shape::double_type, dims, s.data());
            case onnx::TensorProto::UINT32: throw std::runtime_error("");
            case onnx::TensorProto::UINT64: throw std::runtime_error("");
            case onnx::TensorProto::COMPLEX64: throw std::runtime_error("");
            case onnx::TensorProto::COMPLEX128: throw std::runtime_error("");
            }
            MIGRAPHX_THROW("Invalid tensor type");
        }
        switch(t.data_type())
        {
        case onnx::TensorProto::UNDEFINED: throw std::runtime_error("");
        case onnx::TensorProto::FLOAT:
            return create_literal(shape::float_type, dims, t.float_data());
        case onnx::TensorProto::UINT8: throw std::runtime_error("");
        case onnx::TensorProto::INT8:
            return create_literal(shape::int32_type, dims, t.int32_data());
        case onnx::TensorProto::UINT16:
            return create_literal(shape::int32_type, dims, t.int32_data());
        case onnx::TensorProto::INT16:
            return create_literal(shape::int32_type, dims, t.int32_data());
        case onnx::TensorProto::INT32:
            return create_literal(shape::int32_type, dims, t.int32_data());
        case onnx::TensorProto::INT64:
            return create_literal(shape::int64_type, dims, t.int64_data());
        case onnx::TensorProto::STRING: throw std::runtime_error("");
        case onnx::TensorProto::BOOL:
            return create_literal(shape::int32_type, dims, t.int32_data());
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
        case onnx::TensorProto::UINT32: throw std::runtime_error("");
        case onnx::TensorProto::UINT64: throw std::runtime_error("");
        case onnx::TensorProto::COMPLEX64: throw std::runtime_error("");
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

    static shape parse_type(const onnx::TypeProto& t)
    {
        shape::type_t shape_type{};
        switch(t.tensor_type().elem_type())
        {
        case onnx::TensorProto::UNDEFINED:
            break; // throw std::runtime_error("Unsupported type UNDEFINED");
        case onnx::TensorProto::FLOAT: shape_type = shape::float_type; break;
        case onnx::TensorProto::UINT8:
            break; // throw std::runtime_error("Unsupported type UINT8");
        case onnx::TensorProto::INT8: shape_type = shape::int8_type; break;
        case onnx::TensorProto::UINT16: shape_type = shape::uint16_type; break;
        case onnx::TensorProto::INT16: shape_type = shape::int16_type; break;
        case onnx::TensorProto::INT32: shape_type = shape::int32_type; break;
        case onnx::TensorProto::INT64: shape_type = shape::int64_type; break;
        case onnx::TensorProto::STRING:
            break; // throw std::runtime_error("Unsupported type STRING");
        case onnx::TensorProto::BOOL:
            break; // throw std::runtime_error("Unsupported type BOOL");
        case onnx::TensorProto::FLOAT16: shape_type = shape::half_type; break;
        case onnx::TensorProto::DOUBLE: shape_type = shape::double_type; break;
        case onnx::TensorProto::UINT32: shape_type = shape::uint32_type; break;
        case onnx::TensorProto::UINT64: shape_type = shape::uint64_type; break;
        case onnx::TensorProto::COMPLEX64:
            break; // throw std::runtime_error("Unsupported type COMPLEX64");
        case onnx::TensorProto::COMPLEX128:
            break; // throw std::runtime_error("Unsupported type COMPLEX128");
        }
        std::vector<std::size_t> dims;
        auto&& tensor_dims = t.tensor_type().shape().dim();
        std::transform(tensor_dims.begin(),
                       tensor_dims.end(),
                       std::back_inserter(dims),
                       [](auto&& d) -> std::size_t {
                           if(not d.has_dim_value())
                           {
                               long default_batch_size = 1; // FIXME
                               return default_batch_size;
                           }
                           return d.dim_value();
                       });
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
};

program parse_onnx(const std::string& name)
{
    std::fstream input(name.c_str(), std::ios::in | std::ios::binary);
    onnx_parser parser;
#ifndef NDEBUG
    // Log the program when it can't be parsed
    try
    {
        parser.parse_from(input);
    }
    catch(...)
    {
        std::cerr << parser.prog << std::endl;
        throw;
    }
#else
    parser.parse_from(input);
#endif
    return std::move(parser.prog);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
