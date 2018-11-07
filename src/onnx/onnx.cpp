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

#include <migraph/fallthrough.hpp>
#include <migraph/program.hpp>
#include <migraph/operators.hpp>
#include <migraph/ranges.hpp>
#include <migraph/instruction.hpp>

namespace migraph {

struct unknown
{
    std::string op;
    std::string name() const { return "unknown:" + op; }
    shape compute_shape(std::vector<shape> input) const
    {
        if(input.empty())
            return {};
        else
            return input.front();
    }
    friend std::ostream& operator<<(std::ostream& os, const unknown& x)
    {
        os << x.name();
        return os;
    }
};

struct onnx_parser
{
    using attribute_map = std::unordered_map<std::string, onnx::AttributeProto>;
    using node_map      = std::unordered_map<std::string, onnx::NodeProto>;
    using op_func = std::function<instruction_ref(attribute_map, std::vector<instruction_ref>)>;
    node_map nodes;
    std::unordered_map<std::string, instruction_ref> instructions;
    program prog = program();

    std::unordered_map<std::string, op_func> ops;

    onnx_parser()
    {
        add_generic_op("Add", op::add{});
        add_generic_op("Div", op::div{});
        add_generic_op("MatMul", op::dot{});
        add_generic_op("Mul", op::mul{});
        add_generic_op("Relu", op::relu{});
        add_generic_op("Sub", op::sub{});
        add_generic_op("Sum", op::add{});

        add_mem_op("ImageScaler", &onnx_parser::parse_imagescaler);
        add_mem_op("LeakyRelu", &onnx_parser::parse_leaky_relu);
        add_mem_op("Constant", &onnx_parser::parse_constant);
        add_mem_op("Conv", &onnx_parser::parse_conv);
        add_mem_op("MaxPool", &onnx_parser::parse_pooling);
        add_mem_op("AveragePool", &onnx_parser::parse_pooling);
        add_mem_op("GlobalMaxPool", &onnx_parser::parse_pooling);
        add_mem_op("GlobalAveragePool", &onnx_parser::parse_pooling);
        add_mem_op("Reshape", &onnx_parser::parse_reshape);
        add_mem_op("Flatten", &onnx_parser::parse_flatten);
        add_mem_op("Gemm", &onnx_parser::parse_gemm);
        add_mem_op("BatchNormalization", &onnx_parser::parse_batchnorm);
        add_mem_op("Softmax", &onnx_parser::parse_softmax);
        add_mem_op("Squeeze", &onnx_parser::parse_squeeze);
        add_mem_op("Unsqueeze", &onnx_parser::parse_unsqueeze);
        add_mem_op("Slice", &onnx_parser::parse_slice);
        add_mem_op("Concat", &onnx_parser::parse_concat);
    }

    template <class F>
    void add_op(std::string name, F f)
    {
        ops.emplace(name, f);
    }

    template <class F>
    void add_mem_op(std::string name, F f)
    {
        ops.emplace(name, [=](auto&&... xs) {
            return std::mem_fn(f)(*this, name, std::forward<decltype(xs)>(xs)...);
        });
    }

    template <class T>
    void add_generic_op(std::string name, T x)
    {
        ops.emplace(name, [this, x](attribute_map attributes, std::vector<instruction_ref> args) {
            if(args.size() == 2 and contains(attributes, "broadcast"))
            {
                uint64_t broadcasted = parse_value(attributes.at("broadcast")).at<uint64_t>();
                if(broadcasted != 0)
                {
                    uint64_t axis = (contains(attributes, "axis"))
                                        ? parse_value(attributes.at("axis")).at<uint64_t>()
                                        : 0;
                    auto l =
                        prog.add_instruction(op::broadcast{axis, args[0]->get_shape()}, args[1]);
                    return prog.add_instruction(x, args[0], l);
                }
            }
            return prog.add_instruction(x, args);
        });
    }

    instruction_ref
    parse_softmax(const std::string&, const attribute_map&, std::vector<instruction_ref> args)
    {
        auto dims = args.front()->get_shape().lens();
        auto r =
            prog.add_instruction(op::reshape{{long(dims[0]), long(dims[1]), 1, 1}}, args.front());
        auto s = prog.add_instruction(op::softmax{}, r);
        return prog.add_instruction(op::reshape{{long(dims[0]), long(dims[1])}}, s);
    }

    instruction_ref
    parse_conv(const std::string&, attribute_map attributes, std::vector<instruction_ref> args)
    {
        op::convolution op;
        if(contains(attributes, "pads"))
        {
            copy(attributes["pads"].ints(), op.padding.begin());
        }
        if(contains(attributes, "strides"))
        {
            copy(attributes["strides"].ints(), op.stride.begin());
        }
        if(contains(attributes, "dilations"))
        {
            copy(attributes["dilations"].ints(), op.dilation.begin());
        }
        if(args.size() == 3)
        {
            uint64_t axis = 1;
            auto l1       = prog.add_instruction(op, args[0], args[1]);
            auto l2       = prog.add_instruction(op::broadcast{axis, l1->get_shape()}, args[2]);
            return prog.add_instruction(op::add{}, l1, l2);
        }
        return prog.add_instruction(op, args);
    }

    instruction_ref parse_pooling(const std::string& name,
                                  attribute_map attributes,
                                  std::vector<instruction_ref> args)
    {
        op::pooling op{ends_with(name, "MaxPool") ? "max" : "average"};
        if(starts_with(name, "Global"))
        {
            auto lens  = args.front()->get_shape().lens();
            op.lengths = {lens[2], lens[3]};
        }
        if(contains(attributes, "pads"))
        {
            copy(attributes["pads"].ints(), op.padding.begin());
        }
        if(contains(attributes, "strides"))
        {
            copy(attributes["strides"].ints(), op.stride.begin());
        }
        if(contains(attributes, "kernel_shape"))
        {
            copy(attributes["kernel_shape"].ints(), op.lengths.begin());
        }
        return prog.add_instruction(op, std::move(args));
    }

    instruction_ref
    parse_reshape(const std::string&, attribute_map attributes, std::vector<instruction_ref> args)
    {
        op::reshape op;
        if(args.size() == 1)
        {
            literal s = parse_value(attributes.at("shape"));
            s.visit([&](auto v) { copy(v, std::back_inserter(op.dims)); });
        }
        if(args.size() == 2)
        {
            literal s = args[1]->get_literal();
            s.visit([&](auto v) { copy(v, std::back_inserter(op.dims)); });
        }
        return prog.add_instruction(op, args[0]);
    }

    instruction_ref
    parse_flatten(const std::string&, attribute_map attributes, std::vector<instruction_ref> args)
    {
        uint64_t axis = 0;
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
        return prog.add_literal(v);
    }

    instruction_ref
    parse_gemm(const std::string&, attribute_map attributes, std::vector<instruction_ref> args)
    {
        float alpha = 1.0f;
        float beta  = 0.0f;
        bool transa = false;
        bool transb = false;
        if(contains(attributes, "alpha"))
        {
            alpha = parse_value(attributes.at("alpha")).at<float>();
        }
        if(contains(attributes, "beta"))
        {
            alpha = parse_value(attributes.at("beta")).at<float>();
        }
        if(contains(attributes, "transA"))
        {
            transa = parse_value(attributes.at("transA")).at<bool>();
        }
        if(contains(attributes, "transB"))
        {
            transb = parse_value(attributes.at("transB")).at<bool>();
        }
        std::vector<int64_t> perm = {1, 0};
        auto l1 = (transa) ? prog.add_instruction(op::transpose{perm}, args[0]) : args[0];
        auto l2 = (transb) ? prog.add_instruction(op::transpose{perm}, args[1]) : args[1];
        if(args.size() == 3)
        {
            uint64_t axis = 1;
            auto l3       = prog.add_instruction(op::dot{alpha, beta}, l1, l2);
            auto l4       = prog.add_instruction(op::broadcast{axis, l3->get_shape()}, args[2]);
            return prog.add_instruction(op::add{}, l3, l4);
        }
        return prog.add_instruction(op::dot{alpha, beta}, l1, l2);
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
        auto input_shape = args.front()->get_shape();

        auto scale_val = prog.add_literal(scale);
        auto bias_vals = prog.add_literal(
            migraph::literal{migraph::shape{migraph::shape::float_type, {bias.size()}}, bias});

        auto scale_tensor = prog.add_instruction(migraph::op::scalar{input_shape}, scale_val);
        auto img_scaled   = prog.add_instruction(migraph::op::mul{}, args.front(), scale_tensor);
        auto bias_bcast   = prog.add_instruction(migraph::op::broadcast{1, input_shape}, bias_vals);
        return prog.add_instruction(migraph::op::add{}, img_scaled, bias_bcast);
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
            throw std::runtime_error("Failed reading");
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
        for(auto&& p : nodes)
        {
            this->parse_node(get_name(p.second));
        }
    }

    void parse_node(const std::string& name)
    {
        if(name.empty())
            MIGRAPH_THROW("Onnx node must have a name");
        if(instructions.count(name) == 0)
        {
            auto&& node = nodes.at(name);
            std::vector<instruction_ref> args;
            for(auto&& input : node.input())
            {
                if(nodes.count(input) > 0)
                {
                    auto&& iname = get_name(nodes.at(input));
                    assert(name != iname);
                    this->parse_node(iname);
                    args.push_back(instructions.at(iname));
                }
                else
                {
                    args.push_back(instructions.at(input));
                }
            }
            if(ops.count(node.op_type()) == 0)
            {
                instructions[name] = prog.add_instruction(unknown{node.op_type()}, args);
            }
            else
            {
                instructions[name] = ops[node.op_type()](get_attributes(node), args);
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

    static std::string get_name(const onnx::NodeProto& node)
    {
        if(node.name().empty())
        {
            std::string generated = "migraph_unnamed_node";
            return std::accumulate(node.output().begin(),
                                   node.output().end(),
                                   generated,
                                   [](auto x, auto y) { return x + "_" + y; });
        }
        return node.name();
    }

    static node_map get_nodes(const onnx::GraphProto& graph)
    {
        std::unordered_map<std::string, onnx::NodeProto> result;
        for(auto&& node : graph.node())
        {
            result[get_name(node)] = node;
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
        MIGRAPH_THROW("Invalid attribute type");
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
            case onnx::TensorProto::FLOAT: return literal{{shape::float_type, dims}, s.data()};
            case onnx::TensorProto::UINT8: throw std::runtime_error("");
            case onnx::TensorProto::INT8: return literal{{shape::int32_type, dims}, s.data()};
            case onnx::TensorProto::UINT16: return literal{{shape::int32_type, dims}, s.data()};
            case onnx::TensorProto::INT16: return literal{{shape::int32_type, dims}, s.data()};
            case onnx::TensorProto::INT32: return literal{{shape::int32_type, dims}, s.data()};
            case onnx::TensorProto::INT64: return literal{{shape::int64_type, dims}, s.data()};
            case onnx::TensorProto::STRING: throw std::runtime_error("");
            case onnx::TensorProto::BOOL: return literal{{shape::int32_type, dims}, s.data()};
            case onnx::TensorProto::FLOAT16: throw std::runtime_error("");
            case onnx::TensorProto::DOUBLE: return literal{{shape::double_type, dims}, s.data()};
            case onnx::TensorProto::UINT32: throw std::runtime_error("");
            case onnx::TensorProto::UINT64: throw std::runtime_error("");
            case onnx::TensorProto::COMPLEX64: throw std::runtime_error("");
            case onnx::TensorProto::COMPLEX128: throw std::runtime_error("");
            }
            MIGRAPH_THROW("Invalid tensor type");
        }
        switch(t.data_type())
        {
        case onnx::TensorProto::UNDEFINED: throw std::runtime_error("");
        case onnx::TensorProto::FLOAT:
            return literal{{shape::float_type, dims}, t.float_data().begin(), t.float_data().end()};
        case onnx::TensorProto::UINT8: throw std::runtime_error("");
        case onnx::TensorProto::INT8:
            return literal{{shape::int32_type, dims}, t.int32_data().begin(), t.int32_data().end()};
        case onnx::TensorProto::UINT16:
            return literal{{shape::int32_type, dims}, t.int32_data().begin(), t.int32_data().end()};
        case onnx::TensorProto::INT16:
            return literal{{shape::int32_type, dims}, t.int32_data().begin(), t.int32_data().end()};
        case onnx::TensorProto::INT32:
            return literal{{shape::int32_type, dims}, t.int32_data().begin(), t.int32_data().end()};
        case onnx::TensorProto::INT64:
            return literal{{shape::int64_type, dims}, t.int64_data().begin(), t.int64_data().end()};
        case onnx::TensorProto::STRING: throw std::runtime_error("");
        case onnx::TensorProto::BOOL:
            return literal{{shape::int32_type, dims}, t.int32_data().begin(), t.int32_data().end()};
        case onnx::TensorProto::FLOAT16: throw std::runtime_error("");
        case onnx::TensorProto::DOUBLE:
            return literal{
                {shape::double_type, dims}, t.double_data().begin(), t.double_data().end()};
        case onnx::TensorProto::UINT32: throw std::runtime_error("");
        case onnx::TensorProto::UINT64: throw std::runtime_error("");
        case onnx::TensorProto::COMPLEX64: throw std::runtime_error("");
        case onnx::TensorProto::COMPLEX128: throw std::runtime_error("");
        }
        MIGRAPH_THROW("Invalid tensor type");
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
        case onnx::TensorProto::FLOAT16:
            break; // throw std::runtime_error("Unsupported type FLOAT16");
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
        std::transform(
            tensor_dims.begin(), tensor_dims.end(), std::back_inserter(dims), [](auto&& d) {
                if(not d.has_dim_value())
                {
                    long default_batch_size = 1; // FIXME
                    return default_batch_size;
                }
                return d.dim_value();
            });
        return {shape_type, dims};
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

} // namespace migraph
