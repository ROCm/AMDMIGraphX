#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <onnx.pb.h>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <functional>
#include <array>
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
    argument compute(context&, shape, std::vector<argument>) const
    {
        MIGRAPH_THROW("not computable");
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
        add_op("Conv", [this](attribute_map attributes, std::vector<instruction_ref> args) {
            convolution op;
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
                auto l2       = prog.add_instruction(broadcast{axis}, l1, args[2]);
                return prog.add_instruction(add{}, l1, l2);
            }
            return prog.add_instruction(op, args);
        });
        add_op("MatMul", [this](attribute_map, std::vector<instruction_ref> args) {
            return prog.add_instruction(gemm{}, args);
        });
        add_op("MaxPool", [this](attribute_map attributes, std::vector<instruction_ref> args) {
            pooling op{"max"};
            // for(auto&& p:attributes) std::cout << p.first << std::endl;
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
            return prog.add_instruction(op, args);
        });
        add_op("Relu", [this](attribute_map, std::vector<instruction_ref> args) {
            return prog.add_instruction(activation{"relu"}, args);
        });
        add_op("Reshape", [this](attribute_map attributes, std::vector<instruction_ref> args) {
            reshape op;
            if(args.size() == 1)
            {
                literal s = parse_value(attributes.at("shape"));
                s.visit([&](auto v) { copy(v, std::back_inserter(op.dims)); });
            }
            if(args.size() == 2)
            {
                literal s = args[1]->lit;
                s.visit([&](auto v) { copy(v, std::back_inserter(op.dims)); });
            }
            return prog.add_instruction(op, args[0]);
        });
        add_op("Constant", [this](attribute_map attributes, std::vector<instruction_ref>) {
            literal v = parse_value(attributes.at("value"));
            return prog.add_literal(v);
        });
        add_op("Add", [this](attribute_map attributes, std::vector<instruction_ref> args) {
            if(contains(attributes, "broadcast"))
            {
                uint64_t broadcasted = parse_value(attributes.at("broadcast")).at<uint64_t>();
                if(broadcasted != 0)
                {
                    uint64_t axis = (contains(attributes, "axis"))
                                        ? parse_value(attributes.at("axis")).at<uint64_t>()
                                        : 0;
                    auto l = prog.add_instruction(broadcast{axis}, args);
                    return prog.add_instruction(add{}, args[0], l);
                }
            }
            return prog.add_instruction(add{}, args);
        });
        add_op("Sub", [this](attribute_map attributes, std::vector<instruction_ref> args) {
            if(contains(attributes, "broadcast"))
            {
                uint64_t broadcasted = parse_value(attributes.at("broadcast")).at<uint64_t>();
                if(broadcasted != 0)
                {
                    uint64_t axis = (contains(attributes, "axis"))
                                        ? parse_value(attributes.at("axis")).at<uint64_t>()
                                        : 0;
                    auto l = prog.add_instruction(broadcast{axis}, args);
                    return prog.add_instruction(sub{}, args[0], l);
                }
            }
            return prog.add_instruction(sub{}, args);
        });
        add_op("Mul", [this](attribute_map attributes, std::vector<instruction_ref> args) {
            if(contains(attributes, "broadcast"))
            {
                uint64_t broadcasted = parse_value(attributes.at("broadcast")).at<uint64_t>();
                if(broadcasted != 0)
                {
                    uint64_t axis = (contains(attributes, "axis"))
                                        ? parse_value(attributes.at("axis")).at<uint64_t>()
                                        : 0;
                    auto l = prog.add_instruction(broadcast{axis}, args);
                    return prog.add_instruction(mul{}, args[0], l);
                }
            }
            return prog.add_instruction(mul{}, args);
        });
        add_op("Div", [this](attribute_map attributes, std::vector<instruction_ref> args) {
            if(contains(attributes, "broadcast"))
            {
                uint64_t broadcasted = parse_value(attributes.at("broadcast")).at<uint64_t>();
                if(broadcasted != 0)
                {
                    uint64_t axis = (contains(attributes, "axis"))
                                        ? parse_value(attributes.at("axis")).at<uint64_t>()
                                        : 0;
                    auto l = prog.add_instruction(broadcast{axis}, args);
                    return prog.add_instruction(div{}, args[0], l);
                }
            }
            return prog.add_instruction(div{}, args);
        });
    }

    template <class F>
    void add_op(std::string name, F f)
    {
        ops.emplace(name, f);
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
        for(auto&& input : graph.input())
        {
            const std::string& name = input.name();
            // TODO: Get shape of input parameter
            shape s            = parse_type(input.type());
            instructions[name] = prog.add_parameter(name, s);
        }
        for(auto&& p : nodes)
        {
            this->parse_node(get_name(p.second));
        }
    }

    void parse_node(std::string name)
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
            for(auto&& output : node.output())
            {
                generated += "_" + output;
            }
            return generated;
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
            std::string s = t.raw_data();
            if(t.data_type() == onnx::TensorProto::FLOAT)
            {
                return literal{{shape::float_type, dims}, s.data()};
            }
            else if(t.data_type() == onnx::TensorProto::UINT8)
            {
                throw std::runtime_error("");
            }
            else if(t.data_type() == onnx::TensorProto::INT8)
            {
                return literal{{shape::int32_type, dims}, s.data()};
            }
            else if(t.data_type() == onnx::TensorProto::UINT16)
            {
                return literal{{shape::int32_type, dims}, s.data()};
            }
            else if(t.data_type() == onnx::TensorProto::INT16)
            {
                return literal{{shape::int32_type, dims}, s.data()};
            }
            else if(t.data_type() == onnx::TensorProto::INT32)
            {
                return literal{{shape::int32_type, dims}, s.data()};
            }
            else if(t.data_type() == onnx::TensorProto::INT64)
            {
                return literal{{shape::int64_type, dims}, s.data()};
            }
            else if(t.data_type() == onnx::TensorProto::STRING)
            {
                throw std::runtime_error("");
            }
            else if(t.data_type() == onnx::TensorProto::BOOL)
            {
                return literal{{shape::int32_type, dims}, s.data()};
            }
            else if(t.data_type() == onnx::TensorProto::FLOAT16)
            {
                throw std::runtime_error("");
            }
            else if(t.data_type() == onnx::TensorProto::DOUBLE)
            {
                return literal{{shape::double_type, dims}, s.data()};
            }
            else if(t.data_type() == onnx::TensorProto::UINT32)
            {
                throw std::runtime_error("");
            }
            else if(t.data_type() == onnx::TensorProto::UINT64)
            {
                throw std::runtime_error("");
            }
            else if(t.data_type() == onnx::TensorProto::COMPLEX64)
            {
                throw std::runtime_error("");
            }
            else if(t.data_type() == onnx::TensorProto::COMPLEX128)
            {
                throw std::runtime_error("");
            }
            else
            {
                MIGRAPH_THROW("Invalid tensor type");
            }
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
        // TODO: USe std::transform
        for(auto&& d : t.tensor_type().shape().dim())
        {
            dims.push_back(d.dim_value());
        }
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
