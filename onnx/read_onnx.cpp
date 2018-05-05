
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <onnx.pb.h>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <functional>

#include <rtg/fallthrough.hpp>
#include <rtg/program.hpp>
#include <rtg/operators.hpp>

struct unknown
{
    std::string op;
    std::string name() const { return "unknown:" + op; }
    rtg::shape compute_shape(std::vector<rtg::shape> input) const
    {
        if(input.empty())
            return {};
        else
            return input.front();
    }
    rtg::argument compute(std::vector<rtg::argument>) const { RTG_THROW("not computable"); }
};

template <class C, class T>
bool contains(C&& c, T&& x)
{
    return c.find(x) != c.end();
}

template <class Range, class Iterator>
void copy(Range&& r, Iterator it)
{
    std::copy(r.begin(), r.end(), it);
}

struct onnx_parser
{
    using attribute_map = std::unordered_map<std::string, onnx::AttributeProto>;
    using node_map      = std::unordered_map<std::string, onnx::NodeProto>;
    using op_func = std::function<rtg::instruction_ref(attribute_map, std::vector<rtg::instruction_ref>)>;
    node_map nodes;
    std::unordered_map<std::string, rtg::instruction_ref> instructions;
    rtg::program prog = rtg::program();

    std::unordered_map<std::string, op_func> ops;

    onnx_parser()
    {
        add_op("Conv", [this](attribute_map attributes, std::vector<rtg::instruction_ref> args) {
            rtg::convolution op;
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
            return prog.add_instruction(op, args);
        });
        add_op("MaxPool", [this](attribute_map attributes, std::vector<rtg::instruction_ref> args) {
            rtg::pooling op{"max"};
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
        add_op("Relu", [this](attribute_map, std::vector<rtg::instruction_ref> args) {
            return prog.add_instruction(rtg::activation{"relu"}, args);
        });
        add_op("Reshape", [this](attribute_map attributes, std::vector<rtg::instruction_ref> args) {
            rtg::reshape op;
            rtg::literal s = parse_value(attributes.at("shape"));
            s.visit([&](auto v) { copy(v, std::back_inserter(op.dims)); });
            return prog.add_instruction(op, args);
        });
        add_op("Constant", [this](attribute_map attributes, std::vector<rtg::instruction_ref>) {
            rtg::literal v = parse_value(attributes.at("value"));
            return prog.add_literal(v);
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
            rtg::shape s       = parse_type(input.type());
            instructions[name] = prog.add_parameter(name, s);
        }
        for(auto&& p : nodes)
        {
            this->parse_node(p.second.name());
        }
    }

    void parse_node(std::string name)
    {
        if(instructions.count(name) == 0)
        {
            auto&& node = nodes.at(name);
            std::vector<rtg::instruction_ref> args;
            for(auto&& input : node.input())
            {
                if(nodes.count(input) > 0)
                {
                    auto&& iname = nodes.at(input).name();
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

    static node_map get_nodes(const onnx::GraphProto& graph)
    {
        std::unordered_map<std::string, onnx::NodeProto> result;
        for(auto&& node : graph.node())
        {
            result[node.name()] = node;
            for(auto&& output : node.output())
            {
                result[output] = node;
            }
        }
        return result;
    }

    static rtg::literal parse_value(const onnx::AttributeProto& attr)
    {
        switch(attr.type())
        {
        case onnx::AttributeProto::UNDEFINED: return {};
        case onnx::AttributeProto::FLOAT: return rtg::literal{attr.f()};
        case onnx::AttributeProto::INT: return rtg::literal{attr.i()};
        case onnx::AttributeProto::STRING: return {};
        case onnx::AttributeProto::TENSOR: return parse_tensor(attr.t());
        case onnx::AttributeProto::GRAPH: return {};
        case onnx::AttributeProto::FLOATS:
            return rtg::literal{rtg::shape::float_type, attr.floats().begin(), attr.floats().end()};
        case onnx::AttributeProto::INTS:
            return rtg::literal{rtg::shape::int32_type, attr.ints().begin(), attr.ints().end()};
            ;
        case onnx::AttributeProto::STRINGS: return {};
        case onnx::AttributeProto::TENSORS: return {};
        case onnx::AttributeProto::GRAPHS: return {};
        }
    }

    static rtg::literal parse_tensor(const onnx::TensorProto& t)
    {
        std::vector<std::size_t> dims(t.dims().begin(), t.dims().end());
        switch(t.data_type())
        {
        case onnx::TensorProto::UNDEFINED: throw std::runtime_error("");
        case onnx::TensorProto::FLOAT:
            return rtg::literal{
                {rtg::shape::float_type, dims}, t.float_data().begin(), t.float_data().end()};
        case onnx::TensorProto::UINT8: throw std::runtime_error("");
        case onnx::TensorProto::INT8:
            return rtg::literal{
                {rtg::shape::int32_type, dims}, t.int32_data().begin(), t.int32_data().end()};
        case onnx::TensorProto::UINT16:
            return rtg::literal{
                {rtg::shape::int32_type, dims}, t.int32_data().begin(), t.int32_data().end()};
        case onnx::TensorProto::INT16:
            return rtg::literal{
                {rtg::shape::int32_type, dims}, t.int32_data().begin(), t.int32_data().end()};
        case onnx::TensorProto::INT32:
            return rtg::literal{
                {rtg::shape::int32_type, dims}, t.int32_data().begin(), t.int32_data().end()};
        case onnx::TensorProto::INT64:
            return rtg::literal{
                {rtg::shape::int64_type, dims}, t.int64_data().begin(), t.int64_data().end()};
        case onnx::TensorProto::STRING: throw std::runtime_error("");
        case onnx::TensorProto::BOOL:
            return rtg::literal{
                {rtg::shape::int32_type, dims}, t.int32_data().begin(), t.int32_data().end()};
        case onnx::TensorProto::FLOAT16: throw std::runtime_error("");
        case onnx::TensorProto::DOUBLE:
            return rtg::literal{
                {rtg::shape::double_type, dims}, t.double_data().begin(), t.double_data().end()};
        case onnx::TensorProto::UINT32: throw std::runtime_error("");
        case onnx::TensorProto::UINT64: throw std::runtime_error("");
        case onnx::TensorProto::COMPLEX64: throw std::runtime_error("");
        case onnx::TensorProto::COMPLEX128: throw std::runtime_error("");
        }
    }

    static rtg::shape parse_type(const onnx::TypeProto& t)
    {
        rtg::shape::type_t shape_type{};
        switch(t.tensor_type().elem_type())
        {
        case onnx::TensorProto::UNDEFINED:
            break; // throw std::runtime_error("Unsupported type UNDEFINED");
        case onnx::TensorProto::FLOAT: shape_type = rtg::shape::float_type; break;
        case onnx::TensorProto::UINT8:
            break; // throw std::runtime_error("Unsupported type UINT8");
        case onnx::TensorProto::INT8: shape_type = rtg::shape::int8_type; break;
        case onnx::TensorProto::UINT16: shape_type = rtg::shape::uint16_type; break;
        case onnx::TensorProto::INT16: shape_type = rtg::shape::int16_type; break;
        case onnx::TensorProto::INT32: shape_type = rtg::shape::int32_type; break;
        case onnx::TensorProto::INT64: shape_type = rtg::shape::int64_type; break;
        case onnx::TensorProto::STRING:
            break; // throw std::runtime_error("Unsupported type STRING");
        case onnx::TensorProto::BOOL:
            break; // throw std::runtime_error("Unsupported type BOOL");
        case onnx::TensorProto::FLOAT16:
            break; // throw std::runtime_error("Unsupported type FLOAT16");
        case onnx::TensorProto::DOUBLE: shape_type = rtg::shape::double_type; break;
        case onnx::TensorProto::UINT32: shape_type = rtg::shape::uint32_type; break;
        case onnx::TensorProto::UINT64: shape_type = rtg::shape::uint64_type; break;
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

int main(int argc, char const* argv[])
{
    if(argc > 1)
    {
        std::string file = argv[1];
        std::fstream input(file.c_str(), std::ios::in | std::ios::binary);
        onnx_parser parser;
        try
        {
            parser.parse_from(input);
        }
        catch(...)
        {
            parser.prog.print();
            throw;
        }
        parser.prog.print();
    }
}
