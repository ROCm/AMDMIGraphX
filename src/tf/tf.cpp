#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <graph.pb.h>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
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
#include <migraphx/tf.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct tf_parser
{
    using attribute_map = std::unordered_map<std::string, tensorflow::AttrValue>;
    using node_map      = std::unordered_map<std::string, tensorflow::NodeDef>;
    // using input_node_map = std::unordered_map<std::string, std::unordered_set<std::string>>;
    using op_func = std::function<instruction_ref(attribute_map, std::vector<instruction_ref>)>;

    node_map nodes;
    std::vector<tensorflow::NodeDef> input_nodes;
    std::unordered_map<std::string, instruction_ref> instructions;
    program prog = program();
    bool is_nhwc = true;

    std::unordered_map<std::string, op_func> ops;

    tf_parser()
    {
        add_generic_op("Identity", op::identity{});
        add_generic_op("Relu", op::relu{});

        add_binary_op("BiasAdd", op::add{});

        add_mem_op("AvgPool", &tf_parser::parse_pooling);
        // add_mem_op("ConcatV2", &tf_parser::parse_concat);
        add_mem_op("Const", &tf_parser::parse_constant);
        add_mem_op("Conv2D", &tf_parser::parse_conv);
        add_mem_op("FusedBatchNorm", &tf_parser::parse_batchnorm);
    }

    template <class F>
    void add_op(std::string name, F f)
    {
        ops.emplace(name, f);
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
        add_op(name, [this, x](attribute_map, std::vector<instruction_ref> args) {
            if(args.size() != 2)
                MIGRAPHX_THROW("binary operators should have 2 operands");
            return add_broadcastable_binary_op(args[0], args[1], x);
        });
    }

    template <class T>
    instruction_ref add_broadcastable_binary_op(instruction_ref arg0, instruction_ref arg1, T x)
    {
        if(arg0->get_shape() != arg1->get_shape())
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
            //
            // Get lengths for both arguments
            const std::vector<std::size_t>* s0 = &arg0->get_shape().lens();
            const std::vector<std::size_t>* s1 = &arg1->get_shape().lens();

            // Make sure s0 is the smaller size
            if(s0->size() > s1->size())
                std::swap(s0, s1);

            std::vector<std::size_t> output_lens(*s1);
            auto offset = s1->size() - s0->size();
            std::transform(s0->begin(),
                           s0->end(),
                           s1->begin() + offset,
                           output_lens.begin() + offset,
                           [](auto a, auto b) { return std::max(a, b); });

            auto l0 = prog.add_instruction(op::multibroadcast{output_lens}, arg0);
            auto l1 = prog.add_instruction(op::multibroadcast{output_lens}, arg1);
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
        add_op(name, [this, x](attribute_map, std::vector<instruction_ref> args) {
            return prog.add_instruction(x, args);
        });
    }

    instruction_ref
    parse_batchnorm(const std::string&, attribute_map attributes, std::vector<instruction_ref> args)
    {
        float epsilon  = 1e-4f;
        float momentum = 1.f;
        op::batch_norm_inference::bn_infer_mode_t bn_mode =
            op::batch_norm_inference::per_activation;
        if(contains(attributes, "epsilon"))
        {
            epsilon = attributes.at("epsilon").f();
        }

        op::batch_norm_inference op{epsilon, momentum, bn_mode};
        return prog.add_instruction(op, std::move(args));
    }

    instruction_ref
    parse_concat(const std::string&, attribute_map attributes, std::vector<instruction_ref> args)
    {
        // get index for axis within args
        std::size_t axis_idx = attributes.at("N").i();
        std::size_t axis     = args[axis_idx]->eval().at<int64_t>();
        op::concat op{axis};
        return prog.add_instruction(op, std::move(args));
    }

    instruction_ref parse_constant(const std::string&,
                                   attribute_map attributes,
                                   const std::vector<instruction_ref>&)
    {
        literal v = parse_tensor(attributes.at("value").tensor());
        return prog.add_literal(v);
    }

    instruction_ref
    parse_conv(const std::string&, attribute_map attributes, std::vector<instruction_ref> args)
    {
        op::convolution op;
        if(contains(attributes, "padding"))
        {
            const std::string& pad_mode = attributes.at("padding").s();
            if(pad_mode.find("SAME") != std::string::npos)
            {
                op.padding_mode = op::padding_mode_t::same;
            }
            else if(pad_mode.find("EXPLICIT") != std::string::npos)
            {
                std::vector<std::size_t> padding;
                copy(attributes.at("explicit_paddings").list().i(), std::back_inserter(padding));
                if(padding.size() != 4)
                {
                    MIGRAPHX_THROW("padding should have 4 values");
                }
                if(padding[0] != padding[2] || padding[1] != padding[3])
                {
                    MIGRAPHX_THROW("migraphx does not support asymetric padding");
                }
                op.padding[0] = padding[0];
                op.padding[1] = padding[1];
            }
        }
        if(contains(attributes, "strides"))
        {
            std::vector<std::size_t> stride;
            copy(attributes.at("strides").list().i(), std::back_inserter(stride));
            if(stride.size() != 4)
            {
                MIGRAPHX_THROW("strides should have 4 values");
            }
            if(is_nhwc)
            {
                op.stride[0] = stride[1];
                op.stride[1] = stride[2];
                
            }
            else
            {
                op.stride[0] = stride[2];
                op.stride[1] = stride[3];            
            }
            
        }
        if(contains(attributes, "dilations"))
        {
            std::vector<std::size_t> dilation;
            copy(attributes.at("dilations").list().i(), std::back_inserter(dilation));
            if(dilation.size() != 4)
            {
                MIGRAPHX_THROW("dilation should have 4 values");
            }
            if(is_nhwc)
            {
                op.dilation[0] = dilation[1];
                op.dilation[1] = dilation[2];
            }
            else
            {
                op.dilation[0] = dilation[2];
                op.dilation[1] = dilation[3];
            }
            
        }

        auto l0 = prog.add_instruction(op::transpose{{2, 3, 0, 1}}, args[1]);
        return prog.add_instruction(op, {args[0], l0});
    }

    instruction_ref parse_pooling(const std::string& name,
                                  attribute_map attributes,
                                  std::vector<instruction_ref> args)
    {
        op::pooling op{starts_with(name, "Max") ? "max" : "average"};

        if(contains(attributes, "padding"))
        {
            const std::string& pad_mode = attributes.at("padding").s();
            if(pad_mode.find("SAME") != std::string::npos)
            {
                op.padding_mode = op::padding_mode_t::same;
            }
            else if(pad_mode.find("VALID") != std::string::npos)
            {
                op.padding_mode = op::padding_mode_t::valid;
            }
        }
        if(contains(attributes, "strides"))
        {
            std::vector<std::size_t> stride;
            copy(attributes.at("stride").list().i(), std::back_inserter(stride));
            if(stride.size() != 4)
            {
                MIGRAPHX_THROW("strides should have 4 values");
            }
            if(is_nhwc)
            {
                op.stride[0] = stride[1];
                op.stride[1] = stride[2];
            }
            else
            {
                op.stride[0] = stride[2];
                op.stride[1] = stride[3];
            }
            
        }
        if(contains(attributes, "ksize"))
        {
            std::vector<std::size_t> ksize;
            copy(attributes.at("ksize").list().i(), std::back_inserter(ksize));
            if(ksize.size() != 4)
            {
                MIGRAPHX_THROW("ksize should have 4 values");
            }
            if(is_nhwc)
            {
                op.lengths[0] = ksize[1];
                op.lengths[1] = ksize[2];
            }
            else
            {
                op.lengths[0] = ksize[2];
                op.lengths[1] = ksize[3];
            }
        }

        return prog.add_instruction(op, std::move(args));
    }

    void parse_from(std::istream& is)
    {
        tensorflow::GraphDef graph;
        if(graph.ParseFromIstream(&is))
        {
            this->parse_graph(graph);
        }
        else
        {
            throw std::runtime_error("Failed reading tf file");
        }
    }

    void parse_graph(const tensorflow::GraphDef& graph)
    {
        nodes = get_nodes(graph, input_nodes);
        for(auto&& input : input_nodes)
        {
            const std::string& name   = input.name();
            attribute_map input_attrs = get_attributes(input);
            shape::type_t shape_type  = parse_type(input_attrs.at("dtype").type());
            std::vector<size_t> dims  = parse_dims(input_attrs.at("shape").shape());
            shape s                   = shape{shape_type, dims};
            instructions[name]        = prog.add_parameter(name, s);
            if(is_nhwc)
            {
                // nhwc to nchw
                prog.add_instruction(migraphx::op::transpose{{0, 3, 1, 2}}, instructions[name]);
            }
        }
        for(auto&& p : nodes)
        {
            this->parse_node(p.first);
        }
    }

    void parse_node(const std::string& name)
    {
        if(instructions.count(name) == 0)
        {
            auto&& node = nodes.at(name);
            std::vector<instruction_ref> args;
            std::cout << name << std::endl;

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
            if(ops.count(node.op()) == 0)
            {
                instructions[name] = prog.add_instruction(unknown{node.op()}, args);
            }
            else
            {
                instructions[name] = ops[node.op()](get_attributes(node), args);
            }
        }
    }

    static attribute_map get_attributes(const tensorflow::NodeDef& node)
    {
        attribute_map result;
        for(auto&& attr : node.attr())
        {
            result[attr.first] = attr.second;
        }
        return result;
    }

    static std::string get_name(const tensorflow::NodeDef& node) { return node.name(); }

    static node_map get_nodes(const tensorflow::GraphDef& graph,
                              std::vector<tensorflow::NodeDef>& input_nodes)
    {
        node_map result;
        for(auto&& node : graph.node())
        {
            auto node_name = get_name(node);
            // assume each node in graph has an associated name
            if(node_name.empty())
                MIGRAPHX_THROW("tf node with no name found");
            result[node_name] = node;
            if(node.op() == "Placeholder")
            {
                input_nodes.push_back(node);
            }
        }
        return result;
    }

    static shape::type_t parse_type(const tensorflow::DataType t)
    {
        shape::type_t shape_type{};
        switch(t)
        {
        case tensorflow::DataType::DT_INVALID:
            break; // throw std::runtime_error("Unsupported type UNDEFINED");
        case tensorflow::DataType::DT_FLOAT: shape_type = shape::float_type; break;
        case tensorflow::DataType::DT_DOUBLE: shape_type = shape::double_type; break;
        case tensorflow::DataType::DT_INT32: shape_type = shape::int32_type; break;
        case tensorflow::DataType::DT_UINT8:
            break; // throw std::runtime_error("Unsupported type UINT8");
        case tensorflow::DataType::DT_INT16: shape_type = shape::int16_type; break;
        case tensorflow::DataType::DT_INT8: shape_type = shape::int8_type; break;
        case tensorflow::DataType::DT_STRING:
            break; // throw std::runtime_error("Unsupported type STRING");
        case tensorflow::DataType::DT_COMPLEX64:
            break; // throw std::runtime_error("Unsupported type COMPLEX64");
        case tensorflow::DataType::DT_INT64: shape_type = shape::int64_type; break;
        case tensorflow::DataType::DT_BOOL:
            break; // throw std::runtime_error("Unsupported type BOOL");
        case tensorflow::DataType::DT_QINT8:
            break; // throw std::runtime_error("Unsupported type QINT8");
        case tensorflow::DataType::DT_QUINT8:
            break; // throw std::runtime_error("Unsupported type QUINT8");
        case tensorflow::DataType::DT_QINT32:
            break; // throw std::runtime_error("Unsupported type QINT32");
        case tensorflow::DataType::DT_BFLOAT16:
            break; // throw std::runtime_error("Unsupported type BFLOAT16");
        case tensorflow::DataType::DT_QINT16:
            break; // throw std::runtime_error("Unsupported type QINT16");
        case tensorflow::DataType::DT_QUINT16:
            break; // throw std::runtime_error("Unsupported type QUINT16");
        case tensorflow::DataType::DT_UINT16: shape_type = shape::uint16_type; break;
        case tensorflow::DataType::DT_COMPLEX128:
            break; // throw std::runtime_error("Unsupported type COMPLEX128");
        case tensorflow::DataType::DT_HALF: shape_type = shape::half_type; break;
        case tensorflow::DataType::DT_RESOURCE:
            break; // throw std::runtime_error("Unsupported type RESOURCE");
        case tensorflow::DataType::DT_VARIANT:
            break; // throw std::runtime_error("Unsupported type VARIANT");
        case tensorflow::DataType::DT_UINT32: shape_type = shape::uint32_type; break;
        case tensorflow::DataType::DT_UINT64: shape_type = shape::uint64_type; break;
        default: break;
        }
        return shape_type;
    }

    static literal parse_tensor(const tensorflow::TensorProto t)
    {
        std::vector<size_t> dims = parse_dims(t.tensor_shape());
        if(dims.empty())
        {
            dims = {1};
        }

        if(!t.tensor_content().empty()) // has raw data
        {
            const std::string& s = t.tensor_content();
            switch(t.dtype())
            {
            case tensorflow::DataType::DT_INVALID: throw std::runtime_error("");
            case tensorflow::DataType::DT_FLOAT:
                return literal{{shape::float_type, dims}, s.data()};
            case tensorflow::DataType::DT_UINT8: throw std::runtime_error("");
            case tensorflow::DataType::DT_INT8: return literal{{shape::int32_type, dims}, s.data()};
            case tensorflow::DataType::DT_UINT16:
                return literal{{shape::int32_type, dims}, s.data()};
            case tensorflow::DataType::DT_INT16:
                return literal{{shape::int32_type, dims}, s.data()};
            case tensorflow::DataType::DT_INT32:
                return literal{{shape::int32_type, dims}, s.data()};
            case tensorflow::DataType::DT_INT64:
                return literal{{shape::int64_type, dims}, s.data()};
            case tensorflow::DataType::DT_STRING: throw std::runtime_error("");
            case tensorflow::DataType::DT_BOOL: return literal{{shape::int32_type, dims}, s.data()};
            case tensorflow::DataType::DT_HALF: return literal{{shape::half_type, dims}, s.data()};
            case tensorflow::DataType::DT_DOUBLE:
                return literal{{shape::double_type, dims}, s.data()};
            case tensorflow::DataType::DT_UINT32: throw std::runtime_error("");
            case tensorflow::DataType::DT_UINT64: throw std::runtime_error("");
            case tensorflow::DataType::DT_COMPLEX64: throw std::runtime_error("");
            case tensorflow::DataType::DT_COMPLEX128: throw std::runtime_error("");
            default: break;
            }
            MIGRAPHX_THROW("Invalid tensor type");
        }
        switch(t.dtype())
        {
        case tensorflow::DataType::DT_INVALID: throw std::runtime_error("");
        case tensorflow::DataType::DT_FLOAT:
            return literal{{shape::float_type, dims}, t.float_val().begin(), t.float_val().end()};
        case tensorflow::DataType::DT_UINT8: throw std::runtime_error("");
        case tensorflow::DataType::DT_INT8:
            return literal{{shape::int32_type, dims}, t.int_val().begin(), t.int_val().end()};
        case tensorflow::DataType::DT_UINT16:
            return literal{{shape::int32_type, dims}, t.int_val().begin(), t.int_val().end()};
        case tensorflow::DataType::DT_INT16:
            return literal{{shape::int32_type, dims}, t.int_val().begin(), t.int_val().end()};
        case tensorflow::DataType::DT_INT32:
            return literal{{shape::int32_type, dims}, t.int_val().begin(), t.int_val().end()};
        case tensorflow::DataType::DT_INT64:
            return literal{{shape::int64_type, dims}, t.int64_val().begin(), t.int64_val().end()};
        case tensorflow::DataType::DT_STRING: throw std::runtime_error("");
        case tensorflow::DataType::DT_BOOL:
            return literal{{shape::int32_type, dims}, t.bool_val().begin(), t.bool_val().end()};
        case tensorflow::DataType::DT_HALF:
            return literal{{shape::half_type, dims}, t.half_val().begin(), t.half_val().end()};
        case tensorflow::DataType::DT_DOUBLE:
            return literal{
                {shape::double_type, dims}, t.double_val().begin(), t.double_val().end()};
        case tensorflow::DataType::DT_UINT32: throw std::runtime_error("");
        case tensorflow::DataType::DT_UINT64: throw std::runtime_error("");
        case tensorflow::DataType::DT_COMPLEX64: throw std::runtime_error("");
        case tensorflow::DataType::DT_COMPLEX128: throw std::runtime_error("");
        default: break;
        }
        MIGRAPHX_THROW("Invalid tensor type");
    }

    static std::vector<size_t> parse_dims(const tensorflow::TensorShapeProto& s)
    {
        std::vector<size_t> dims;
        auto input_dims = s.dim();
        for(auto dim : input_dims)
        {
            dims.push_back(dim.size());
        }
        return dims;
    }
};

program parse_tf(const std::string& name, bool is_nhwc)
{
    std::fstream input(name.c_str(), std::ios::in | std::ios::binary);
    tf_parser parser;
    parser.is_nhwc = is_nhwc;

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
