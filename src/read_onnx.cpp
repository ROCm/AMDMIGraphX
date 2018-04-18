
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <onnx.pb.h>
#include <iostream>
#include <fstream>
#include <unordered_map>

#include <rtg/program.hpp>

struct unknown
{
    std::string op;
    std::string name() const
    {
        return "unknown:"+op;
    }
    rtg::shape compute_shape(std::vector<rtg::shape> input) const
    {
        if(input.empty()) return {};
        else return input.front();
    }
    rtg::argument compute(std::vector<rtg::argument> input) const
    {
        throw "not computable";
    }
};

struct onnx_parser 
{
    std::unordered_map<std::string, onnx::NodeProto> nodes;
    std::unordered_map<std::string, rtg::instruction*> instructions;
    std::shared_ptr<rtg::program> prog = std::make_shared<rtg::program>();

    void parse_graph(const onnx::GraphProto& graph)
    {
        nodes = get_nodes(graph);
        for(auto&& input:graph.input())
        {
            std::string name = input.name();
            // TODO: Get shape of input parameter
            instructions[name] = prog->add_parameter(name, rtg::shape{});
        }
        for(auto&& p:nodes)
        {
            this->parse_node(p.second.name());
        }
    }

    void parse_node(std::string name)
    {
        if (instructions.count(name) == 0)
        {
            auto&& node = nodes.at(name);
            std::vector<rtg::instruction*> args;
            for(auto&& input:node.input())
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
            instructions[name] = prog->add_instruction(unknown{node.op_type()}, args);
        }
    }

    static std::unordered_map<std::string, onnx::AttributeProto> get_attributes(const onnx::NodeProto& node)
    {
        std::unordered_map<std::string, onnx::AttributeProto> result;
        for(auto&& attr:node.attribute())
        {
            result[attr.name()] = attr;
        }
        return result;
    }

    static std::unordered_map<std::string, onnx::NodeProto> get_nodes(const onnx::GraphProto& graph)
    {
        std::unordered_map<std::string, onnx::NodeProto> result;
        for(auto&& node:graph.node())
        {
            result[node.name()] = node;
            for(auto&& output:node.output())
            {
                result[output] = node;
            }

        }
        return result;
    }
};

std::shared_ptr<rtg::program> parse_onnx(std::istream& is)
{
    onnx_parser parser;
    onnx::ModelProto model;
    if(model.ParseFromIstream(&is)) {
        if(model.has_graph()) {
            parser.parse_graph(model.graph());
        }
    } else {
        throw "Failed reading";
    }
    return parser.prog;
}

int main(int argc, char const *argv[])
{
    if(argc > 1)
    {
        std::string file = argv[1];
        std::fstream input(file.c_str(), std::ios::in | std::ios::binary);
        auto prog = parse_onnx(input);
        prog->print();
    }
}
