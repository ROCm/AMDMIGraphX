
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <onnx.pb.h>
#include <iostream>
#include <fstream>
#include <unordered_map>

#include <rtg/program.hpp>

struct unknown
{
    rtg::shape s;
    std::string op;
    std::string name() const
    {
        return "unknown:"+op;
    }
    rtg::shape compute_shape(std::vector<rtg::shape> input) const
    {
        return s;
    }
    rtg::argument compute(std::vector<rtg::argument> input) const
    {
        throw "not computable";
    }
};

std::unordered_map<std::string, onnx::AttributeProto> get_attributes(const onnx::NodeProto& node)
{
    std::unordered_map<std::string, onnx::AttributeProto> result;
    for(auto&& attr:node.attribute())
    {
        result[attr.name()] = attr;
    }
    return result;
}

std::unordered_map<std::string, onnx::NodeProto> get_nodes(const onnx::GraphProto& graph)
{
    std::unordered_map<std::string, onnx::NodeProto> result;
    for(auto&& node:graph.node())
    {
        result[node.name()] = node;
    }
    return result;
}

void parse_graph(onnx::GraphProto graph)
{
    std::cout << "Graph name: " << graph.name() << std::endl;
    for(onnx::NodeProto node:graph.node()) {
        std::cout << "Layer: " << node.op_type() << std::endl;
        std::cout << "    Name: " << node.name() << std::endl;
        if(node.input_size() > 0)
            std::cout << "    Input: " << node.input(0) << std::endl;
        if(node.output_size() > 0)
            std::cout << "    Output: " << node.output(0) << std::endl;
        
        std::cout << "    Attributes: " << std::endl;
        for(auto&& attr:node.attribute())
        {
            std::cout << "        " << attr.name() << std::endl;
        }
    }
}

int main(int argc, char const *argv[])
{
    if(argc > 1)
    {
        std::string file = argv[1];
        std::fstream input(file.c_str(), std::ios::in | std::ios::binary);
        onnx::ModelProto model;
        if(model.ParseFromIstream(&input)) {
            std::cout << "Model version: " << model.model_version() << std::endl;
            std::cout << "Producer name: " << model.producer_name() << std::endl;
            std::cout << "Producer version: " << model.release_producer_version() << std::endl;
            if(model.has_graph()) {
                std::cout << "Model has graph" << std::endl;
                parse_graph(model.graph());
            }
        } else {
            std::cout << "Failed reading: " << file << std::endl;
        }

    }
}
