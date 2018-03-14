
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <onnx.pb.h>
#include <iostream>
#include <fstream>


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
                onnx::GraphProto graph = model.graph();
                std::cout << "Graph name: " << graph.name() << std::endl;
                for(int i=0; i < graph.node_size(); i++) {
                    onnx::NodeProto node = graph.node(i);
                    std::cout << "Layer: " << node.op_type() << std::endl;
                }
            }
        } else {
            std::cout << "Failed reading: " << file << std::endl;
        }

    }
}
