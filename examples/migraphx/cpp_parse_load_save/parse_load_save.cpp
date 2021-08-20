#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

// MIGraphX C++ API
#include <migraphx/migraphx.hpp>

char* getCmdOption(char**, char**, const std::string&);

bool cmdOptionExists(char**, char**, const std::string&);

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <input_file> "
                  << "[options]" << std::endl;
        std::cout << "options:" << std::endl;
        std::cout << "\t--parse onnx" << std::endl;
        std::cout << "\t--load  json/msgpack" << std::endl;
        std::cout << "\t--save  <output_file>" << std::endl;
        return 0;
    }

    char* parse_arg        = getCmdOption(argv + 2, argv + argc, "--parse");
    char* load_arg         = getCmdOption(argv + 2, argv + argc, "--load");
    char* save_arg         = getCmdOption(argv + 2, argv + argc, "--save");
    const char* input_file = argv[1];

    migraphx::program p;

    if(cmdOptionExists(argv + 2, argv + argc, "--parse") ||
       !cmdOptionExists(argv + 2, argv + argc, "--load"))
    {
        std::cout << "Parsing ONNX File" << std::endl;
        migraphx::onnx_options options;
        p = parse_onnx(input_file, options);
    }
    else if(load_arg != nullptr)
    {
        std::cout << "Loading Graph File" << std::endl;
        std::string format = load_arg;
        if(format == "json")
        {
            migraphx_file_options options;
            options.format = "json";
            p              = migraphx::load(input_file, options);
        }
        else if(format == "msgpack")
        {
            migraphx_file_options options;
            options.format = "msgpack";
            p              = migraphx::load(input_file, options);
        }
        else
            p = migraphx::load(input_file);
    }
    else
    {
        std::cout << "Error: Incorrect Usage" << std::endl;
        std::cout << "Usage: " << argv[0] << " <input_file> "
                  << "[options]" << std::endl;
        std::cout << "options:" << std::endl;
        std::cout << "\t--parse onnx" << std::endl;
        std::cout << "\t--load  json/msgpack" << std::endl;
        std::cout << "\t--save  <output_file>" << std::endl;
        return 0;
    }

    std::cout << "Input Graph: " << std::endl;
    p.print();
    std::cout << std::endl;

    if(cmdOptionExists(argv + 2, argv + argc, "--save"))
    {
        std::cout << "Saving program..." << std::endl;
        std::string output_file;
        output_file = save_arg == nullptr ? "out" : save_arg;
        output_file.append(".msgpack");

        migraphx_file_options options;
        options.format = "msgpack";
        migraphx::save(p, output_file.c_str(), options);
        std::cout << "Program has been saved as ./" << output_file << std::endl;
    }

    return 0;
}

char* getCmdOption(char** begin, char** end, const std::string& option)
{
    char** itr = std::find(begin, end, option);
    if(itr != end && ++itr != end)
    {
        return *itr;
    }

    return nullptr;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}
