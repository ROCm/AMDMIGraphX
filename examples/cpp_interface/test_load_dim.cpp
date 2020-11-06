#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <migraphx/migraphx.hpp>

void load_onnx_file(std::string file_name, migraphx::onnx_options options)
{
    auto prog = migraphx::parse_onnx(file_name.c_str(), options);
    std::cout << "Load program is: " << std::endl;
    prog.print();
    std::cout << std::endl;

    migraphx::target t = migraphx::target("gpu");
    prog.compile(t);

    std::cout << "Compiled program is: " << std::endl;
    prog.print();
    std::cout << std::endl;
}

std::string process_one_line(std::string line, std::size_t& val)
{
    std::cout << "line = " << line << std::endl;
    std::size_t start_pos = 0;
    auto pos              = line.find(' ', start_pos);
    auto name             = line.substr(start_pos, pos);
    auto val_str          = line.substr(pos + 1);
    val                   = std::stoi(val_str);

    return name;
}

migraphx::onnx_options load_option_file(std::string file)
{
    migraphx::onnx_options options;
    std::ifstream ifs(file);
    if(!ifs.is_open())
    {
        return options;
    }

    std::string line;
    while(std::getline(ifs, line))
    {
        if(line.empty())
            break;

        std::size_t val;
        auto name = process_one_line(line, val);
        options.set_input_parameter_shape(name, val);
    }

    return options;
}

int main(int argc, char** argv)
{
    if(argc != 3)
    {
        std::cout << "Usage: " << argv[0] << " onnxfile optionfile" << std::endl;
        return 0;
    }

    auto options = load_option_file(argv[2]);

    load_onnx_file(argv[1], options);

    return 0;
}
