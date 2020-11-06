#include <iostream>
#include <fstream>
#include "load_shape_file.hpp"

static std::string process_name_dim_line(std::string line, std::vector<std::size_t>& dims)
{
    std::cout << "line = " << line << std::endl;
    std::size_t start_pos = 0;
    auto pos              = line.find(' ', start_pos);
    auto name             = line.substr(start_pos, pos);
    start_pos             = line.find('{', pos + 1);
    pos                   = line.find('}', start_pos + 1);
    auto dim_str          = line.substr(start_pos + 1, pos - start_pos - 1);
    dims.clear();
    start_pos = 0;
    while(true)
    {
        pos = dim_str.find(',', start_pos);
        if(pos == std::string::npos)
            break;

        auto sub_str = dim_str.substr(start_pos, pos - start_pos);
        std::cout << "sub_str = " << sub_str << std::endl;
        dims.push_back(std::stoi(sub_str));
        start_pos = pos + 1;
    }
    dims.push_back(std::stoi(dim_str.substr(start_pos)));

    return name;
}

migraphx::onnx_options load_name_dim_file(std::string file)
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

        std::vector<std::size_t> dims;
        auto name = process_name_dim_line(line, dims);
        options.set_input_parameter_shape(name, dims);
    }

    return options;
}

static std::string process_name_val_line(std::string line, std::size_t& val)
{
    std::cout << "line = " << line << std::endl;
    std::size_t start_pos = 0;
    auto pos              = line.find(' ', start_pos);
    auto name             = line.substr(start_pos, pos);
    auto val_str          = line.substr(pos + 1);
    val                   = std::stoi(val_str);

    return name;
}

migraphx::onnx_options load_name_val_file(std::string file)
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
        auto name = process_name_val_line(line, val);
        // options.set_input_parameter_shape(name, val);
    }

    return options;
}
