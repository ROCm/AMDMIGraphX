#ifndef __LOAD_SHAPE_FILE_HPP__
#define __LOAD_SHAPE_FILE_HPP__

#include <string>
#include <migraphx/migraphx.hpp>

migraphx::onnx_options load_name_dim_file(std::string file);
migraphx::onnx_options load_name_val_file(std::string file);

#endif
