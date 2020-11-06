#ifndef _TEST_UTILITIES_HPP_
#define _TEST_UTILITIES_HPP_

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <cmath>

#include <migraphx/migraphx.hpp>

template<typename T>
void print_res(const T& res)
{
    for (std::size_t i = 0; i < res.size(); ++i)
    {
        std::cout << std::setprecision(9) << std::setw(12) << res[i] << ", ";
        if ((i + 1) % 6 == 0) {
            std::cout << std::endl;
        }
    }
}

template <class T>
auto get_hash(const T& x)
{
    return std::hash<T>{}(x);
}

void print_vec(std::vector<float>& vec, std::size_t column_size)
{
    for (std::size_t i = 0; i < vec.size(); ++i)
    {
        std::cout << vec[i] << "\t";
        if ((i + 1) % column_size == 0)
            std::cout << std::endl;
    }
    std::cout << std::endl;
}

template<class T>
std::ostream& operator << (std::ostream& os, const std::vector<T>& dims)
{
    os << "{";
    for (auto it = dims.begin(); it != dims.end(); ++it)
    {
        if (it != dims.begin())
        {
            os << ", ";
        }
        os << *it;
    }
    os << "}" << std::endl;

    return os;
}

template <class T>
void assign_value(const T* val, size_t num, std::vector<float>& output)
{
    for (size_t i = 0; i < num; ++i)
    {
        output.push_back(val[i]);
    }
}

void retrieve_argument_data(migraphx::argument& argu, std::vector<float>& output)
{
    auto s = argu.get_shape();
    auto lens = s.lengths();
    auto elem_num = std::accumulate(lens.begin(), lens.end(), 1, std::multiplies<size_t>());
    migraphx_shape_datatype_t type = s.type();
    if (type == migraphx_shape_float_type)
    {
        float *ptr = reinterpret_cast<float*>(argu.data());
        assign_value(ptr, elem_num, output);
    }
    else if (type == migraphx_shape_double_type)
    {
        double *ptr = reinterpret_cast<double*>(argu.data());
        assign_value(ptr, elem_num, output);
    }
    else if (type == migraphx_shape_int32_type)
    {
        int *ptr = reinterpret_cast<int*>(argu.data());
        assign_value(ptr, elem_num, output);
    }
    else if (type == migraphx_shape_int64_type)
    {
        int64_t *ptr = reinterpret_cast<int64_t*>(argu.data());
        assign_value(ptr, elem_num, output);
    }
    else if (type == migraphx_shape_int8_type)
    {
        int8_t *ptr = reinterpret_cast<int8_t*>(argu.data());
        assign_value(ptr, elem_num, output);
    }
    else if (type == migraphx_shape_uint32_type)
    {
        uint32_t *ptr = reinterpret_cast<uint32_t*>(argu.data());
        assign_value(ptr, elem_num, output);
    }
    else if (type == migraphx_shape_uint64_type)
    {
        uint64_t *ptr = reinterpret_cast<uint64_t*>(argu.data());
        assign_value(ptr, elem_num, output);
    }
    else if (type == migraphx_shape_uint8_type)
    {
        uint8_t *ptr = reinterpret_cast<uint8_t*>(argu.data());
        assign_value(ptr, elem_num, output);
    }
    else if (type == migraphx_shape_bool_type)
    {
        bool *ptr = reinterpret_cast<bool*>(argu.data());
        assign_value(ptr, elem_num, output);
    }
    else
    {
        std::cout << "Type not support" << std::endl;
        std::abort();
    }
}

template <class T>
void run_prog(migraphx::program p, const migraphx::target& t, std::vector<std::vector<T>> &resData)
{
    migraphx_compile_options options;
    options.offload_copy = true;
    p.compile(t, options);
    std::cout << "compiled program = " << std::endl;
    p.print();
    std::cout << std::endl;

    std::vector<int> indices = {2, 1, 2, 0, 1, 0};
    std::vector<char> vec_int;
    migraphx::program_parameters m;
    auto param_shapes = p.get_parameter_shapes();
    for (auto &&name : param_shapes.names())
    {
        auto s = param_shapes[name];
        migraphx::argument argu;
        std::cout << "input: " << name << "\'shape = " << s.lengths() << std::endl;
        if (std::string(name) == "indices")
        {
            argu = migraphx::argument(s, indices.data());
        }
        else if (s.type() == migraphx_shape_int32_type or
            s.type() == migraphx_shape_int64_type)
        {
            vec_int.resize(s.bytes(), 0);
            argu = migraphx::argument(s, vec_int.data());
        }
        else
        {
            argu = migraphx::argument::generate(s, get_hash(std::string(name)));
        }
        m.add(name, argu);
    }

    std::cout << "Begin execution ...." << std::endl;
    auto outputs = p.eval(m);
    std::cout << "End execution ...." << std::endl;

    size_t output_num = outputs.size();
    for (size_t i = 0; i < output_num; ++i)
    {
        auto out_argu = outputs[i];
        migraphx::shape out_s = out_argu.get_shape();

        std::cout << "Output_" << i << "_shape = " << out_s.lengths() << std::endl;
        std::cout << "Result_" << i << " = " << std::endl;

        std::vector<float> resTmp;
        retrieve_argument_data(out_argu, resTmp);
        resData.push_back(resTmp);
        print_res(resTmp);
        std::cout << std::endl;
    }
}

template<typename T>
bool compare_results(const T& cpu_res, const T& gpu_res)
{
    bool passed = true;
    std::size_t cpu_size = cpu_res.size();
    float fmax_diff = 0.0f;
    size_t max_index = 0;
    for (std::size_t i = 0; i < cpu_size; i++) {
        auto diff = fabs(cpu_res[i] - gpu_res[i]);
        if (diff > 1.0e-3)
        {
            if (fmax_diff < diff) 
            {
                fmax_diff = diff;
                max_index = i;
                passed = false;
            }
            std::cout << "cpu_result[" << i << "] (" << cpu_res[i] << ") != gpu_result[" << i << "] (" <<
                gpu_res[i] << ")!!!!!!" << std::endl;
        }
    }

    if (!passed)
    {
        size_t i = max_index;
        std::cout << "cpu_result[" << i << "] (" << cpu_res[i] << ") != gpu_result[" << i << "] (" <<
            gpu_res[i] << ")!!!!!!" << std::endl;

        std::cout << "max_diff = " << fmax_diff << std::endl;
    }

    return passed;
}

bool compare_results(const std::vector<int>&cpu_res, const std::vector<int>& gpu_res)
{
    bool passed = true;
    std::size_t cpu_size = cpu_res.size();
    for (std::size_t i = 0; i < cpu_size; i++) {
        if (cpu_res[i] - gpu_res[i] != 0)
        {
            std::cout << "cpu_result[" << i << "] (" << cpu_res[i] << ") != gpu_result[" << i << "] (" <<
                gpu_res[i] << ")!!!!!!" << std::endl;
            passed = false;
        }
    }

    return passed;
}

bool compare_results(const std::vector<int64_t>&cpu_res, const std::vector<int64_t>& gpu_res)
{
    bool passed = true;
    std::size_t cpu_size = cpu_res.size();
    for (std::size_t i = 0; i < cpu_size; i++) {
        if (cpu_res[i] - gpu_res[i] != 0)
        {
            std::cout << "cpu_result[" << i << "] (" << cpu_res[i] << ") != gpu_result[" << i << "] (" <<
                gpu_res[i] << ")!!!!!!" << std::endl;
            passed = false;
        }
    }

    return passed;
}


#endif

