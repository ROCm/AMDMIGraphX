/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "migraphx/compile_options.hpp"
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>
#include <migraphx/stringutils.hpp>

#include <fstream>

inline void readPGMFile(const std::string& fileName, uint8_t* buffer, int32_t inH, int32_t inW)
{
    std::ifstream infile(fileName, std::ifstream::binary);
    EXPECT(infile.is_open());
    std::string magic, w, h, max;
    infile >> magic >> w >> h >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(buffer), inH * inW);
}

TEST_CASE(sampleOnnxMNIST)
{
    migraphx::program p = migraphx::parse_onnx("mnist/mnist.onnx");
    migraphx::compile_options opts;
    opts.offload_copy = true;
    p.compile(migraphx::make_target("gpu"), opts);

    const int inputH = 28;
    const int inputW = 28;

    srand(unsigned(time(nullptr)));
    std::vector<uint8_t> fileData(inputH * inputW);
    int number = rand() % 10;
    readPGMFile("mnist/" + std::to_string(number) + ".pgm", fileData.data(), inputH, inputW);

    // Print an ascii representation
    std::cout << "Input:" << std::endl;
    for(int i = 0; i < inputH * inputW; i++)
    {
        std::cout << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
    }
    std::cout << std::endl;

    std::vector<float> hostData(inputH * inputW);
    for(int i = 0; i < hostData.size(); i++)
    {
        hostData[i] = 1.0 - float(fileData[i] / 255.0);
    }

    migraphx::shape input_shape{migraphx::shape::float_type, {1, 1, 28, 28}};

    migraphx::parameter_map param_map;
    param_map["Input3"] = migraphx::argument(input_shape, reinterpret_cast<void*>(hostData.data()));

    auto result = p.eval(param_map).back();

    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::cout << "Output:" << std::endl;
    std::cout << migraphx::to_string_range(result_vector) << std::endl;

    auto it = std::max_element(result_vector.begin(), result_vector.end());
    EXPECT(*it == result_vector[number]);
}
