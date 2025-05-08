/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 *
 */
#ifndef MIGRAPHX_GUARD_INCLUDE_READ_TF_HPP
#define MIGRAPHX_GUARD_INCLUDE_READ_TF_HPP

#include <pb_files.hpp>

inline migraphx::program read_tf(const std::string& name,
                                 const migraphx::tf_options& options = migraphx::tf_options{})
{
    static auto pb_files{::pb_files()};
    if(pb_files.find(name) == pb_files.end())
    {
        std::cerr << "Can not find TensorFlow Protobuf file by name: " << name
                  << " , aborting the program\n"
                  << std::endl;
        std::abort();
    }
    return migraphx::parse_tf_buffer(std::string{pb_files.at(name)}, options);
}

#endif // MIGRAPHX_GUARD_INCLUDE_READ_TF_HPP
