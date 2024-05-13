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

#ifndef MIGRAPHX_GUARD_TEST_ONNX_ONNX_TEST_HPP
#define MIGRAPHX_GUARD_TEST_ONNX_ONNX_TEST_HPP

#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/rewrite_quantization.hpp>
#include <migraphx/eliminate_identity.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/common.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/ranges.hpp>
#include <onnx_files.hpp>
#include <weight_files.hpp>
#include <test.hpp>
#include <thread>

struct weight_file
{
    std::unique_ptr<migraphx::fs::path> path = nullptr;

    weight_file() = default;

    explicit weight_file(const std::pair<std::string_view, std::string_view>& pair)
        : path{std::make_unique<migraphx::fs::path>(pair.first)}
    {
        if(path->has_parent_path())
        {
            migraphx::fs::path parent_path = path->parent_path();
            migraphx::fs::create_directories(parent_path);
        }
        migraphx::write_buffer(*path, pair.second.begin(), pair.second.length());
    }

    weight_file(weight_file&& copy) noexcept : path(std::move(copy.path)){};

    ~weight_file()
    {
        if(path != nullptr)
        {
            constexpr int max_retries_count = 5;
            for([[maybe_unused]] auto count : migraphx::range(max_retries_count))
            {
                std::error_code ec;
                migraphx::fs::remove_all(*path, ec);
                if(not ec)
                    break;
                std::cerr << "Failed to remove " << *path << ": " << ec.message() << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(125));
            }
        }
    }
};

inline static bool read_weight_files()
{
    static auto weight_files{::weight_files()};
    static std::vector<weight_file> weights;
    for(const auto& i : weight_files)
    {
        weights.push_back(weight_file{i});
    }
    return true;
}

inline migraphx::program read_onnx(const std::string& name,
                                   const migraphx::onnx_options& options = migraphx::onnx_options{})
{
    static auto onnx_files{::onnx_files()};
    static bool read_once = read_weight_files();
    (void)(read_once);
    auto prog = migraphx::parse_onnx_buffer(std::string{onnx_files[name]}, options);
    return prog;
}

inline migraphx::program optimize_onnx(const std::string& name, bool run_passes = false)
{
    migraphx::onnx_options options;
    options.skip_unknown_operators = true;
    auto prog                      = read_onnx(name, options);
    auto* mm                       = prog.get_main_module();
    if(run_passes)
        migraphx::run_passes(*mm,
                             {migraphx::rewrite_quantization{}, migraphx::dead_code_elimination{}});

    // remove the last identity instruction
    auto last_ins = std::prev(mm->end());
    if(last_ins->name() == "@return")
    {
        mm->remove_instruction(last_ins);
    }

    return prog;
}

#endif
