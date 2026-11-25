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
 */
#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/tmp_dir.hpp>
#include <algorithm>
#include <iterator>
#include <string>
#include <vector>
#include "test.hpp"

TEST_CASE(load_save_default)
{
    std::string filename = "migraphx_api_load_save.mxr";
    auto p1              = migraphx::parse_onnx("conv_relu_maxpool_test.onnx");
    auto s1              = p1.get_output_shapes();
    migraphx::save(p1, filename.c_str());
    auto p2 = migraphx::load(filename.c_str());
    auto s2 = p2.get_output_shapes();
    EXPECT(s1.size() == s2.size());
    EXPECT(s1.front() == s2.front());
    EXPECT(p1.sort() == p2.sort());
    std::remove(filename.c_str());
}

TEST_CASE(load_save_json)
{
    std::string filename = "migraphx_api_load_save.json";
    auto p1              = migraphx::parse_onnx("conv_relu_maxpool_test.onnx");
    auto s1              = p1.get_output_shapes();
    migraphx::file_options options;
    options.set_file_format("json");

    migraphx::save(p1, filename.c_str(), options);
    auto p2 = migraphx::load(filename.c_str(), options);
    auto s2 = p2.get_output_shapes();
    EXPECT(s1.size() == s2.size());
    EXPECT(s1.front() == s2.front());
    EXPECT(p1.sort() == p2.sort());
    std::remove(filename.c_str());
}

TEST_CASE(load_save_argument)
{
    migraphx::shape s1{migraphx_shape_float_type, {2, 2}};
    std::vector<float> data{1, 2, 3, 4};
    migraphx::argument a1{s1, data.data()};
    migraphx::argument::save_argument(a1, "migraphx_api_load_save_argument.msgpack");
    migraphx::argument a2 =
        migraphx::argument::load_argument("migraphx_api_load_save_argument.msgpack");
    EXPECT(a1 == a2);
}

TEST_CASE(load_save_external_weights)
{
    migraphx::tmp_dir td{"api-external-weights"};
    auto filename = (td.path / "model.mxr").string();
    auto p1       = migraphx::parse_onnx("conv_relu_maxpool_test.onnx");
    migraphx::file_options options;
    options.set_file_format("msgpack");
    options.set_external_weights_threshold(1);
    options.set_external_weights_file((td.path / "weights.data").string());
    migraphx::save(p1, filename.c_str(), options);

    // The weights go to a generation of the base name, never to the base name itself
    std::vector<migraphx::fs::path> generations;
    std::copy_if(migraphx::fs::directory_iterator{td.path},
                 migraphx::fs::directory_iterator{},
                 std::back_inserter(generations),
                 [](const migraphx::fs::path& p) {
                     auto name = p.filename().string();
                     return name != "weights.data" and name.rfind("weights.", 0) == 0 and
                            p.extension() == ".data";
                 });
    EXPECT(generations.size() == 1);
    EXPECT(not migraphx::fs::exists(td.path / "weights.data"));

    auto p2 = migraphx::load(filename.c_str());
    EXPECT(p1.sort() == p2.sort());

    // A corrupted weights file loads unless its checksums are verified
    auto weights = migraphx::read_buffer(generations.front());
    weights.at(3)++;
    migraphx::write_buffer(generations.front(), weights);
    migraphx::file_options verify;
    verify.set_verify_external_weights(true);
    EXPECT(test::throws([&] { migraphx::load(filename.c_str(), verify); }));
    auto p3 = migraphx::load(filename.c_str());
    EXPECT(p3.get_output_shapes().front() == p1.get_output_shapes().front());
}

TEST_CASE(load_save_reuse_external_data)
{
    migraphx::tmp_dir td{"api-external-weights"};
    migraphx::fs::create_directories(td.path / "a");
    migraphx::fs::create_directories(td.path / "b");
    auto original = (td.path / "a" / "model.mxr").string();
    auto p1       = migraphx::parse_onnx("conv_relu_maxpool_test.onnx");
    migraphx::file_options options;
    options.set_external_weights_threshold(1);
    migraphx::save(p1, original.c_str(), options);
    auto p2 = migraphx::load(original.c_str());

    // Saved elsewhere with reuse, the program references the original weights by absolute path
    auto other = (td.path / "b" / "other.mxr").string();
    migraphx::file_options reuse;
    reuse.set_reuse_external_data();
    migraphx::save(p2, other.c_str(), reuse);
    EXPECT(std::distance(migraphx::fs::directory_iterator{td.path / "b"},
                         migraphx::fs::directory_iterator{}) == 1);

    EXPECT(test::throws([&] { migraphx::load(other.c_str()); }));
    migraphx::file_options allowed;
    allowed.add_external_data_allowed_path((td.path / "a").string());
    EXPECT(p1.sort() == migraphx::load(other.c_str(), allowed).sort());
    migraphx::file_options rooted;
    rooted.set_external_data_path((td.path / "a").string());
    EXPECT(p1.sort() == migraphx::load(other.c_str(), rooted).sort());
}

TEST_CASE(file_options_null_paths)
{
    migraphx_file_options_t options = nullptr;
    EXPECT(migraphx_file_options_create(&options) == migraphx_status_success);
    EXPECT(migraphx_file_options_set_external_weights_file(options, nullptr) ==
           migraphx_status_bad_param);
    EXPECT(migraphx_file_options_set_external_data_path(options, nullptr) ==
           migraphx_status_bad_param);
    EXPECT(migraphx_file_options_add_external_data_allowed_path(options, nullptr) ==
           migraphx_status_bad_param);
    EXPECT(migraphx_file_options_destroy(options) == migraphx_status_success);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
