/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/external_data.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/compile_options.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/load_save.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/msgpack.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/tmp_dir.hpp>
#include <algorithm>
#include <iterator>
#include <string>
#include <vector>
#include <test.hpp>

TEST_CASE(compiled_program_external_weights)
{
    migraphx::tmp_dir td{"gpu-external-weights"};
    auto filename = (td.path / "model.mxr").string();
    migraphx::shape s{migraphx::shape::float_type, {33, 17}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto a   = mm->add_literal(migraphx::generate_literal(s, 1));
        auto b   = mm->add_literal(migraphx::generate_literal(s, 2));
        auto add = mm->add_instruction(migraphx::make_op("add"), x, a);
        auto mul = mm->add_instruction(migraphx::make_op("mul"), add, b);
        mm->add_return({mm->add_instruction(migraphx::make_op("relu"), mul)});
    }
    migraphx::compile_options options;
    options.offload_copy = true;
    p1.compile(migraphx::make_target("gpu"), options);

    migraphx::file_options save_options;
    save_options.external_weights_threshold = 1;
    migraphx::save(p1, filename, save_options);
    EXPECT(migraphx::fs::exists(filename + ".data"));

    auto v = migraphx::from_msgpack(migraphx::read_buffer(filename));
    EXPECT(v.at("version").to<int>() == p1.get_program_file_version() + 1);
    const auto& nodes = v.at("modules").at("main").at("nodes");
    std::vector<migraphx::external_data> refs;
    migraphx::transform_if(
        nodes.begin(),
        nodes.end(),
        std::back_inserter(refs),
        [](const migraphx::value& node) {
            return node.at("name").to<std::string>() == "gpu::literal" and
                   node.at("operator").at("data").contains("external");
        },
        [](const migraphx::value& node) {
            return migraphx::from_value<migraphx::external_data>(
                node.at("operator").at("data").at("external"));
        });
    EXPECT(not refs.empty());
    EXPECT(std::all_of(refs.begin(), refs.end(), [](const auto& ref) {
        return ref.offset % migraphx::external_data_alignment == 0 and
               ref.file_path == "model.mxr.data";
    }));

    migraphx::program p2 = migraphx::load(filename);
    auto params          = migraphx::parameter_map{{"x", migraphx::generate_argument(s, 3)}};
    auto r1              = p1.eval(params).back();
    auto r2              = p2.eval(params).back();
    EXPECT(r1 == r2);

    // The same compiled program saved without external weights still round trips
    auto inline_file = (td.path / "inline.mxr").string();
    migraphx::save(p1, inline_file);
    EXPECT(not migraphx::fs::exists(inline_file + ".data"));
    migraphx::program p3 = migraphx::load(inline_file);
    EXPECT(p3.eval(params).back() == r1);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
