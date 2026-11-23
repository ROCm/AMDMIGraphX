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
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <test.hpp>

// Transposes what a submodule returns, which changes the shape of its outputs
struct transpose_submodule_outputs
{
    std::string name() const { return "transpose_submodule_outputs"; }
    void apply(migraphx::module& m) const
    {
        if(m.name() == "main")
            return;
        auto ret = std::prev(m.end());
        std::vector<migraphx::instruction_ref> outputs;
        for(auto output : ret->inputs())
            outputs.push_back(m.insert_instruction(
                ret, migraphx::make_op("transpose", {{"permutation", {1, 0}}}), output));
        m.replace_return(outputs);
    }
};

TEST_CASE(if_shape_follows_branch_outputs)
{
    // A pass that changes the outputs of the branches of an if also changes the shape of the if
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    auto cond   = mm->add_parameter("cond", migraphx::shape{migraphx::shape::bool_type});
    auto x      = mm->add_parameter("x", s);
    auto branch = [&](const std::string& name, const std::string& op) {
        auto* bm = p.create_module(name);
        bm->add_return({bm->add_instruction(migraphx::make_op(op), x)});
        return bm;
    };
    auto* then_mod = branch("If_0_if", "relu");
    auto* else_mod = branch("If_0_else", "neg");
    auto ret       = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
    auto out       = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), ret);
    mm->add_return({out});

    migraphx::run_passes(p, {transpose_submodule_outputs{}});
    EXPECT(mm->validate() == mm->end());
    EXPECT(ret->get_shape().sub_shapes().front().lens() == std::vector<std::size_t>{3, 2});
    EXPECT(out->get_shape().lens() == std::vector<std::size_t>{3, 2});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
