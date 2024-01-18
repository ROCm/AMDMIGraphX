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
#include <migraphx/simplify_dyn_ops.hpp>
#include <migraphx/split_single_dyn_dim.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/program.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/make_op.hpp>
#include <test.hpp>

void run_pass(migraphx::program& p)
{
    // The find_resize_static matcher in simplify_dyn_ops has a runtime dependency on split_single_dyn_dim.
    //  split_single_dyn_dim must be run before find_resize_static in order to convert a dynamic batch size
    //  to static input dimensions which are
    //  only known at runtime. 
    migraphx::run_passes(p,  {migraphx::split_single_dyn_dim{},  migraphx::simplify_dyn_ops{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(resize_dynamic)
{
    migraphx::program p0;
    {
        migraphx::module* m0 = p0.get_main_module();
        {
            std::vector<int64_t> ds = {1, 1, 4, 6};
            migraphx::shape ss{migraphx::shape::int64_type, {4}};

            auto li = m0->add_literal(migraphx::literal{ss, ds});
            m0->add_instruction(migraphx::make_op("undefined"));

            migraphx::shape sx{migraphx::shape::float_type, {{1, 4, {1, 4}}, {1, 1}, {2, 2}, {2, 2}}};
            auto inx = m0->add_parameter("X", sx);

            auto r =
                m0->add_instruction(migraphx::make_op("resize",
                                                    {{"mode", "nearest"},
                                                    {"nearest_mode", "floor"},
                                                    {"scales", {1., 2.1, 3.1, 4.1}},
                                                    {"coordinate_transformation_mode", "asymmetric"}}),
                                    inx,
                                    li);

            m0->add_return({r});
        }
        run_pass(p0);
    std::cout << "\n\n   ******               module after passes test: m0= \n";
    m0->debug_print();



         std::cout << "\n\n   ******  the program \n";   
        p0.debug_print();
    }

    migraphx::program p1;
    {
        migraphx::module* m1 = p1.get_main_module();
        {

            migraphx::shape sx{migraphx::shape::int64_type, {1, 1, 2, 2}};
            auto inx = m1->add_parameter("X", sx);

            // TODO:  verification that these indices are correctly calculated.  I just copied them.
            std::vector<float> indices = {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3};
            migraphx::shape ss{migraphx::shape::int32_type, {1, 1, 4, 6}};
            auto li = m1->insert_literal(inx, migraphx::literal{ss, indices});

            auto reshape_ins = m1->add_instruction(
                migraphx::make_op("reshape", {{"dims", {4}}}), inx);
            auto gather_ins = m1->add_instruction(
                migraphx::make_op("gather", {{"axis", 0}}), reshape_ins, li);
            m1->add_return({gather_ins});
        
        }
        std::cout << "\n\n   ******               end of resize test: m1= \n";
        m1->debug_print();


        // TODO:  Need to add the submodules, for starters...
        // How can I run a "verify" test to make sure the indices are what we really want?
    }
    EXPECT(p0 == p1);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
