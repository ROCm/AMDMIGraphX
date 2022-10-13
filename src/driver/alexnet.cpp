
/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/json.hpp>
#include "models.hpp"
namespace migraphx {
namespace driver {
inline namespace MIGRAPHX_INLINE_NS {
migraphx::program alexnet(unsigned batch) // NOLINT(readability-function-size)
{
    migraphx::program p;
    migraphx::module_ref mmain = p.get_main_module();
    auto x_main_module_0       = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 0)));
    auto x_main_module_1       = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 1)));
    auto x_main_module_2       = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 2)));
    auto x_data_0              = mmain->add_parameter(
        "data_0", migraphx::shape{migraphx::shape::float_type, {batch, 3, 224, 224}});
    auto x_main_module_4 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1000, 4096}}, 3));
    auto x_main_module_5 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1000}}, 4));
    auto x_main_module_6 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {4096, 4096}}, 5));
    auto x_main_module_7 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {4096}}, 6)));
    auto x_main_module_8 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {4096, 9216}}, 7));
    auto x_main_module_9 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {4096}}, 8));
    auto x_main_module_10 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 192, 3, 3}}, 9));
    auto x_main_module_11 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 10));
    auto x_main_module_12 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 192, 3, 3}}, 11));
    auto x_main_module_13 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 12));
    auto x_main_module_14 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 256, 3, 3}}, 13));
    auto x_main_module_15 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 14));
    auto x_main_module_16 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 48, 5, 5}}, 15));
    auto x_main_module_17 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 16)));
    auto x_main_module_18 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 3, 11, 11}}, 17));
    auto x_main_module_19 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 18));
    auto x_main_module_20 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[4,"
                               "4],use_dynamic_same_auto_pad:0}"),
        x_data_0,
        x_main_module_18);
    auto x_main_module_21 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,96,54,54]}"), x_main_module_19);
    auto x_main_module_22 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_20, x_main_module_21);
    auto x_main_module_23 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_22);
    auto x_main_module_24 = mmain->add_instruction(
        migraphx::make_json_op("lrn", "{alpha:9.999999747378752e-05,beta:0.75,bias:1.0,size:5}"),
        x_main_module_23);
    auto x_main_module_25 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:1,padding:[0,0,0,0],stride:[2,2]}"),
        x_main_module_24);
    auto x_main_module_26 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:2,padding:[2,2,2,2],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_25,
        x_main_module_16);
    auto x_main_module_27 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,256,26,26]}"), x_main_module_17);
    auto x_main_module_28 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_26, x_main_module_27);
    auto x_main_module_29 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_28);
    auto x_main_module_30 = mmain->add_instruction(
        migraphx::make_json_op("lrn", "{alpha:9.999999747378752e-05,beta:0.75,bias:1.0,size:5}"),
        x_main_module_29);
    auto x_main_module_31 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:1,padding:[0,0,0,0],stride:[2,2]}"),
        x_main_module_30);
    auto x_main_module_32 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_31,
        x_main_module_14);
    auto x_main_module_33 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,384,12,12]}"), x_main_module_15);
    auto x_main_module_34 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_32, x_main_module_33);
    auto x_main_module_35 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_34);
    auto x_main_module_36 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:2,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_35,
        x_main_module_12);
    auto x_main_module_37 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,384,12,12]}"), x_main_module_13);
    auto x_main_module_38 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_36, x_main_module_37);
    auto x_main_module_39 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_38);
    auto x_main_module_40 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:2,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_39,
        x_main_module_10);
    auto x_main_module_41 = mmain->add_instruction(
        migraphx::make_json_op("broadcast", "{axis:1,out_lens:[1,256,12,12]}"), x_main_module_11);
    auto x_main_module_42 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_40, x_main_module_41);
    auto x_main_module_43 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_42);
    auto x_main_module_44 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:1,padding:[0,0,1,1],stride:[2,2]}"),
        x_main_module_43);
    auto x_main_module_45 = mmain->add_instruction(
        migraphx::make_json_op("reshape", "{dims:[1,9216]}"), x_main_module_44);
    auto x_main_module_46 = mmain->add_instruction(
        migraphx::make_json_op("transpose", "{permutation:[1,0]}"), x_main_module_8);
    auto x_main_module_47 =
        mmain->add_instruction(migraphx::make_op("dot"), x_main_module_45, x_main_module_46);
    auto x_main_module_48 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,4096]}"), x_main_module_9);
    auto x_main_module_49 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,4096]}"), x_main_module_2);
    auto x_main_module_50 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_48, x_main_module_49);
    auto x_main_module_51 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_47, x_main_module_50);
    auto x_main_module_52 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_51);
    auto x_main_module_53 = mmain->add_instruction(migraphx::make_op("identity"), x_main_module_52);
    auto x_main_module_54 = mmain->add_instruction(
        migraphx::make_json_op("transpose", "{permutation:[1,0]}"), x_main_module_6);
    auto x_main_module_55 =
        mmain->add_instruction(migraphx::make_op("dot"), x_main_module_53, x_main_module_54);
    auto x_main_module_56 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,4096]}"), x_main_module_7);
    auto x_main_module_57 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,4096]}"), x_main_module_1);
    auto x_main_module_58 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_56, x_main_module_57);
    auto x_main_module_59 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_55, x_main_module_58);
    auto x_main_module_60 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_59);
    auto x_main_module_61 = mmain->add_instruction(migraphx::make_op("identity"), x_main_module_60);
    auto x_main_module_62 = mmain->add_instruction(
        migraphx::make_json_op("transpose", "{permutation:[1,0]}"), x_main_module_4);
    auto x_main_module_63 =
        mmain->add_instruction(migraphx::make_op("dot"), x_main_module_61, x_main_module_62);
    auto x_main_module_64 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1000]}"), x_main_module_5);
    auto x_main_module_65 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1000]}"), x_main_module_0);
    auto x_main_module_66 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_64, x_main_module_65);
    auto x_main_module_67 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_63, x_main_module_66);
    auto x_main_module_68 =
        mmain->add_instruction(migraphx::make_json_op("softmax", "{axis:1}"), x_main_module_67);
    mmain->add_return({x_main_module_68});

    return p;
}
} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx
