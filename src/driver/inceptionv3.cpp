
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
#include "models.hpp"
namespace migraphx {
namespace driver {
inline namespace MIGRAPHX_INLINE_NS {
migraphx::program inceptionv3(unsigned batch) // NOLINT(readability-function-size)
{
    migraphx::program p;
    migraphx::module_ref mmain = p.get_main_module();
    auto x_main_module_0       = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 0)));
    auto x_0                   = mmain->add_parameter(
        "0", migraphx::shape{migraphx::shape::float_type, {batch, 3, 299, 299}});
    auto x_main_module_2 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1000}}, 1));
    auto x_main_module_3 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1000, 2048}}, 2));
    auto x_main_module_4 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 3)));
    auto x_main_module_5 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 4));
    auto x_main_module_6 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 5));
    auto x_main_module_7  = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 6)));
    auto x_main_module_8  = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 2048, 1, 1}}, 7));
    auto x_main_module_9  = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 8)));
    auto x_main_module_10 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 9));
    auto x_main_module_11 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 10));
    auto x_main_module_12 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 11)));
    auto x_main_module_13 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 3, 1}}, 12));
    auto x_main_module_14 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 13)));
    auto x_main_module_15 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 14));
    auto x_main_module_16 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 15));
    auto x_main_module_17 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 16)));
    auto x_main_module_18 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 1, 3}}, 17));
    auto x_main_module_19 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 18)));
    auto x_main_module_20 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 19));
    auto x_main_module_21 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 20));
    auto x_main_module_22 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 21)));
    auto x_main_module_23 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 448, 3, 3}}, 22));
    auto x_main_module_24 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 23)));
    auto x_main_module_25 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 24));
    auto x_main_module_26 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 25));
    auto x_main_module_27 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 26)));
    auto x_main_module_28 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {448, 2048, 1, 1}}, 27));
    auto x_main_module_29 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 28)));
    auto x_main_module_30 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 29)));
    auto x_main_module_31 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 30));
    auto x_main_module_32 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 31)));
    auto x_main_module_33 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 3, 1}}, 32));
    auto x_main_module_34 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 33)));
    auto x_main_module_35 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 34)));
    auto x_main_module_36 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 35));
    auto x_main_module_37 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 36)));
    auto x_main_module_38 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 1, 3}}, 37));
    auto x_main_module_39 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 38)));
    auto x_main_module_40 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 39));
    auto x_main_module_41 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 40));
    auto x_main_module_42 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 41)));
    auto x_main_module_43 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 2048, 1, 1}}, 42));
    auto x_main_module_44 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 43)));
    auto x_main_module_45 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 44));
    auto x_main_module_46 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 45));
    auto x_main_module_47 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 46)));
    auto x_main_module_48 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {320, 2048, 1, 1}}, 47));
    auto x_main_module_49 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 48)));
    auto x_main_module_50 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 49));
    auto x_main_module_51 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 50));
    auto x_main_module_52 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 51)));
    auto x_main_module_53 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 1280, 1, 1}}, 52));
    auto x_main_module_54 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 53)));
    auto x_main_module_55 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 54));
    auto x_main_module_56 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 55));
    auto x_main_module_57 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 56)));
    auto x_main_module_58 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 3, 1}}, 57));
    auto x_main_module_59 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 58)));
    auto x_main_module_60 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 59));
    auto x_main_module_61 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 60));
    auto x_main_module_62 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 61)));
    auto x_main_module_63 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 1, 3}}, 62));
    auto x_main_module_64 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 63)));
    auto x_main_module_65 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 64));
    auto x_main_module_66 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 65));
    auto x_main_module_67 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 66)));
    auto x_main_module_68 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 448, 3, 3}}, 67));
    auto x_main_module_69 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 68)));
    auto x_main_module_70 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 69));
    auto x_main_module_71 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 70));
    auto x_main_module_72 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 71)));
    auto x_main_module_73 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {448, 1280, 1, 1}}, 72));
    auto x_main_module_74 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 73)));
    auto x_main_module_75 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 74));
    auto x_main_module_76 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 75));
    auto x_main_module_77 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 76)));
    auto x_main_module_78 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 3, 1}}, 77));
    auto x_main_module_79 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 78)));
    auto x_main_module_80 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 79));
    auto x_main_module_81 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 80));
    auto x_main_module_82 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 81)));
    auto x_main_module_83 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 1, 3}}, 82));
    auto x_main_module_84 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 83)));
    auto x_main_module_85 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 84));
    auto x_main_module_86 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 85));
    auto x_main_module_87 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 86)));
    auto x_main_module_88 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 1280, 1, 1}}, 87));
    auto x_main_module_89 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 88)));
    auto x_main_module_90 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 89));
    auto x_main_module_91 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 90));
    auto x_main_module_92 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 91)));
    auto x_main_module_93 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {320, 1280, 1, 1}}, 92));
    auto x_main_module_94 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 93)));
    auto x_main_module_95 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 94));
    auto x_main_module_96 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 95));
    auto x_main_module_97  = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 96)));
    auto x_main_module_98  = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 3, 3}}, 97));
    auto x_main_module_99  = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 98)));
    auto x_main_module_100 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 99));
    auto x_main_module_101 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 100));
    auto x_main_module_102 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 101)));
    auto x_main_module_103 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 7, 1}}, 102));
    auto x_main_module_104 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 103)));
    auto x_main_module_105 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 104));
    auto x_main_module_106 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 105));
    auto x_main_module_107 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 106)));
    auto x_main_module_108 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 1, 7}}, 107));
    auto x_main_module_109 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 108)));
    auto x_main_module_110 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 109));
    auto x_main_module_111 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 110));
    auto x_main_module_112 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 111)));
    auto x_main_module_113 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 112));
    auto x_main_module_114 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 113)));
    auto x_main_module_115 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 114));
    auto x_main_module_116 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 115));
    auto x_main_module_117 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 116)));
    auto x_main_module_118 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {320, 192, 3, 3}}, 117));
    auto x_main_module_119 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 118)));
    auto x_main_module_120 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 119));
    auto x_main_module_121 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 120));
    auto x_main_module_122 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 121)));
    auto x_main_module_123 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 122));
    auto x_main_module_124 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 123)));
    auto x_main_module_125 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 124));
    auto x_main_module_126 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 125));
    auto x_main_module_127 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 126)));
    auto x_main_module_128 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 127));
    auto x_main_module_129 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 128)));
    auto x_main_module_130 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 129));
    auto x_main_module_131 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 130));
    auto x_main_module_132 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 131)));
    auto x_main_module_133 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 1, 7}}, 132));
    auto x_main_module_134 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 133)));
    auto x_main_module_135 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 134));
    auto x_main_module_136 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 135));
    auto x_main_module_137 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 136)));
    auto x_main_module_138 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 7, 1}}, 137));
    auto x_main_module_139 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 138)));
    auto x_main_module_140 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 139));
    auto x_main_module_141 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 140));
    auto x_main_module_142 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 141)));
    auto x_main_module_143 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 1, 7}}, 142));
    auto x_main_module_144 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 143)));
    auto x_main_module_145 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 144));
    auto x_main_module_146 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 145));
    auto x_main_module_147 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 146)));
    auto x_main_module_148 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 7, 1}}, 147));
    auto x_main_module_149 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 148)));
    auto x_main_module_150 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 149));
    auto x_main_module_151 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 150));
    auto x_main_module_152 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 151)));
    auto x_main_module_153 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 152));
    auto x_main_module_154 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 153)));
    auto x_main_module_155 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 154));
    auto x_main_module_156 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 155));
    auto x_main_module_157 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 156)));
    auto x_main_module_158 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 7, 1}}, 157));
    auto x_main_module_159 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 158)));
    auto x_main_module_160 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 159));
    auto x_main_module_161 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 160));
    auto x_main_module_162 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 161)));
    auto x_main_module_163 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 1, 7}}, 162));
    auto x_main_module_164 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 163)));
    auto x_main_module_165 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 164));
    auto x_main_module_166 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 165));
    auto x_main_module_167 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 166)));
    auto x_main_module_168 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 167));
    auto x_main_module_169 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 168)));
    auto x_main_module_170 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 169));
    auto x_main_module_171 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 170));
    auto x_main_module_172 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 171)));
    auto x_main_module_173 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 172));
    auto x_main_module_174 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 173)));
    auto x_main_module_175 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 174));
    auto x_main_module_176 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 175));
    auto x_main_module_177 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 176)));
    auto x_main_module_178 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 177));
    auto x_main_module_179 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 178)));
    auto x_main_module_180 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 179));
    auto x_main_module_181 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 180));
    auto x_main_module_182 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 181)));
    auto x_main_module_183 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 160, 1, 7}}, 182));
    auto x_main_module_184 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 183)));
    auto x_main_module_185 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 184));
    auto x_main_module_186 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 185));
    auto x_main_module_187 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 186)));
    auto x_main_module_188 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 7, 1}}, 187));
    auto x_main_module_189 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 188)));
    auto x_main_module_190 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 189));
    auto x_main_module_191 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 190));
    auto x_main_module_192 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 191)));
    auto x_main_module_193 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 1, 7}}, 192));
    auto x_main_module_194 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 193)));
    auto x_main_module_195 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 194));
    auto x_main_module_196 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 195));
    auto x_main_module_197 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 196)));
    auto x_main_module_198 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 7, 1}}, 197));
    auto x_main_module_199 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 198)));
    auto x_main_module_200 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 199));
    auto x_main_module_201 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 200));
    auto x_main_module_202 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 201)));
    auto x_main_module_203 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 768, 1, 1}}, 202));
    auto x_main_module_204 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 203)));
    auto x_main_module_205 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 204));
    auto x_main_module_206 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 205));
    auto x_main_module_207 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 206)));
    auto x_main_module_208 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 160, 7, 1}}, 207));
    auto x_main_module_209 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 208)));
    auto x_main_module_210 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 209));
    auto x_main_module_211 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 210));
    auto x_main_module_212 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 211)));
    auto x_main_module_213 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 1, 7}}, 212));
    auto x_main_module_214 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 213)));
    auto x_main_module_215 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 214));
    auto x_main_module_216 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 215));
    auto x_main_module_217 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 216)));
    auto x_main_module_218 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 768, 1, 1}}, 217));
    auto x_main_module_219 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 218)));
    auto x_main_module_220 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 219));
    auto x_main_module_221 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 220));
    auto x_main_module_222 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 221)));
    auto x_main_module_223 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 222));
    auto x_main_module_224 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 223)));
    auto x_main_module_225 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 224));
    auto x_main_module_226 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 225));
    auto x_main_module_227 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 226)));
    auto x_main_module_228 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 227));
    auto x_main_module_229 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 228)));
    auto x_main_module_230 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 229));
    auto x_main_module_231 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 230));
    auto x_main_module_232 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 231)));
    auto x_main_module_233 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 160, 1, 7}}, 232));
    auto x_main_module_234 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 233)));
    auto x_main_module_235 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 234));
    auto x_main_module_236 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 235));
    auto x_main_module_237 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 236)));
    auto x_main_module_238 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 7, 1}}, 237));
    auto x_main_module_239 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 238)));
    auto x_main_module_240 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 239));
    auto x_main_module_241 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 240));
    auto x_main_module_242 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 241)));
    auto x_main_module_243 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 1, 7}}, 242));
    auto x_main_module_244 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 243)));
    auto x_main_module_245 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 244));
    auto x_main_module_246 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 245));
    auto x_main_module_247 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 246)));
    auto x_main_module_248 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 7, 1}}, 247));
    auto x_main_module_249 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 248)));
    auto x_main_module_250 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 249));
    auto x_main_module_251 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 250));
    auto x_main_module_252 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 251)));
    auto x_main_module_253 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 768, 1, 1}}, 252));
    auto x_main_module_254 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 253)));
    auto x_main_module_255 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 254));
    auto x_main_module_256 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 255));
    auto x_main_module_257 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 256)));
    auto x_main_module_258 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 160, 7, 1}}, 257));
    auto x_main_module_259 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 258)));
    auto x_main_module_260 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 259));
    auto x_main_module_261 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 260));
    auto x_main_module_262 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 261)));
    auto x_main_module_263 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 1, 7}}, 262));
    auto x_main_module_264 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 263)));
    auto x_main_module_265 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 264));
    auto x_main_module_266 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 265));
    auto x_main_module_267 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 266)));
    auto x_main_module_268 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 768, 1, 1}}, 267));
    auto x_main_module_269 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 268)));
    auto x_main_module_270 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 269));
    auto x_main_module_271 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 270));
    auto x_main_module_272 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 271)));
    auto x_main_module_273 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 272));
    auto x_main_module_274 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 273)));
    auto x_main_module_275 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 274));
    auto x_main_module_276 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 275));
    auto x_main_module_277 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 276)));
    auto x_main_module_278 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 277));
    auto x_main_module_279 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 278)));
    auto x_main_module_280 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 279));
    auto x_main_module_281 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 280));
    auto x_main_module_282 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 281)));
    auto x_main_module_283 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 128, 1, 7}}, 282));
    auto x_main_module_284 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 283)));
    auto x_main_module_285 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 284));
    auto x_main_module_286 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 285));
    auto x_main_module_287 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 286)));
    auto x_main_module_288 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 7, 1}}, 287));
    auto x_main_module_289 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 288)));
    auto x_main_module_290 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 289));
    auto x_main_module_291 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 290));
    auto x_main_module_292 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 291)));
    auto x_main_module_293 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 1, 7}}, 292));
    auto x_main_module_294 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 293)));
    auto x_main_module_295 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 294));
    auto x_main_module_296 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 295));
    auto x_main_module_297 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 296)));
    auto x_main_module_298 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 7, 1}}, 297));
    auto x_main_module_299 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 298)));
    auto x_main_module_300 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 299));
    auto x_main_module_301 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 300));
    auto x_main_module_302 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 301)));
    auto x_main_module_303 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 768, 1, 1}}, 302));
    auto x_main_module_304 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 303)));
    auto x_main_module_305 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 304));
    auto x_main_module_306 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 305));
    auto x_main_module_307 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 306)));
    auto x_main_module_308 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 128, 7, 1}}, 307));
    auto x_main_module_309 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 308)));
    auto x_main_module_310 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 309));
    auto x_main_module_311 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 310));
    auto x_main_module_312 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 311)));
    auto x_main_module_313 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 1, 7}}, 312));
    auto x_main_module_314 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 313)));
    auto x_main_module_315 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 314));
    auto x_main_module_316 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 315));
    auto x_main_module_317 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 316)));
    auto x_main_module_318 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 768, 1, 1}}, 317));
    auto x_main_module_319 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 318)));
    auto x_main_module_320 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 319));
    auto x_main_module_321 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 320));
    auto x_main_module_322 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 321)));
    auto x_main_module_323 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 322));
    auto x_main_module_324 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 323)));
    auto x_main_module_325 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 324));
    auto x_main_module_326 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 325));
    auto x_main_module_327 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 326)));
    auto x_main_module_328 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 96, 3, 3}}, 327));
    auto x_main_module_329 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 328)));
    auto x_main_module_330 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 329));
    auto x_main_module_331 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 330));
    auto x_main_module_332 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 331)));
    auto x_main_module_333 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 64, 3, 3}}, 332));
    auto x_main_module_334 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 333)));
    auto x_main_module_335 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 334));
    auto x_main_module_336 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 335));
    auto x_main_module_337 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 336)));
    auto x_main_module_338 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 288, 1, 1}}, 337));
    auto x_main_module_339 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 338)));
    auto x_main_module_340 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 339));
    auto x_main_module_341 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 340));
    auto x_main_module_342 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 341)));
    auto x_main_module_343 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 288, 3, 3}}, 342));
    auto x_main_module_344 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 343)));
    auto x_main_module_345 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 344));
    auto x_main_module_346 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 345));
    auto x_main_module_347 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 346)));
    auto x_main_module_348 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 288, 1, 1}}, 347));
    auto x_main_module_349 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 348)));
    auto x_main_module_350 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 349));
    auto x_main_module_351 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 350));
    auto x_main_module_352 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 351)));
    auto x_main_module_353 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 96, 3, 3}}, 352));
    auto x_main_module_354 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 353)));
    auto x_main_module_355 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 354));
    auto x_main_module_356 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 355));
    auto x_main_module_357 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 356)));
    auto x_main_module_358 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 64, 3, 3}}, 357));
    auto x_main_module_359 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 358)));
    auto x_main_module_360 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 359));
    auto x_main_module_361 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 360));
    auto x_main_module_362 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 361)));
    auto x_main_module_363 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 288, 1, 1}}, 362));
    auto x_main_module_364 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 363)));
    auto x_main_module_365 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 364));
    auto x_main_module_366 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 365));
    auto x_main_module_367 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 366)));
    auto x_main_module_368 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 48, 5, 5}}, 367));
    auto x_main_module_369 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 368)));
    auto x_main_module_370 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 369));
    auto x_main_module_371 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 370));
    auto x_main_module_372 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 371)));
    auto x_main_module_373 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {48, 288, 1, 1}}, 372));
    auto x_main_module_374 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 373)));
    auto x_main_module_375 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 374));
    auto x_main_module_376 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 375));
    auto x_main_module_377 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 376)));
    auto x_main_module_378 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 288, 1, 1}}, 377));
    auto x_main_module_379 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 378)));
    auto x_main_module_380 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 379));
    auto x_main_module_381 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 380));
    auto x_main_module_382 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 381)));
    auto x_main_module_383 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 256, 1, 1}}, 382));
    auto x_main_module_384 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 383)));
    auto x_main_module_385 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 384));
    auto x_main_module_386 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 385));
    auto x_main_module_387 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 386)));
    auto x_main_module_388 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 96, 3, 3}}, 387));
    auto x_main_module_389 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 388)));
    auto x_main_module_390 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 389));
    auto x_main_module_391 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 390));
    auto x_main_module_392 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 391)));
    auto x_main_module_393 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 64, 3, 3}}, 392));
    auto x_main_module_394 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 393)));
    auto x_main_module_395 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 394));
    auto x_main_module_396 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 395));
    auto x_main_module_397 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 396)));
    auto x_main_module_398 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 256, 1, 1}}, 397));
    auto x_main_module_399 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 398)));
    auto x_main_module_400 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 399));
    auto x_main_module_401 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 400));
    auto x_main_module_402 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 401)));
    auto x_main_module_403 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 48, 5, 5}}, 402));
    auto x_main_module_404 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 403)));
    auto x_main_module_405 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 404));
    auto x_main_module_406 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 405));
    auto x_main_module_407 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 406)));
    auto x_main_module_408 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {48, 256, 1, 1}}, 407));
    auto x_main_module_409 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 408)));
    auto x_main_module_410 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 409));
    auto x_main_module_411 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 410));
    auto x_main_module_412 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 411)));
    auto x_main_module_413 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 256, 1, 1}}, 412));
    auto x_main_module_414 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 413)));
    auto x_main_module_415 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 414));
    auto x_main_module_416 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 415));
    auto x_main_module_417 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 416)));
    auto x_main_module_418 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {32, 192, 1, 1}}, 417));
    auto x_main_module_419 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 418)));
    auto x_main_module_420 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 419));
    auto x_main_module_421 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 420));
    auto x_main_module_422 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 421)));
    auto x_main_module_423 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 96, 3, 3}}, 422));
    auto x_main_module_424 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 423)));
    auto x_main_module_425 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 424));
    auto x_main_module_426 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 425));
    auto x_main_module_427 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 426)));
    auto x_main_module_428 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 64, 3, 3}}, 427));
    auto x_main_module_429 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 428)));
    auto x_main_module_430 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 429));
    auto x_main_module_431 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 430));
    auto x_main_module_432 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 431)));
    auto x_main_module_433 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 192, 1, 1}}, 432));
    auto x_main_module_434 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 433)));
    auto x_main_module_435 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 434));
    auto x_main_module_436 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 435));
    auto x_main_module_437 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 436)));
    auto x_main_module_438 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 48, 5, 5}}, 437));
    auto x_main_module_439 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 438)));
    auto x_main_module_440 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 439));
    auto x_main_module_441 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 440));
    auto x_main_module_442 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 441)));
    auto x_main_module_443 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {48, 192, 1, 1}}, 442));
    auto x_main_module_444 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 443)));
    auto x_main_module_445 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 444));
    auto x_main_module_446 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 445));
    auto x_main_module_447 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 446)));
    auto x_main_module_448 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 192, 1, 1}}, 447));
    auto x_main_module_449 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 448)));
    auto x_main_module_450 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 449));
    auto x_main_module_451 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 450));
    auto x_main_module_452 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 451)));
    auto x_main_module_453 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 80, 3, 3}}, 452));
    auto x_main_module_454 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {80}}, 453)));
    auto x_main_module_455 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {80}}, 454));
    auto x_main_module_456 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {80}}, 455));
    auto x_main_module_457 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {80}}, 456)));
    auto x_main_module_458 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {80, 64, 1, 1}}, 457));
    auto x_main_module_459 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 458)));
    auto x_main_module_460 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 459));
    auto x_main_module_461 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 460));
    auto x_main_module_462 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 461)));
    auto x_main_module_463 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 32, 3, 3}}, 462));
    auto x_main_module_464 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 463)));
    auto x_main_module_465 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 464));
    auto x_main_module_466 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 465));
    auto x_main_module_467 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 466)));
    auto x_main_module_468 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {32, 32, 3, 3}}, 467));
    auto x_main_module_469 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 468)));
    auto x_main_module_470 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 469));
    auto x_main_module_471 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 470));
    auto x_main_module_472 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 471)));
    auto x_main_module_473 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {32, 3, 3, 3}}, 472));
    auto x_main_module_474 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[2,"
                               "2],use_dynamic_same_auto_pad:0}"),
        x_0,
        x_main_module_473);
    auto x_main_module_475 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_474,
        x_main_module_472,
        x_main_module_471,
        x_main_module_470,
        x_main_module_469);
    auto x_main_module_476 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_475);
    auto x_main_module_477 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_476,
        x_main_module_468);
    auto x_main_module_478 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_477,
        x_main_module_467,
        x_main_module_466,
        x_main_module_465,
        x_main_module_464);
    auto x_main_module_479 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_478);
    auto x_main_module_480 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_479,
        x_main_module_463);
    auto x_main_module_481 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_480,
        x_main_module_462,
        x_main_module_461,
        x_main_module_460,
        x_main_module_459);
    auto x_main_module_482 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_481);
    auto x_main_module_483 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:1,padding:[0,0,0,0],stride:[2,2]}"),
        x_main_module_482);
    auto x_main_module_484 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_483,
        x_main_module_458);
    auto x_main_module_485 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_484,
        x_main_module_457,
        x_main_module_456,
        x_main_module_455,
        x_main_module_454);
    auto x_main_module_486 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_485);
    auto x_main_module_487 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_486,
        x_main_module_453);
    auto x_main_module_488 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_487,
        x_main_module_452,
        x_main_module_451,
        x_main_module_450,
        x_main_module_449);
    auto x_main_module_489 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_488);
    auto x_main_module_490 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:1,padding:[0,0,0,0],stride:[2,2]}"),
        x_main_module_489);
    auto x_main_module_491 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_490,
        x_main_module_448);
    auto x_main_module_492 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_491,
        x_main_module_447,
        x_main_module_446,
        x_main_module_445,
        x_main_module_444);
    auto x_main_module_493 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_492);
    auto x_main_module_494 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_490,
        x_main_module_443);
    auto x_main_module_495 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_494,
        x_main_module_442,
        x_main_module_441,
        x_main_module_440,
        x_main_module_439);
    auto x_main_module_496 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_495);
    auto x_main_module_497 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[2,2,2,2],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_496,
        x_main_module_438);
    auto x_main_module_498 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_497,
        x_main_module_437,
        x_main_module_436,
        x_main_module_435,
        x_main_module_434);
    auto x_main_module_499 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_498);
    auto x_main_module_500 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_490,
        x_main_module_433);
    auto x_main_module_501 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_500,
        x_main_module_432,
        x_main_module_431,
        x_main_module_430,
        x_main_module_429);
    auto x_main_module_502 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_501);
    auto x_main_module_503 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_502,
        x_main_module_428);
    auto x_main_module_504 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_503,
        x_main_module_427,
        x_main_module_426,
        x_main_module_425,
        x_main_module_424);
    auto x_main_module_505 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_504);
    auto x_main_module_506 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_505,
        x_main_module_423);
    auto x_main_module_507 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_506,
        x_main_module_422,
        x_main_module_421,
        x_main_module_420,
        x_main_module_419);
    auto x_main_module_508 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_507);
    auto x_main_module_509 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:0,padding:[1,1,1,1],stride:[1,1]}"),
        x_main_module_490);
    auto x_main_module_510 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_509,
        x_main_module_418);
    auto x_main_module_511 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_510,
        x_main_module_417,
        x_main_module_416,
        x_main_module_415,
        x_main_module_414);
    auto x_main_module_512 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_511);
    auto x_main_module_513 = mmain->add_instruction(migraphx::make_json_op("concat", "{axis:1}"),
                                                    x_main_module_493,
                                                    x_main_module_499,
                                                    x_main_module_508,
                                                    x_main_module_512);
    auto x_main_module_514 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_513,
        x_main_module_413);
    auto x_main_module_515 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_514,
        x_main_module_412,
        x_main_module_411,
        x_main_module_410,
        x_main_module_409);
    auto x_main_module_516 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_515);
    auto x_main_module_517 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_513,
        x_main_module_408);
    auto x_main_module_518 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_517,
        x_main_module_407,
        x_main_module_406,
        x_main_module_405,
        x_main_module_404);
    auto x_main_module_519 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_518);
    auto x_main_module_520 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[2,2,2,2],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_519,
        x_main_module_403);
    auto x_main_module_521 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_520,
        x_main_module_402,
        x_main_module_401,
        x_main_module_400,
        x_main_module_399);
    auto x_main_module_522 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_521);
    auto x_main_module_523 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_513,
        x_main_module_398);
    auto x_main_module_524 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_523,
        x_main_module_397,
        x_main_module_396,
        x_main_module_395,
        x_main_module_394);
    auto x_main_module_525 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_524);
    auto x_main_module_526 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_525,
        x_main_module_393);
    auto x_main_module_527 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_526,
        x_main_module_392,
        x_main_module_391,
        x_main_module_390,
        x_main_module_389);
    auto x_main_module_528 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_527);
    auto x_main_module_529 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_528,
        x_main_module_388);
    auto x_main_module_530 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_529,
        x_main_module_387,
        x_main_module_386,
        x_main_module_385,
        x_main_module_384);
    auto x_main_module_531 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_530);
    auto x_main_module_532 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:0,padding:[1,1,1,1],stride:[1,1]}"),
        x_main_module_513);
    auto x_main_module_533 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_532,
        x_main_module_383);
    auto x_main_module_534 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_533,
        x_main_module_382,
        x_main_module_381,
        x_main_module_380,
        x_main_module_379);
    auto x_main_module_535 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_534);
    auto x_main_module_536 = mmain->add_instruction(migraphx::make_json_op("concat", "{axis:1}"),
                                                    x_main_module_516,
                                                    x_main_module_522,
                                                    x_main_module_531,
                                                    x_main_module_535);
    auto x_main_module_537 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_536,
        x_main_module_378);
    auto x_main_module_538 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_537,
        x_main_module_377,
        x_main_module_376,
        x_main_module_375,
        x_main_module_374);
    auto x_main_module_539 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_538);
    auto x_main_module_540 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_536,
        x_main_module_373);
    auto x_main_module_541 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_540,
        x_main_module_372,
        x_main_module_371,
        x_main_module_370,
        x_main_module_369);
    auto x_main_module_542 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_541);
    auto x_main_module_543 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[2,2,2,2],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_542,
        x_main_module_368);
    auto x_main_module_544 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_543,
        x_main_module_367,
        x_main_module_366,
        x_main_module_365,
        x_main_module_364);
    auto x_main_module_545 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_544);
    auto x_main_module_546 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_536,
        x_main_module_363);
    auto x_main_module_547 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_546,
        x_main_module_362,
        x_main_module_361,
        x_main_module_360,
        x_main_module_359);
    auto x_main_module_548 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_547);
    auto x_main_module_549 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_548,
        x_main_module_358);
    auto x_main_module_550 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_549,
        x_main_module_357,
        x_main_module_356,
        x_main_module_355,
        x_main_module_354);
    auto x_main_module_551 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_550);
    auto x_main_module_552 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_551,
        x_main_module_353);
    auto x_main_module_553 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_552,
        x_main_module_352,
        x_main_module_351,
        x_main_module_350,
        x_main_module_349);
    auto x_main_module_554 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_553);
    auto x_main_module_555 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:0,padding:[1,1,1,1],stride:[1,1]}"),
        x_main_module_536);
    auto x_main_module_556 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_555,
        x_main_module_348);
    auto x_main_module_557 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_556,
        x_main_module_347,
        x_main_module_346,
        x_main_module_345,
        x_main_module_344);
    auto x_main_module_558 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_557);
    auto x_main_module_559 = mmain->add_instruction(migraphx::make_json_op("concat", "{axis:1}"),
                                                    x_main_module_539,
                                                    x_main_module_545,
                                                    x_main_module_554,
                                                    x_main_module_558);
    auto x_main_module_560 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[2,"
                               "2],use_dynamic_same_auto_pad:0}"),
        x_main_module_559,
        x_main_module_343);
    auto x_main_module_561 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_560,
        x_main_module_342,
        x_main_module_341,
        x_main_module_340,
        x_main_module_339);
    auto x_main_module_562 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_561);
    auto x_main_module_563 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_559,
        x_main_module_338);
    auto x_main_module_564 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_563,
        x_main_module_337,
        x_main_module_336,
        x_main_module_335,
        x_main_module_334);
    auto x_main_module_565 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_564);
    auto x_main_module_566 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_565,
        x_main_module_333);
    auto x_main_module_567 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_566,
        x_main_module_332,
        x_main_module_331,
        x_main_module_330,
        x_main_module_329);
    auto x_main_module_568 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_567);
    auto x_main_module_569 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[2,"
                               "2],use_dynamic_same_auto_pad:0}"),
        x_main_module_568,
        x_main_module_328);
    auto x_main_module_570 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_569,
        x_main_module_327,
        x_main_module_326,
        x_main_module_325,
        x_main_module_324);
    auto x_main_module_571 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_570);
    auto x_main_module_572 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:1,padding:[0,0,0,0],stride:[2,2]}"),
        x_main_module_559);
    auto x_main_module_573 = mmain->add_instruction(migraphx::make_json_op("concat", "{axis:1}"),
                                                    x_main_module_562,
                                                    x_main_module_571,
                                                    x_main_module_572);
    auto x_main_module_574 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_573,
        x_main_module_323);
    auto x_main_module_575 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_574,
        x_main_module_322,
        x_main_module_321,
        x_main_module_320,
        x_main_module_319);
    auto x_main_module_576 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_575);
    auto x_main_module_577 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_573,
        x_main_module_318);
    auto x_main_module_578 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_577,
        x_main_module_317,
        x_main_module_316,
        x_main_module_315,
        x_main_module_314);
    auto x_main_module_579 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_578);
    auto x_main_module_580 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_579,
        x_main_module_313);
    auto x_main_module_581 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_580,
        x_main_module_312,
        x_main_module_311,
        x_main_module_310,
        x_main_module_309);
    auto x_main_module_582 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_581);
    auto x_main_module_583 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_582,
        x_main_module_308);
    auto x_main_module_584 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_583,
        x_main_module_307,
        x_main_module_306,
        x_main_module_305,
        x_main_module_304);
    auto x_main_module_585 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_584);
    auto x_main_module_586 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_573,
        x_main_module_303);
    auto x_main_module_587 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_586,
        x_main_module_302,
        x_main_module_301,
        x_main_module_300,
        x_main_module_299);
    auto x_main_module_588 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_587);
    auto x_main_module_589 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_588,
        x_main_module_298);
    auto x_main_module_590 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_589,
        x_main_module_297,
        x_main_module_296,
        x_main_module_295,
        x_main_module_294);
    auto x_main_module_591 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_590);
    auto x_main_module_592 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_591,
        x_main_module_293);
    auto x_main_module_593 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_592,
        x_main_module_292,
        x_main_module_291,
        x_main_module_290,
        x_main_module_289);
    auto x_main_module_594 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_593);
    auto x_main_module_595 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_594,
        x_main_module_288);
    auto x_main_module_596 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_595,
        x_main_module_287,
        x_main_module_286,
        x_main_module_285,
        x_main_module_284);
    auto x_main_module_597 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_596);
    auto x_main_module_598 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_597,
        x_main_module_283);
    auto x_main_module_599 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_598,
        x_main_module_282,
        x_main_module_281,
        x_main_module_280,
        x_main_module_279);
    auto x_main_module_600 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_599);
    auto x_main_module_601 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:0,padding:[1,1,1,1],stride:[1,1]}"),
        x_main_module_573);
    auto x_main_module_602 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_601,
        x_main_module_278);
    auto x_main_module_603 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_602,
        x_main_module_277,
        x_main_module_276,
        x_main_module_275,
        x_main_module_274);
    auto x_main_module_604 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_603);
    auto x_main_module_605 = mmain->add_instruction(migraphx::make_json_op("concat", "{axis:1}"),
                                                    x_main_module_576,
                                                    x_main_module_585,
                                                    x_main_module_600,
                                                    x_main_module_604);
    auto x_main_module_606 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_605,
        x_main_module_273);
    auto x_main_module_607 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_606,
        x_main_module_272,
        x_main_module_271,
        x_main_module_270,
        x_main_module_269);
    auto x_main_module_608 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_607);
    auto x_main_module_609 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_605,
        x_main_module_268);
    auto x_main_module_610 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_609,
        x_main_module_267,
        x_main_module_266,
        x_main_module_265,
        x_main_module_264);
    auto x_main_module_611 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_610);
    auto x_main_module_612 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_611,
        x_main_module_263);
    auto x_main_module_613 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_612,
        x_main_module_262,
        x_main_module_261,
        x_main_module_260,
        x_main_module_259);
    auto x_main_module_614 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_613);
    auto x_main_module_615 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_614,
        x_main_module_258);
    auto x_main_module_616 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_615,
        x_main_module_257,
        x_main_module_256,
        x_main_module_255,
        x_main_module_254);
    auto x_main_module_617 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_616);
    auto x_main_module_618 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_605,
        x_main_module_253);
    auto x_main_module_619 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_618,
        x_main_module_252,
        x_main_module_251,
        x_main_module_250,
        x_main_module_249);
    auto x_main_module_620 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_619);
    auto x_main_module_621 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_620,
        x_main_module_248);
    auto x_main_module_622 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_621,
        x_main_module_247,
        x_main_module_246,
        x_main_module_245,
        x_main_module_244);
    auto x_main_module_623 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_622);
    auto x_main_module_624 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_623,
        x_main_module_243);
    auto x_main_module_625 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_624,
        x_main_module_242,
        x_main_module_241,
        x_main_module_240,
        x_main_module_239);
    auto x_main_module_626 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_625);
    auto x_main_module_627 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_626,
        x_main_module_238);
    auto x_main_module_628 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_627,
        x_main_module_237,
        x_main_module_236,
        x_main_module_235,
        x_main_module_234);
    auto x_main_module_629 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_628);
    auto x_main_module_630 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_629,
        x_main_module_233);
    auto x_main_module_631 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_630,
        x_main_module_232,
        x_main_module_231,
        x_main_module_230,
        x_main_module_229);
    auto x_main_module_632 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_631);
    auto x_main_module_633 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:0,padding:[1,1,1,1],stride:[1,1]}"),
        x_main_module_605);
    auto x_main_module_634 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_633,
        x_main_module_228);
    auto x_main_module_635 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_634,
        x_main_module_227,
        x_main_module_226,
        x_main_module_225,
        x_main_module_224);
    auto x_main_module_636 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_635);
    auto x_main_module_637 = mmain->add_instruction(migraphx::make_json_op("concat", "{axis:1}"),
                                                    x_main_module_608,
                                                    x_main_module_617,
                                                    x_main_module_632,
                                                    x_main_module_636);
    auto x_main_module_638 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_637,
        x_main_module_223);
    auto x_main_module_639 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_638,
        x_main_module_222,
        x_main_module_221,
        x_main_module_220,
        x_main_module_219);
    auto x_main_module_640 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_639);
    auto x_main_module_641 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_637,
        x_main_module_218);
    auto x_main_module_642 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_641,
        x_main_module_217,
        x_main_module_216,
        x_main_module_215,
        x_main_module_214);
    auto x_main_module_643 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_642);
    auto x_main_module_644 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_643,
        x_main_module_213);
    auto x_main_module_645 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_644,
        x_main_module_212,
        x_main_module_211,
        x_main_module_210,
        x_main_module_209);
    auto x_main_module_646 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_645);
    auto x_main_module_647 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_646,
        x_main_module_208);
    auto x_main_module_648 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_647,
        x_main_module_207,
        x_main_module_206,
        x_main_module_205,
        x_main_module_204);
    auto x_main_module_649 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_648);
    auto x_main_module_650 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_637,
        x_main_module_203);
    auto x_main_module_651 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_650,
        x_main_module_202,
        x_main_module_201,
        x_main_module_200,
        x_main_module_199);
    auto x_main_module_652 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_651);
    auto x_main_module_653 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_652,
        x_main_module_198);
    auto x_main_module_654 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_653,
        x_main_module_197,
        x_main_module_196,
        x_main_module_195,
        x_main_module_194);
    auto x_main_module_655 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_654);
    auto x_main_module_656 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_655,
        x_main_module_193);
    auto x_main_module_657 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_656,
        x_main_module_192,
        x_main_module_191,
        x_main_module_190,
        x_main_module_189);
    auto x_main_module_658 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_657);
    auto x_main_module_659 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_658,
        x_main_module_188);
    auto x_main_module_660 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_659,
        x_main_module_187,
        x_main_module_186,
        x_main_module_185,
        x_main_module_184);
    auto x_main_module_661 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_660);
    auto x_main_module_662 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_661,
        x_main_module_183);
    auto x_main_module_663 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_662,
        x_main_module_182,
        x_main_module_181,
        x_main_module_180,
        x_main_module_179);
    auto x_main_module_664 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_663);
    auto x_main_module_665 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:0,padding:[1,1,1,1],stride:[1,1]}"),
        x_main_module_637);
    auto x_main_module_666 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_665,
        x_main_module_178);
    auto x_main_module_667 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_666,
        x_main_module_177,
        x_main_module_176,
        x_main_module_175,
        x_main_module_174);
    auto x_main_module_668 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_667);
    auto x_main_module_669 = mmain->add_instruction(migraphx::make_json_op("concat", "{axis:1}"),
                                                    x_main_module_640,
                                                    x_main_module_649,
                                                    x_main_module_664,
                                                    x_main_module_668);
    auto x_main_module_670 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_669,
        x_main_module_173);
    auto x_main_module_671 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_670,
        x_main_module_172,
        x_main_module_171,
        x_main_module_170,
        x_main_module_169);
    auto x_main_module_672 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_671);
    auto x_main_module_673 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_669,
        x_main_module_168);
    auto x_main_module_674 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_673,
        x_main_module_167,
        x_main_module_166,
        x_main_module_165,
        x_main_module_164);
    auto x_main_module_675 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_674);
    auto x_main_module_676 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_675,
        x_main_module_163);
    auto x_main_module_677 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_676,
        x_main_module_162,
        x_main_module_161,
        x_main_module_160,
        x_main_module_159);
    auto x_main_module_678 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_677);
    auto x_main_module_679 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_678,
        x_main_module_158);
    auto x_main_module_680 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_679,
        x_main_module_157,
        x_main_module_156,
        x_main_module_155,
        x_main_module_154);
    auto x_main_module_681 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_680);
    auto x_main_module_682 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_669,
        x_main_module_153);
    auto x_main_module_683 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_682,
        x_main_module_152,
        x_main_module_151,
        x_main_module_150,
        x_main_module_149);
    auto x_main_module_684 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_683);
    auto x_main_module_685 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_684,
        x_main_module_148);
    auto x_main_module_686 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_685,
        x_main_module_147,
        x_main_module_146,
        x_main_module_145,
        x_main_module_144);
    auto x_main_module_687 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_686);
    auto x_main_module_688 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_687,
        x_main_module_143);
    auto x_main_module_689 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_688,
        x_main_module_142,
        x_main_module_141,
        x_main_module_140,
        x_main_module_139);
    auto x_main_module_690 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_689);
    auto x_main_module_691 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_690,
        x_main_module_138);
    auto x_main_module_692 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_691,
        x_main_module_137,
        x_main_module_136,
        x_main_module_135,
        x_main_module_134);
    auto x_main_module_693 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_692);
    auto x_main_module_694 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_693,
        x_main_module_133);
    auto x_main_module_695 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_694,
        x_main_module_132,
        x_main_module_131,
        x_main_module_130,
        x_main_module_129);
    auto x_main_module_696 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_695);
    auto x_main_module_697 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:0,padding:[1,1,1,1],stride:[1,1]}"),
        x_main_module_669);
    auto x_main_module_698 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_697,
        x_main_module_128);
    auto x_main_module_699 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_698,
        x_main_module_127,
        x_main_module_126,
        x_main_module_125,
        x_main_module_124);
    auto x_main_module_700 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_699);
    auto x_main_module_701 = mmain->add_instruction(migraphx::make_json_op("concat", "{axis:1}"),
                                                    x_main_module_672,
                                                    x_main_module_681,
                                                    x_main_module_696,
                                                    x_main_module_700);
    auto x_main_module_702 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_701,
        x_main_module_123);
    auto x_main_module_703 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_702,
        x_main_module_122,
        x_main_module_121,
        x_main_module_120,
        x_main_module_119);
    auto x_main_module_704 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_703);
    auto x_main_module_705 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[2,"
                               "2],use_dynamic_same_auto_pad:0}"),
        x_main_module_704,
        x_main_module_118);
    auto x_main_module_706 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_705,
        x_main_module_117,
        x_main_module_116,
        x_main_module_115,
        x_main_module_114);
    auto x_main_module_707 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_706);
    auto x_main_module_708 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_701,
        x_main_module_113);
    auto x_main_module_709 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_708,
        x_main_module_112,
        x_main_module_111,
        x_main_module_110,
        x_main_module_109);
    auto x_main_module_710 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_709);
    auto x_main_module_711 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_710,
        x_main_module_108);
    auto x_main_module_712 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_711,
        x_main_module_107,
        x_main_module_106,
        x_main_module_105,
        x_main_module_104);
    auto x_main_module_713 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_712);
    auto x_main_module_714 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_713,
        x_main_module_103);
    auto x_main_module_715 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_714,
        x_main_module_102,
        x_main_module_101,
        x_main_module_100,
        x_main_module_99);
    auto x_main_module_716 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_715);
    auto x_main_module_717 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[2,"
                               "2],use_dynamic_same_auto_pad:0}"),
        x_main_module_716,
        x_main_module_98);
    auto x_main_module_718 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_717,
        x_main_module_97,
        x_main_module_96,
        x_main_module_95,
        x_main_module_94);
    auto x_main_module_719 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_718);
    auto x_main_module_720 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:1,padding:[0,0,0,0],stride:[2,2]}"),
        x_main_module_701);
    auto x_main_module_721 = mmain->add_instruction(migraphx::make_json_op("concat", "{axis:1}"),
                                                    x_main_module_707,
                                                    x_main_module_719,
                                                    x_main_module_720);
    auto x_main_module_722 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_721,
        x_main_module_93);
    auto x_main_module_723 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_722,
        x_main_module_92,
        x_main_module_91,
        x_main_module_90,
        x_main_module_89);
    auto x_main_module_724 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_723);
    auto x_main_module_725 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_721,
        x_main_module_88);
    auto x_main_module_726 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_725,
        x_main_module_87,
        x_main_module_86,
        x_main_module_85,
        x_main_module_84);
    auto x_main_module_727 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_726);
    auto x_main_module_728 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,1,0,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_727,
        x_main_module_83);
    auto x_main_module_729 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_728,
        x_main_module_82,
        x_main_module_81,
        x_main_module_80,
        x_main_module_79);
    auto x_main_module_730 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_729);
    auto x_main_module_731 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,0,1,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_727,
        x_main_module_78);
    auto x_main_module_732 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_731,
        x_main_module_77,
        x_main_module_76,
        x_main_module_75,
        x_main_module_74);
    auto x_main_module_733 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_732);
    auto x_main_module_734 = mmain->add_instruction(
        migraphx::make_json_op("concat", "{axis:1}"), x_main_module_730, x_main_module_733);
    auto x_main_module_735 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_721,
        x_main_module_73);
    auto x_main_module_736 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_735,
        x_main_module_72,
        x_main_module_71,
        x_main_module_70,
        x_main_module_69);
    auto x_main_module_737 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_736);
    auto x_main_module_738 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_737,
        x_main_module_68);
    auto x_main_module_739 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_738,
        x_main_module_67,
        x_main_module_66,
        x_main_module_65,
        x_main_module_64);
    auto x_main_module_740 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_739);
    auto x_main_module_741 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,1,0,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_740,
        x_main_module_63);
    auto x_main_module_742 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_741,
        x_main_module_62,
        x_main_module_61,
        x_main_module_60,
        x_main_module_59);
    auto x_main_module_743 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_742);
    auto x_main_module_744 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,0,1,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_740,
        x_main_module_58);
    auto x_main_module_745 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_744,
        x_main_module_57,
        x_main_module_56,
        x_main_module_55,
        x_main_module_54);
    auto x_main_module_746 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_745);
    auto x_main_module_747 = mmain->add_instruction(
        migraphx::make_json_op("concat", "{axis:1}"), x_main_module_743, x_main_module_746);
    auto x_main_module_748 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:0,padding:[1,1,1,1],stride:[1,1]}"),
        x_main_module_721);
    auto x_main_module_749 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_748,
        x_main_module_53);
    auto x_main_module_750 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_749,
        x_main_module_52,
        x_main_module_51,
        x_main_module_50,
        x_main_module_49);
    auto x_main_module_751 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_750);
    auto x_main_module_752 = mmain->add_instruction(migraphx::make_json_op("concat", "{axis:1}"),
                                                    x_main_module_724,
                                                    x_main_module_734,
                                                    x_main_module_747,
                                                    x_main_module_751);
    auto x_main_module_753 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_752,
        x_main_module_48);
    auto x_main_module_754 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_753,
        x_main_module_47,
        x_main_module_46,
        x_main_module_45,
        x_main_module_44);
    auto x_main_module_755 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_754);
    auto x_main_module_756 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_752,
        x_main_module_43);
    auto x_main_module_757 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_756,
        x_main_module_42,
        x_main_module_41,
        x_main_module_40,
        x_main_module_39);
    auto x_main_module_758 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_757);
    auto x_main_module_759 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,1,0,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_758,
        x_main_module_38);
    auto x_main_module_760 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_759,
        x_main_module_37,
        x_main_module_36,
        x_main_module_35,
        x_main_module_34);
    auto x_main_module_761 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_760);
    auto x_main_module_762 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,0,1,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_758,
        x_main_module_33);
    auto x_main_module_763 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_762,
        x_main_module_32,
        x_main_module_31,
        x_main_module_30,
        x_main_module_29);
    auto x_main_module_764 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_763);
    auto x_main_module_765 = mmain->add_instruction(
        migraphx::make_json_op("concat", "{axis:1}"), x_main_module_761, x_main_module_764);
    auto x_main_module_766 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_752,
        x_main_module_28);
    auto x_main_module_767 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_766,
        x_main_module_27,
        x_main_module_26,
        x_main_module_25,
        x_main_module_24);
    auto x_main_module_768 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_767);
    auto x_main_module_769 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_768,
        x_main_module_23);
    auto x_main_module_770 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_769,
        x_main_module_22,
        x_main_module_21,
        x_main_module_20,
        x_main_module_19);
    auto x_main_module_771 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_770);
    auto x_main_module_772 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,1,0,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_771,
        x_main_module_18);
    auto x_main_module_773 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_772,
        x_main_module_17,
        x_main_module_16,
        x_main_module_15,
        x_main_module_14);
    auto x_main_module_774 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_773);
    auto x_main_module_775 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,0,1,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_771,
        x_main_module_13);
    auto x_main_module_776 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_775,
        x_main_module_12,
        x_main_module_11,
        x_main_module_10,
        x_main_module_9);
    auto x_main_module_777 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_776);
    auto x_main_module_778 = mmain->add_instruction(
        migraphx::make_json_op("concat", "{axis:1}"), x_main_module_774, x_main_module_777);
    auto x_main_module_779 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:0,padding:[1,1,1,1],stride:[1,1]}"),
        x_main_module_752);
    auto x_main_module_780 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_779,
        x_main_module_8);
    auto x_main_module_781 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:0.0010000000474974513,momentum:0.8999999761581421}"),
        x_main_module_780,
        x_main_module_7,
        x_main_module_6,
        x_main_module_5,
        x_main_module_4);
    auto x_main_module_782 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_781);
    auto x_main_module_783 = mmain->add_instruction(migraphx::make_json_op("concat", "{axis:1}"),
                                                    x_main_module_755,
                                                    x_main_module_765,
                                                    x_main_module_778,
                                                    x_main_module_782);
    auto x_main_module_784 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[8,8],lp_order:2,mode:0,padding:[0,0,0,0],stride:[8,8]}"),
        x_main_module_783);
    auto x_main_module_785 =
        mmain->add_instruction(migraphx::make_op("identity"), x_main_module_784);
    auto x_main_module_786 =
        mmain->add_instruction(migraphx::make_json_op("flatten", "{axis:1}"), x_main_module_785);
    auto x_main_module_787 = mmain->add_instruction(
        migraphx::make_json_op("transpose", "{permutation:[1,0]}"), x_main_module_3);
    auto x_main_module_788 =
        mmain->add_instruction(migraphx::make_op("dot"), x_main_module_786, x_main_module_787);
    auto x_main_module_789 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1000]}"), x_main_module_2);
    auto x_main_module_790 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1000]}"), x_main_module_0);
    auto x_main_module_791 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_789, x_main_module_790);
    auto x_main_module_792 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_788, x_main_module_791);
    mmain->add_return({x_main_module_792});

    return p;
}
} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx
