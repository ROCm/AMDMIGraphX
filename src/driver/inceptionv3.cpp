
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

migraphx::program inceptionv3(unsigned batch) // NOLINT(readability-function-size)
{
    migraphx::program p;
    migraphx::module_ref mmain = p.get_main_module();
    auto x_main_module_0       = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 0)));
    auto x_input_1             = mmain->add_parameter(
        "input.1", migraphx::shape{migraphx::shape::float_type, {batch, 3, 299, 299}});
    auto x_main_module_2 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1000, 2048}}, 1));
    auto x_main_module_3 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1000}}, 2));
    auto x_main_module_4 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 96, 3, 3}}, 3));
    auto x_main_module_5 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 4));
    auto x_main_module_6 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 64, 3, 3}}, 5));
    auto x_main_module_7 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 6));
    auto x_main_module_8 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 288, 1, 1}}, 7));
    auto x_main_module_9 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 8));
    auto x_main_module_10 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 288, 3, 3}}, 9));
    auto x_main_module_11 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 10));
    auto x_main_module_12 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 288, 1, 1}}, 11));
    auto x_main_module_13 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 12));
    auto x_main_module_14 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 96, 3, 3}}, 13));
    auto x_main_module_15 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 14));
    auto x_main_module_16 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 64, 3, 3}}, 15));
    auto x_main_module_17 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 16));
    auto x_main_module_18 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 288, 1, 1}}, 17));
    auto x_main_module_19 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 18));
    auto x_main_module_20 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 48, 5, 5}}, 19));
    auto x_main_module_21 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 20));
    auto x_main_module_22 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {48, 288, 1, 1}}, 21));
    auto x_main_module_23 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 22));
    auto x_main_module_24 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 288, 1, 1}}, 23));
    auto x_main_module_25 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 24));
    auto x_main_module_26 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 256, 1, 1}}, 25));
    auto x_main_module_27 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 26));
    auto x_main_module_28 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 96, 3, 3}}, 27));
    auto x_main_module_29 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 28));
    auto x_main_module_30 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 64, 3, 3}}, 29));
    auto x_main_module_31 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 30));
    auto x_main_module_32 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 256, 1, 1}}, 31));
    auto x_main_module_33 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 32));
    auto x_main_module_34 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 48, 5, 5}}, 33));
    auto x_main_module_35 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 34));
    auto x_main_module_36 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {48, 256, 1, 1}}, 35));
    auto x_main_module_37 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 36));
    auto x_main_module_38 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 256, 1, 1}}, 37));
    auto x_main_module_39 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 38));
    auto x_main_module_40 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {32, 192, 1, 1}}, 39));
    auto x_main_module_41 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 40));
    auto x_main_module_42 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 96, 3, 3}}, 41));
    auto x_main_module_43 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 42));
    auto x_main_module_44 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 64, 3, 3}}, 43));
    auto x_main_module_45 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 44));
    auto x_main_module_46 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 192, 1, 1}}, 45));
    auto x_main_module_47 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 46));
    auto x_main_module_48 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 48, 5, 5}}, 47));
    auto x_main_module_49 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 48));
    auto x_main_module_50 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {48, 192, 1, 1}}, 49));
    auto x_main_module_51 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 50));
    auto x_main_module_52 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 192, 1, 1}}, 51));
    auto x_main_module_53 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 52));
    auto x_main_module_54 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 80, 3, 3}}, 53));
    auto x_main_module_55 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {80}}, 54));
    auto x_main_module_56 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {80, 64, 1, 1}}, 55));
    auto x_main_module_57 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 56));
    auto x_main_module_58 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 32, 3, 3}}, 57));
    auto x_main_module_59 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 58));
    auto x_main_module_60 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {32, 32, 3, 3}}, 59));
    auto x_main_module_61 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 60));
    auto x_main_module_62 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {32, 3, 3, 3}}, 61));
    auto x_main_module_63 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 62));
    auto x_main_module_64 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 2048, 1, 1}}, 63));
    auto x_main_module_65 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 64));
    auto x_main_module_66 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 3, 1}}, 65));
    auto x_main_module_67 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 66));
    auto x_main_module_68 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 1, 3}}, 67));
    auto x_main_module_69 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 68));
    auto x_main_module_70 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 448, 3, 3}}, 69));
    auto x_main_module_71 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 70));
    auto x_main_module_72 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {448, 2048, 1, 1}}, 71));
    auto x_main_module_73 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 72));
    auto x_main_module_74 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 3, 1}}, 73));
    auto x_main_module_75 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 74));
    auto x_main_module_76 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 1, 3}}, 75));
    auto x_main_module_77 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 76));
    auto x_main_module_78 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 2048, 1, 1}}, 77));
    auto x_main_module_79 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 78));
    auto x_main_module_80 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {320, 2048, 1, 1}}, 79));
    auto x_main_module_81 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 80));
    auto x_main_module_82 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 1280, 1, 1}}, 81));
    auto x_main_module_83 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 82));
    auto x_main_module_84 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 3, 1}}, 83));
    auto x_main_module_85 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 84));
    auto x_main_module_86 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 1, 3}}, 85));
    auto x_main_module_87 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 86));
    auto x_main_module_88 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 448, 3, 3}}, 87));
    auto x_main_module_89 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 88));
    auto x_main_module_90 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {448, 1280, 1, 1}}, 89));
    auto x_main_module_91 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 90));
    auto x_main_module_92 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 3, 1}}, 91));
    auto x_main_module_93 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 92));
    auto x_main_module_94 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 1, 3}}, 93));
    auto x_main_module_95 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 94));
    auto x_main_module_96 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 1280, 1, 1}}, 95));
    auto x_main_module_97 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 96));
    auto x_main_module_98 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {320, 1280, 1, 1}}, 97));
    auto x_main_module_99 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 98));
    auto x_main_module_100 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 3, 3}}, 99));
    auto x_main_module_101 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 100));
    auto x_main_module_102 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 7, 1}}, 101));
    auto x_main_module_103 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 102));
    auto x_main_module_104 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 1, 7}}, 103));
    auto x_main_module_105 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 104));
    auto x_main_module_106 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 105));
    auto x_main_module_107 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 106));
    auto x_main_module_108 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {320, 192, 3, 3}}, 107));
    auto x_main_module_109 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 108));
    auto x_main_module_110 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 109));
    auto x_main_module_111 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 110));
    auto x_main_module_112 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 111));
    auto x_main_module_113 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 112));
    auto x_main_module_114 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 1, 7}}, 113));
    auto x_main_module_115 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 114));
    auto x_main_module_116 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 7, 1}}, 115));
    auto x_main_module_117 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 116));
    auto x_main_module_118 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 1, 7}}, 117));
    auto x_main_module_119 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 118));
    auto x_main_module_120 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 7, 1}}, 119));
    auto x_main_module_121 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 120));
    auto x_main_module_122 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 121));
    auto x_main_module_123 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 122));
    auto x_main_module_124 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 7, 1}}, 123));
    auto x_main_module_125 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 124));
    auto x_main_module_126 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 1, 7}}, 125));
    auto x_main_module_127 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 126));
    auto x_main_module_128 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 127));
    auto x_main_module_129 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 128));
    auto x_main_module_130 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 129));
    auto x_main_module_131 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 130));
    auto x_main_module_132 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 131));
    auto x_main_module_133 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 132));
    auto x_main_module_134 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 160, 1, 7}}, 133));
    auto x_main_module_135 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 134));
    auto x_main_module_136 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 7, 1}}, 135));
    auto x_main_module_137 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 136));
    auto x_main_module_138 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 1, 7}}, 137));
    auto x_main_module_139 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 138));
    auto x_main_module_140 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 7, 1}}, 139));
    auto x_main_module_141 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 140));
    auto x_main_module_142 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 768, 1, 1}}, 141));
    auto x_main_module_143 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 142));
    auto x_main_module_144 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 160, 7, 1}}, 143));
    auto x_main_module_145 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 144));
    auto x_main_module_146 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 1, 7}}, 145));
    auto x_main_module_147 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 146));
    auto x_main_module_148 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 768, 1, 1}}, 147));
    auto x_main_module_149 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 148));
    auto x_main_module_150 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 149));
    auto x_main_module_151 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 150));
    auto x_main_module_152 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 151));
    auto x_main_module_153 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 152));
    auto x_main_module_154 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 160, 1, 7}}, 153));
    auto x_main_module_155 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 154));
    auto x_main_module_156 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 7, 1}}, 155));
    auto x_main_module_157 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 156));
    auto x_main_module_158 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 1, 7}}, 157));
    auto x_main_module_159 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 158));
    auto x_main_module_160 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 7, 1}}, 159));
    auto x_main_module_161 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 160));
    auto x_main_module_162 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 768, 1, 1}}, 161));
    auto x_main_module_163 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 162));
    auto x_main_module_164 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 160, 7, 1}}, 163));
    auto x_main_module_165 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 164));
    auto x_main_module_166 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 1, 7}}, 165));
    auto x_main_module_167 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 166));
    auto x_main_module_168 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 768, 1, 1}}, 167));
    auto x_main_module_169 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 168));
    auto x_main_module_170 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 169));
    auto x_main_module_171 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 170));
    auto x_main_module_172 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 171));
    auto x_main_module_173 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 172));
    auto x_main_module_174 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 128, 1, 7}}, 173));
    auto x_main_module_175 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 174));
    auto x_main_module_176 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 7, 1}}, 175));
    auto x_main_module_177 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 176));
    auto x_main_module_178 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 1, 7}}, 177));
    auto x_main_module_179 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 178));
    auto x_main_module_180 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 7, 1}}, 179));
    auto x_main_module_181 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 180));
    auto x_main_module_182 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 768, 1, 1}}, 181));
    auto x_main_module_183 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 182));
    auto x_main_module_184 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 128, 7, 1}}, 183));
    auto x_main_module_185 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 184));
    auto x_main_module_186 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 1, 7}}, 185));
    auto x_main_module_187 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 186));
    auto x_main_module_188 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 768, 1, 1}}, 187));
    auto x_main_module_189 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 188));
    auto x_main_module_190 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 189));
    auto x_main_module_191 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 190));
    auto x_main_module_192 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[2,2]}")),
        x_input_1,
        x_main_module_62);
    auto x_main_module_193 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,32,149,149]}")),
        x_main_module_61);
    auto x_main_module_194 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_192, x_main_module_193);
    auto x_main_module_195 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_194);
    auto x_main_module_196 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_195,
        x_main_module_60);
    auto x_main_module_197 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,32,147,147]}")),
        x_main_module_59);
    auto x_main_module_198 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_196, x_main_module_197);
    auto x_main_module_199 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_198);
    auto x_main_module_200 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}")),
        x_main_module_199,
        x_main_module_58);
    auto x_main_module_201 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,64,147,147]}")),
        x_main_module_57);
    auto x_main_module_202 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_200, x_main_module_201);
    auto x_main_module_203 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_202);
    auto x_main_module_204 = mmain->add_instruction(
        migraphx::make_op(
            "pooling",
            migraphx::from_json_string(
                "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:1,padding:[0,0,0,0],stride:[2,2]}")),
        x_main_module_203);
    auto x_main_module_205 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_204,
        x_main_module_56);
    auto x_main_module_206 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,80,73,73]}")),
        x_main_module_55);
    auto x_main_module_207 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_205, x_main_module_206);
    auto x_main_module_208 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_207);
    auto x_main_module_209 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_208,
        x_main_module_54);
    auto x_main_module_210 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,192,71,71]}")),
        x_main_module_53);
    auto x_main_module_211 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_209, x_main_module_210);
    auto x_main_module_212 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_211);
    auto x_main_module_213 = mmain->add_instruction(
        migraphx::make_op(
            "pooling",
            migraphx::from_json_string(
                "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:1,padding:[0,0,0,0],stride:[2,2]}")),
        x_main_module_212);
    auto x_main_module_214 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_213,
        x_main_module_52);
    auto x_main_module_215 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,64,35,35]}")),
        x_main_module_51);
    auto x_main_module_216 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_214, x_main_module_215);
    auto x_main_module_217 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_216);
    auto x_main_module_218 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_213,
        x_main_module_50);
    auto x_main_module_219 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,48,35,35]}")),
        x_main_module_49);
    auto x_main_module_220 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_218, x_main_module_219);
    auto x_main_module_221 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_220);
    auto x_main_module_222 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[2,2,2,2],padding_mode:0,stride:[1,1]}")),
        x_main_module_221,
        x_main_module_48);
    auto x_main_module_223 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,64,35,35]}")),
        x_main_module_47);
    auto x_main_module_224 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_222, x_main_module_223);
    auto x_main_module_225 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_224);
    auto x_main_module_226 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_213,
        x_main_module_46);
    auto x_main_module_227 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,64,35,35]}")),
        x_main_module_45);
    auto x_main_module_228 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_226, x_main_module_227);
    auto x_main_module_229 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_228);
    auto x_main_module_230 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}")),
        x_main_module_229,
        x_main_module_44);
    auto x_main_module_231 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,96,35,35]}")),
        x_main_module_43);
    auto x_main_module_232 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_230, x_main_module_231);
    auto x_main_module_233 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_232);
    auto x_main_module_234 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}")),
        x_main_module_233,
        x_main_module_42);
    auto x_main_module_235 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,96,35,35]}")),
        x_main_module_41);
    auto x_main_module_236 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_234, x_main_module_235);
    auto x_main_module_237 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_236);
    auto x_main_module_238 = mmain->add_instruction(
        migraphx::make_op("pad",
                          migraphx::from_json_string("{mode:0,pads:[0,0,1,1,0,0,1,1],value:0.0}")),
        x_main_module_213);
    auto x_main_module_239 = mmain->add_instruction(
        migraphx::make_op(
            "pooling",
            migraphx::from_json_string(
                "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:0,padding:[0,0,0,0],stride:[1,1]}")),
        x_main_module_238);
    auto x_main_module_240 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_239,
        x_main_module_40);
    auto x_main_module_241 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,32,35,35]}")),
        x_main_module_39);
    auto x_main_module_242 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_240, x_main_module_241);
    auto x_main_module_243 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_242);
    auto x_main_module_244 =
        mmain->add_instruction(migraphx::make_op("concat", migraphx::from_json_string("{axis:1}")),
                               x_main_module_217,
                               x_main_module_225,
                               x_main_module_237,
                               x_main_module_243);
    auto x_main_module_245 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_244,
        x_main_module_38);
    auto x_main_module_246 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,64,35,35]}")),
        x_main_module_37);
    auto x_main_module_247 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_245, x_main_module_246);
    auto x_main_module_248 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_247);
    auto x_main_module_249 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_244,
        x_main_module_36);
    auto x_main_module_250 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,48,35,35]}")),
        x_main_module_35);
    auto x_main_module_251 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_249, x_main_module_250);
    auto x_main_module_252 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_251);
    auto x_main_module_253 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[2,2,2,2],padding_mode:0,stride:[1,1]}")),
        x_main_module_252,
        x_main_module_34);
    auto x_main_module_254 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,64,35,35]}")),
        x_main_module_33);
    auto x_main_module_255 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_253, x_main_module_254);
    auto x_main_module_256 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_255);
    auto x_main_module_257 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_244,
        x_main_module_32);
    auto x_main_module_258 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,64,35,35]}")),
        x_main_module_31);
    auto x_main_module_259 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_257, x_main_module_258);
    auto x_main_module_260 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_259);
    auto x_main_module_261 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}")),
        x_main_module_260,
        x_main_module_30);
    auto x_main_module_262 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,96,35,35]}")),
        x_main_module_29);
    auto x_main_module_263 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_261, x_main_module_262);
    auto x_main_module_264 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_263);
    auto x_main_module_265 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}")),
        x_main_module_264,
        x_main_module_28);
    auto x_main_module_266 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,96,35,35]}")),
        x_main_module_27);
    auto x_main_module_267 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_265, x_main_module_266);
    auto x_main_module_268 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_267);
    auto x_main_module_269 = mmain->add_instruction(
        migraphx::make_op("pad",
                          migraphx::from_json_string("{mode:0,pads:[0,0,1,1,0,0,1,1],value:0.0}")),
        x_main_module_244);
    auto x_main_module_270 = mmain->add_instruction(
        migraphx::make_op(
            "pooling",
            migraphx::from_json_string(
                "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:0,padding:[0,0,0,0],stride:[1,1]}")),
        x_main_module_269);
    auto x_main_module_271 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_270,
        x_main_module_26);
    auto x_main_module_272 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,64,35,35]}")),
        x_main_module_25);
    auto x_main_module_273 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_271, x_main_module_272);
    auto x_main_module_274 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_273);
    auto x_main_module_275 =
        mmain->add_instruction(migraphx::make_op("concat", migraphx::from_json_string("{axis:1}")),
                               x_main_module_248,
                               x_main_module_256,
                               x_main_module_268,
                               x_main_module_274);
    auto x_main_module_276 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_275,
        x_main_module_24);
    auto x_main_module_277 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,64,35,35]}")),
        x_main_module_23);
    auto x_main_module_278 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_276, x_main_module_277);
    auto x_main_module_279 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_278);
    auto x_main_module_280 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_275,
        x_main_module_22);
    auto x_main_module_281 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,48,35,35]}")),
        x_main_module_21);
    auto x_main_module_282 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_280, x_main_module_281);
    auto x_main_module_283 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_282);
    auto x_main_module_284 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[2,2,2,2],padding_mode:0,stride:[1,1]}")),
        x_main_module_283,
        x_main_module_20);
    auto x_main_module_285 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,64,35,35]}")),
        x_main_module_19);
    auto x_main_module_286 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_284, x_main_module_285);
    auto x_main_module_287 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_286);
    auto x_main_module_288 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_275,
        x_main_module_18);
    auto x_main_module_289 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,64,35,35]}")),
        x_main_module_17);
    auto x_main_module_290 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_288, x_main_module_289);
    auto x_main_module_291 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_290);
    auto x_main_module_292 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}")),
        x_main_module_291,
        x_main_module_16);
    auto x_main_module_293 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,96,35,35]}")),
        x_main_module_15);
    auto x_main_module_294 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_292, x_main_module_293);
    auto x_main_module_295 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_294);
    auto x_main_module_296 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}")),
        x_main_module_295,
        x_main_module_14);
    auto x_main_module_297 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,96,35,35]}")),
        x_main_module_13);
    auto x_main_module_298 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_296, x_main_module_297);
    auto x_main_module_299 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_298);
    auto x_main_module_300 = mmain->add_instruction(
        migraphx::make_op("pad",
                          migraphx::from_json_string("{mode:0,pads:[0,0,1,1,0,0,1,1],value:0.0}")),
        x_main_module_275);
    auto x_main_module_301 = mmain->add_instruction(
        migraphx::make_op(
            "pooling",
            migraphx::from_json_string(
                "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:0,padding:[0,0,0,0],stride:[1,1]}")),
        x_main_module_300);
    auto x_main_module_302 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_301,
        x_main_module_12);
    auto x_main_module_303 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,64,35,35]}")),
        x_main_module_11);
    auto x_main_module_304 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_302, x_main_module_303);
    auto x_main_module_305 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_304);
    auto x_main_module_306 =
        mmain->add_instruction(migraphx::make_op("concat", migraphx::from_json_string("{axis:1}")),
                               x_main_module_279,
                               x_main_module_287,
                               x_main_module_299,
                               x_main_module_305);
    auto x_main_module_307 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[2,2]}")),
        x_main_module_306,
        x_main_module_10);
    auto x_main_module_308 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,384,17,17]}")),
        x_main_module_9);
    auto x_main_module_309 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_307, x_main_module_308);
    auto x_main_module_310 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_309);
    auto x_main_module_311 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_306,
        x_main_module_8);
    auto x_main_module_312 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,64,35,35]}")),
        x_main_module_7);
    auto x_main_module_313 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_311, x_main_module_312);
    auto x_main_module_314 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_313);
    auto x_main_module_315 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}")),
        x_main_module_314,
        x_main_module_6);
    auto x_main_module_316 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,96,35,35]}")),
        x_main_module_5);
    auto x_main_module_317 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_315, x_main_module_316);
    auto x_main_module_318 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_317);
    auto x_main_module_319 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[2,2]}")),
        x_main_module_318,
        x_main_module_4);
    auto x_main_module_320 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,96,17,17]}")),
        x_main_module_191);
    auto x_main_module_321 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_319, x_main_module_320);
    auto x_main_module_322 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_321);
    auto x_main_module_323 = mmain->add_instruction(
        migraphx::make_op(
            "pooling",
            migraphx::from_json_string(
                "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:1,padding:[0,0,0,0],stride:[2,2]}")),
        x_main_module_306);
    auto x_main_module_324 =
        mmain->add_instruction(migraphx::make_op("concat", migraphx::from_json_string("{axis:1}")),
                               x_main_module_310,
                               x_main_module_322,
                               x_main_module_323);
    auto x_main_module_325 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_324,
        x_main_module_190);
    auto x_main_module_326 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,192,17,17]}")),
        x_main_module_189);
    auto x_main_module_327 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_325, x_main_module_326);
    auto x_main_module_328 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_327);
    auto x_main_module_329 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_324,
        x_main_module_188);
    auto x_main_module_330 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,128,17,17]}")),
        x_main_module_187);
    auto x_main_module_331 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_329, x_main_module_330);
    auto x_main_module_332 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_331);
    auto x_main_module_333 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,1]}")),
        x_main_module_332,
        x_main_module_186);
    auto x_main_module_334 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,128,17,17]}")),
        x_main_module_185);
    auto x_main_module_335 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_333, x_main_module_334);
    auto x_main_module_336 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_335);
    auto x_main_module_337 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_336,
        x_main_module_184);
    auto x_main_module_338 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,192,17,17]}")),
        x_main_module_183);
    auto x_main_module_339 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_337, x_main_module_338);
    auto x_main_module_340 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_339);
    auto x_main_module_341 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_324,
        x_main_module_182);
    auto x_main_module_342 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,128,17,17]}")),
        x_main_module_181);
    auto x_main_module_343 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_341, x_main_module_342);
    auto x_main_module_344 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_343);
    auto x_main_module_345 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_344,
        x_main_module_180);
    auto x_main_module_346 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,128,17,17]}")),
        x_main_module_179);
    auto x_main_module_347 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_345, x_main_module_346);
    auto x_main_module_348 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_347);
    auto x_main_module_349 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,1]}")),
        x_main_module_348,
        x_main_module_178);
    auto x_main_module_350 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,128,17,17]}")),
        x_main_module_177);
    auto x_main_module_351 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_349, x_main_module_350);
    auto x_main_module_352 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_351);
    auto x_main_module_353 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_352,
        x_main_module_176);
    auto x_main_module_354 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,128,17,17]}")),
        x_main_module_175);
    auto x_main_module_355 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_353, x_main_module_354);
    auto x_main_module_356 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_355);
    auto x_main_module_357 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,1]}")),
        x_main_module_356,
        x_main_module_174);
    auto x_main_module_358 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,192,17,17]}")),
        x_main_module_173);
    auto x_main_module_359 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_357, x_main_module_358);
    auto x_main_module_360 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_359);
    auto x_main_module_361 = mmain->add_instruction(
        migraphx::make_op("pad",
                          migraphx::from_json_string("{mode:0,pads:[0,0,1,1,0,0,1,1],value:0.0}")),
        x_main_module_324);
    auto x_main_module_362 = mmain->add_instruction(
        migraphx::make_op(
            "pooling",
            migraphx::from_json_string(
                "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:0,padding:[0,0,0,0],stride:[1,1]}")),
        x_main_module_361);
    auto x_main_module_363 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_362,
        x_main_module_172);
    auto x_main_module_364 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,192,17,17]}")),
        x_main_module_171);
    auto x_main_module_365 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_363, x_main_module_364);
    auto x_main_module_366 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_365);
    auto x_main_module_367 =
        mmain->add_instruction(migraphx::make_op("concat", migraphx::from_json_string("{axis:1}")),
                               x_main_module_328,
                               x_main_module_340,
                               x_main_module_360,
                               x_main_module_366);
    auto x_main_module_368 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_367,
        x_main_module_170);
    auto x_main_module_369 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,192,17,17]}")),
        x_main_module_169);
    auto x_main_module_370 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_368, x_main_module_369);
    auto x_main_module_371 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_370);
    auto x_main_module_372 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_367,
        x_main_module_168);
    auto x_main_module_373 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,160,17,17]}")),
        x_main_module_167);
    auto x_main_module_374 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_372, x_main_module_373);
    auto x_main_module_375 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_374);
    auto x_main_module_376 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,1]}")),
        x_main_module_375,
        x_main_module_166);
    auto x_main_module_377 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,160,17,17]}")),
        x_main_module_165);
    auto x_main_module_378 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_376, x_main_module_377);
    auto x_main_module_379 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_378);
    auto x_main_module_380 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_379,
        x_main_module_164);
    auto x_main_module_381 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,192,17,17]}")),
        x_main_module_163);
    auto x_main_module_382 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_380, x_main_module_381);
    auto x_main_module_383 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_382);
    auto x_main_module_384 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_367,
        x_main_module_162);
    auto x_main_module_385 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,160,17,17]}")),
        x_main_module_161);
    auto x_main_module_386 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_384, x_main_module_385);
    auto x_main_module_387 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_386);
    auto x_main_module_388 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_387,
        x_main_module_160);
    auto x_main_module_389 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,160,17,17]}")),
        x_main_module_159);
    auto x_main_module_390 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_388, x_main_module_389);
    auto x_main_module_391 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_390);
    auto x_main_module_392 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,1]}")),
        x_main_module_391,
        x_main_module_158);
    auto x_main_module_393 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,160,17,17]}")),
        x_main_module_157);
    auto x_main_module_394 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_392, x_main_module_393);
    auto x_main_module_395 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_394);
    auto x_main_module_396 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_395,
        x_main_module_156);
    auto x_main_module_397 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,160,17,17]}")),
        x_main_module_155);
    auto x_main_module_398 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_396, x_main_module_397);
    auto x_main_module_399 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_398);
    auto x_main_module_400 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,1]}")),
        x_main_module_399,
        x_main_module_154);
    auto x_main_module_401 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,192,17,17]}")),
        x_main_module_153);
    auto x_main_module_402 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_400, x_main_module_401);
    auto x_main_module_403 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_402);
    auto x_main_module_404 = mmain->add_instruction(
        migraphx::make_op("pad",
                          migraphx::from_json_string("{mode:0,pads:[0,0,1,1,0,0,1,1],value:0.0}")),
        x_main_module_367);
    auto x_main_module_405 = mmain->add_instruction(
        migraphx::make_op(
            "pooling",
            migraphx::from_json_string(
                "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:0,padding:[0,0,0,0],stride:[1,1]}")),
        x_main_module_404);
    auto x_main_module_406 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_405,
        x_main_module_152);
    auto x_main_module_407 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,192,17,17]}")),
        x_main_module_151);
    auto x_main_module_408 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_406, x_main_module_407);
    auto x_main_module_409 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_408);
    auto x_main_module_410 =
        mmain->add_instruction(migraphx::make_op("concat", migraphx::from_json_string("{axis:1}")),
                               x_main_module_371,
                               x_main_module_383,
                               x_main_module_403,
                               x_main_module_409);
    auto x_main_module_411 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_410,
        x_main_module_150);
    auto x_main_module_412 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,192,17,17]}")),
        x_main_module_149);
    auto x_main_module_413 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_411, x_main_module_412);
    auto x_main_module_414 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_413);
    auto x_main_module_415 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_410,
        x_main_module_148);
    auto x_main_module_416 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,160,17,17]}")),
        x_main_module_147);
    auto x_main_module_417 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_415, x_main_module_416);
    auto x_main_module_418 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_417);
    auto x_main_module_419 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,1]}")),
        x_main_module_418,
        x_main_module_146);
    auto x_main_module_420 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,160,17,17]}")),
        x_main_module_145);
    auto x_main_module_421 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_419, x_main_module_420);
    auto x_main_module_422 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_421);
    auto x_main_module_423 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_422,
        x_main_module_144);
    auto x_main_module_424 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,192,17,17]}")),
        x_main_module_143);
    auto x_main_module_425 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_423, x_main_module_424);
    auto x_main_module_426 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_425);
    auto x_main_module_427 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_410,
        x_main_module_142);
    auto x_main_module_428 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,160,17,17]}")),
        x_main_module_141);
    auto x_main_module_429 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_427, x_main_module_428);
    auto x_main_module_430 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_429);
    auto x_main_module_431 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_430,
        x_main_module_140);
    auto x_main_module_432 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,160,17,17]}")),
        x_main_module_139);
    auto x_main_module_433 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_431, x_main_module_432);
    auto x_main_module_434 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_433);
    auto x_main_module_435 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,1]}")),
        x_main_module_434,
        x_main_module_138);
    auto x_main_module_436 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,160,17,17]}")),
        x_main_module_137);
    auto x_main_module_437 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_435, x_main_module_436);
    auto x_main_module_438 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_437);
    auto x_main_module_439 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_438,
        x_main_module_136);
    auto x_main_module_440 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,160,17,17]}")),
        x_main_module_135);
    auto x_main_module_441 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_439, x_main_module_440);
    auto x_main_module_442 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_441);
    auto x_main_module_443 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,1]}")),
        x_main_module_442,
        x_main_module_134);
    auto x_main_module_444 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,192,17,17]}")),
        x_main_module_133);
    auto x_main_module_445 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_443, x_main_module_444);
    auto x_main_module_446 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_445);
    auto x_main_module_447 = mmain->add_instruction(
        migraphx::make_op("pad",
                          migraphx::from_json_string("{mode:0,pads:[0,0,1,1,0,0,1,1],value:0.0}")),
        x_main_module_410);
    auto x_main_module_448 = mmain->add_instruction(
        migraphx::make_op(
            "pooling",
            migraphx::from_json_string(
                "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:0,padding:[0,0,0,0],stride:[1,1]}")),
        x_main_module_447);
    auto x_main_module_449 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_448,
        x_main_module_132);
    auto x_main_module_450 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,192,17,17]}")),
        x_main_module_131);
    auto x_main_module_451 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_449, x_main_module_450);
    auto x_main_module_452 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_451);
    auto x_main_module_453 =
        mmain->add_instruction(migraphx::make_op("concat", migraphx::from_json_string("{axis:1}")),
                               x_main_module_414,
                               x_main_module_426,
                               x_main_module_446,
                               x_main_module_452);
    auto x_main_module_454 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_453,
        x_main_module_130);
    auto x_main_module_455 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,192,17,17]}")),
        x_main_module_129);
    auto x_main_module_456 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_454, x_main_module_455);
    auto x_main_module_457 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_456);
    auto x_main_module_458 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_453,
        x_main_module_128);
    auto x_main_module_459 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,192,17,17]}")),
        x_main_module_127);
    auto x_main_module_460 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_458, x_main_module_459);
    auto x_main_module_461 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_460);
    auto x_main_module_462 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,1]}")),
        x_main_module_461,
        x_main_module_126);
    auto x_main_module_463 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,192,17,17]}")),
        x_main_module_125);
    auto x_main_module_464 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_462, x_main_module_463);
    auto x_main_module_465 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_464);
    auto x_main_module_466 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_465,
        x_main_module_124);
    auto x_main_module_467 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,192,17,17]}")),
        x_main_module_123);
    auto x_main_module_468 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_466, x_main_module_467);
    auto x_main_module_469 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_468);
    auto x_main_module_470 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_453,
        x_main_module_122);
    auto x_main_module_471 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,192,17,17]}")),
        x_main_module_121);
    auto x_main_module_472 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_470, x_main_module_471);
    auto x_main_module_473 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_472);
    auto x_main_module_474 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_473,
        x_main_module_120);
    auto x_main_module_475 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,192,17,17]}")),
        x_main_module_119);
    auto x_main_module_476 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_474, x_main_module_475);
    auto x_main_module_477 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_476);
    auto x_main_module_478 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,1]}")),
        x_main_module_477,
        x_main_module_118);
    auto x_main_module_479 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,192,17,17]}")),
        x_main_module_117);
    auto x_main_module_480 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_478, x_main_module_479);
    auto x_main_module_481 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_480);
    auto x_main_module_482 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_481,
        x_main_module_116);
    auto x_main_module_483 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,192,17,17]}")),
        x_main_module_115);
    auto x_main_module_484 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_482, x_main_module_483);
    auto x_main_module_485 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_484);
    auto x_main_module_486 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,1]}")),
        x_main_module_485,
        x_main_module_114);
    auto x_main_module_487 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,192,17,17]}")),
        x_main_module_113);
    auto x_main_module_488 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_486, x_main_module_487);
    auto x_main_module_489 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_488);
    auto x_main_module_490 = mmain->add_instruction(
        migraphx::make_op("pad",
                          migraphx::from_json_string("{mode:0,pads:[0,0,1,1,0,0,1,1],value:0.0}")),
        x_main_module_453);
    auto x_main_module_491 = mmain->add_instruction(
        migraphx::make_op(
            "pooling",
            migraphx::from_json_string(
                "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:0,padding:[0,0,0,0],stride:[1,1]}")),
        x_main_module_490);
    auto x_main_module_492 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_491,
        x_main_module_112);
    auto x_main_module_493 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,192,17,17]}")),
        x_main_module_111);
    auto x_main_module_494 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_492, x_main_module_493);
    auto x_main_module_495 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_494);
    auto x_main_module_496 =
        mmain->add_instruction(migraphx::make_op("concat", migraphx::from_json_string("{axis:1}")),
                               x_main_module_457,
                               x_main_module_469,
                               x_main_module_489,
                               x_main_module_495);
    auto x_main_module_497 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_496,
        x_main_module_110);
    auto x_main_module_498 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,192,17,17]}")),
        x_main_module_109);
    auto x_main_module_499 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_497, x_main_module_498);
    auto x_main_module_500 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_499);
    auto x_main_module_501 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[2,2]}")),
        x_main_module_500,
        x_main_module_108);
    auto x_main_module_502 = mmain->add_instruction(
        migraphx::make_op("broadcast", migraphx::from_json_string("{axis:1,out_lens:[1,320,8,8]}")),
        x_main_module_107);
    auto x_main_module_503 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_501, x_main_module_502);
    auto x_main_module_504 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_503);
    auto x_main_module_505 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_496,
        x_main_module_106);
    auto x_main_module_506 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,192,17,17]}")),
        x_main_module_105);
    auto x_main_module_507 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_505, x_main_module_506);
    auto x_main_module_508 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_507);
    auto x_main_module_509 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,1]}")),
        x_main_module_508,
        x_main_module_104);
    auto x_main_module_510 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,192,17,17]}")),
        x_main_module_103);
    auto x_main_module_511 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_509, x_main_module_510);
    auto x_main_module_512 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_511);
    auto x_main_module_513 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_512,
        x_main_module_102);
    auto x_main_module_514 = mmain->add_instruction(
        migraphx::make_op("broadcast",
                          migraphx::from_json_string("{axis:1,out_lens:[1,192,17,17]}")),
        x_main_module_101);
    auto x_main_module_515 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_513, x_main_module_514);
    auto x_main_module_516 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_515);
    auto x_main_module_517 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[2,2]}")),
        x_main_module_516,
        x_main_module_100);
    auto x_main_module_518 = mmain->add_instruction(
        migraphx::make_op("broadcast", migraphx::from_json_string("{axis:1,out_lens:[1,192,8,8]}")),
        x_main_module_99);
    auto x_main_module_519 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_517, x_main_module_518);
    auto x_main_module_520 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_519);
    auto x_main_module_521 = mmain->add_instruction(
        migraphx::make_op(
            "pooling",
            migraphx::from_json_string(
                "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:1,padding:[0,0,0,0],stride:[2,2]}")),
        x_main_module_496);
    auto x_main_module_522 =
        mmain->add_instruction(migraphx::make_op("concat", migraphx::from_json_string("{axis:1}")),
                               x_main_module_504,
                               x_main_module_520,
                               x_main_module_521);
    auto x_main_module_523 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_522,
        x_main_module_98);
    auto x_main_module_524 = mmain->add_instruction(
        migraphx::make_op("broadcast", migraphx::from_json_string("{axis:1,out_lens:[1,320,8,8]}")),
        x_main_module_97);
    auto x_main_module_525 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_523, x_main_module_524);
    auto x_main_module_526 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_525);
    auto x_main_module_527 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_522,
        x_main_module_96);
    auto x_main_module_528 = mmain->add_instruction(
        migraphx::make_op("broadcast", migraphx::from_json_string("{axis:1,out_lens:[1,384,8,8]}")),
        x_main_module_95);
    auto x_main_module_529 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_527, x_main_module_528);
    auto x_main_module_530 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_529);
    auto x_main_module_531 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,1,0,1],padding_mode:0,stride:[1,1]}")),
        x_main_module_530,
        x_main_module_94);
    auto x_main_module_532 = mmain->add_instruction(
        migraphx::make_op("broadcast", migraphx::from_json_string("{axis:1,out_lens:[1,384,8,8]}")),
        x_main_module_93);
    auto x_main_module_533 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_531, x_main_module_532);
    auto x_main_module_534 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_533);
    auto x_main_module_535 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[1,0,1,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_530,
        x_main_module_92);
    auto x_main_module_536 = mmain->add_instruction(
        migraphx::make_op("broadcast", migraphx::from_json_string("{axis:1,out_lens:[1,384,8,8]}")),
        x_main_module_91);
    auto x_main_module_537 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_535, x_main_module_536);
    auto x_main_module_538 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_537);
    auto x_main_module_539 =
        mmain->add_instruction(migraphx::make_op("concat", migraphx::from_json_string("{axis:1}")),
                               x_main_module_534,
                               x_main_module_538);
    auto x_main_module_540 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_522,
        x_main_module_90);
    auto x_main_module_541 = mmain->add_instruction(
        migraphx::make_op("broadcast", migraphx::from_json_string("{axis:1,out_lens:[1,448,8,8]}")),
        x_main_module_89);
    auto x_main_module_542 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_540, x_main_module_541);
    auto x_main_module_543 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_542);
    auto x_main_module_544 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}")),
        x_main_module_543,
        x_main_module_88);
    auto x_main_module_545 = mmain->add_instruction(
        migraphx::make_op("broadcast", migraphx::from_json_string("{axis:1,out_lens:[1,384,8,8]}")),
        x_main_module_87);
    auto x_main_module_546 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_544, x_main_module_545);
    auto x_main_module_547 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_546);
    auto x_main_module_548 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,1,0,1],padding_mode:0,stride:[1,1]}")),
        x_main_module_547,
        x_main_module_86);
    auto x_main_module_549 = mmain->add_instruction(
        migraphx::make_op("broadcast", migraphx::from_json_string("{axis:1,out_lens:[1,384,8,8]}")),
        x_main_module_85);
    auto x_main_module_550 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_548, x_main_module_549);
    auto x_main_module_551 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_550);
    auto x_main_module_552 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[1,0,1,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_547,
        x_main_module_84);
    auto x_main_module_553 = mmain->add_instruction(
        migraphx::make_op("broadcast", migraphx::from_json_string("{axis:1,out_lens:[1,384,8,8]}")),
        x_main_module_83);
    auto x_main_module_554 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_552, x_main_module_553);
    auto x_main_module_555 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_554);
    auto x_main_module_556 =
        mmain->add_instruction(migraphx::make_op("concat", migraphx::from_json_string("{axis:1}")),
                               x_main_module_551,
                               x_main_module_555);
    auto x_main_module_557 = mmain->add_instruction(
        migraphx::make_op("pad",
                          migraphx::from_json_string("{mode:0,pads:[0,0,1,1,0,0,1,1],value:0.0}")),
        x_main_module_522);
    auto x_main_module_558 = mmain->add_instruction(
        migraphx::make_op(
            "pooling",
            migraphx::from_json_string(
                "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:0,padding:[0,0,0,0],stride:[1,1]}")),
        x_main_module_557);
    auto x_main_module_559 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_558,
        x_main_module_82);
    auto x_main_module_560 = mmain->add_instruction(
        migraphx::make_op("broadcast", migraphx::from_json_string("{axis:1,out_lens:[1,192,8,8]}")),
        x_main_module_81);
    auto x_main_module_561 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_559, x_main_module_560);
    auto x_main_module_562 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_561);
    auto x_main_module_563 =
        mmain->add_instruction(migraphx::make_op("concat", migraphx::from_json_string("{axis:1}")),
                               x_main_module_526,
                               x_main_module_539,
                               x_main_module_556,
                               x_main_module_562);
    auto x_main_module_564 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_563,
        x_main_module_80);
    auto x_main_module_565 = mmain->add_instruction(
        migraphx::make_op("broadcast", migraphx::from_json_string("{axis:1,out_lens:[1,320,8,8]}")),
        x_main_module_79);
    auto x_main_module_566 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_564, x_main_module_565);
    auto x_main_module_567 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_566);
    auto x_main_module_568 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_563,
        x_main_module_78);
    auto x_main_module_569 = mmain->add_instruction(
        migraphx::make_op("broadcast", migraphx::from_json_string("{axis:1,out_lens:[1,384,8,8]}")),
        x_main_module_77);
    auto x_main_module_570 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_568, x_main_module_569);
    auto x_main_module_571 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_570);
    auto x_main_module_572 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,1,0,1],padding_mode:0,stride:[1,1]}")),
        x_main_module_571,
        x_main_module_76);
    auto x_main_module_573 = mmain->add_instruction(
        migraphx::make_op("broadcast", migraphx::from_json_string("{axis:1,out_lens:[1,384,8,8]}")),
        x_main_module_75);
    auto x_main_module_574 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_572, x_main_module_573);
    auto x_main_module_575 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_574);
    auto x_main_module_576 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[1,0,1,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_571,
        x_main_module_74);
    auto x_main_module_577 = mmain->add_instruction(
        migraphx::make_op("broadcast", migraphx::from_json_string("{axis:1,out_lens:[1,384,8,8]}")),
        x_main_module_73);
    auto x_main_module_578 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_576, x_main_module_577);
    auto x_main_module_579 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_578);
    auto x_main_module_580 =
        mmain->add_instruction(migraphx::make_op("concat", migraphx::from_json_string("{axis:1}")),
                               x_main_module_575,
                               x_main_module_579);
    auto x_main_module_581 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_563,
        x_main_module_72);
    auto x_main_module_582 = mmain->add_instruction(
        migraphx::make_op("broadcast", migraphx::from_json_string("{axis:1,out_lens:[1,448,8,8]}")),
        x_main_module_71);
    auto x_main_module_583 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_581, x_main_module_582);
    auto x_main_module_584 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_583);
    auto x_main_module_585 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}")),
        x_main_module_584,
        x_main_module_70);
    auto x_main_module_586 = mmain->add_instruction(
        migraphx::make_op("broadcast", migraphx::from_json_string("{axis:1,out_lens:[1,384,8,8]}")),
        x_main_module_69);
    auto x_main_module_587 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_585, x_main_module_586);
    auto x_main_module_588 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_587);
    auto x_main_module_589 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,1,0,1],padding_mode:0,stride:[1,1]}")),
        x_main_module_588,
        x_main_module_68);
    auto x_main_module_590 = mmain->add_instruction(
        migraphx::make_op("broadcast", migraphx::from_json_string("{axis:1,out_lens:[1,384,8,8]}")),
        x_main_module_67);
    auto x_main_module_591 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_589, x_main_module_590);
    auto x_main_module_592 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_591);
    auto x_main_module_593 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[1,0,1,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_588,
        x_main_module_66);
    auto x_main_module_594 = mmain->add_instruction(
        migraphx::make_op("broadcast", migraphx::from_json_string("{axis:1,out_lens:[1,384,8,8]}")),
        x_main_module_65);
    auto x_main_module_595 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_593, x_main_module_594);
    auto x_main_module_596 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_595);
    auto x_main_module_597 =
        mmain->add_instruction(migraphx::make_op("concat", migraphx::from_json_string("{axis:1}")),
                               x_main_module_592,
                               x_main_module_596);
    auto x_main_module_598 = mmain->add_instruction(
        migraphx::make_op("pad",
                          migraphx::from_json_string("{mode:0,pads:[0,0,1,1,0,0,1,1],value:0.0}")),
        x_main_module_563);
    auto x_main_module_599 = mmain->add_instruction(
        migraphx::make_op(
            "pooling",
            migraphx::from_json_string(
                "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:0,padding:[0,0,0,0],stride:[1,1]}")),
        x_main_module_598);
    auto x_main_module_600 = mmain->add_instruction(
        migraphx::make_op(
            "convolution",
            migraphx::from_json_string(
                "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}")),
        x_main_module_599,
        x_main_module_64);
    auto x_main_module_601 = mmain->add_instruction(
        migraphx::make_op("broadcast", migraphx::from_json_string("{axis:1,out_lens:[1,192,8,8]}")),
        x_main_module_63);
    auto x_main_module_602 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_600, x_main_module_601);
    auto x_main_module_603 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_602);
    auto x_main_module_604 =
        mmain->add_instruction(migraphx::make_op("concat", migraphx::from_json_string("{axis:1}")),
                               x_main_module_567,
                               x_main_module_580,
                               x_main_module_597,
                               x_main_module_603);
    auto x_main_module_605 =
        mmain->add_instruction(migraphx::make_op("identity"), x_main_module_604);
    auto x_main_module_606 = mmain->add_instruction(
        migraphx::make_op(
            "pooling",
            migraphx::from_json_string(
                "{ceil_mode:0,lengths:[8,8],lp_order:2,mode:0,padding:[0,0,0,0],stride:[8,8]}")),
        x_main_module_605);
    auto x_main_module_607 = mmain->add_instruction(
        migraphx::make_op("reshape", migraphx::from_json_string("{dims:[1,-1]}")),
        x_main_module_606);
    auto x_main_module_608 = mmain->add_instruction(
        migraphx::make_op("transpose", migraphx::from_json_string("{permutation:[1,0]}")),
        x_main_module_2);
    auto x_main_module_609 =
        mmain->add_instruction(migraphx::make_op("dot"), x_main_module_607, x_main_module_608);
    auto x_main_module_610 = mmain->add_instruction(
        migraphx::make_op("multibroadcast", migraphx::from_json_string("{out_lens:[1,1000]}")),
        x_main_module_3);
    auto x_main_module_611 = mmain->add_instruction(
        migraphx::make_op("multibroadcast", migraphx::from_json_string("{out_lens:[1,1000]}")),
        x_main_module_0);
    auto x_main_module_612 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_610, x_main_module_611);
    auto x_main_module_613 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_609, x_main_module_612);
    mmain->add_return({x_main_module_613});

    return p;
}
} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx
