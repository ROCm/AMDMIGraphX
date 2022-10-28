
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
migraphx::program resnet50(unsigned batch) // NOLINT(readability-function-size)
{
    migraphx::program p;
    migraphx::module_ref mmain = p.get_main_module();
    auto x_main_module_0       = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 0)));
    auto x_main_module_1       = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 1)));
    auto x_main_module_2       = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 2)));
    auto x_main_module_3       = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 3)));
    auto x_main_module_4       = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 4)));
    auto x_main_module_5       = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 5)));
    auto x_main_module_6       = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 6)));
    auto x_main_module_7       = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 7)));
    auto x_main_module_8       = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 8)));
    auto x_main_module_9       = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 9)));
    auto x_main_module_10      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 10)));
    auto x_main_module_11      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 11)));
    auto x_main_module_12      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 12)));
    auto x_main_module_13      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 13)));
    auto x_main_module_14      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 14)));
    auto x_main_module_15      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 15)));
    auto x_main_module_16      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 16)));
    auto x_main_module_17      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 17)));
    auto x_main_module_18      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 18)));
    auto x_main_module_19      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 19)));
    auto x_main_module_20      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 20)));
    auto x_main_module_21      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 21)));
    auto x_main_module_22      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 22)));
    auto x_main_module_23      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 23)));
    auto x_main_module_24      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 24)));
    auto x_main_module_25      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 25)));
    auto x_main_module_26      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 26)));
    auto x_main_module_27      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 27)));
    auto x_main_module_28      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 28)));
    auto x_main_module_29      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 29)));
    auto x_main_module_30      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 30)));
    auto x_main_module_31      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 31)));
    auto x_main_module_32      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 32)));
    auto x_main_module_33      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 33)));
    auto x_main_module_34      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 34)));
    auto x_main_module_35      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 35)));
    auto x_main_module_36      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 36)));
    auto x_main_module_37      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 37)));
    auto x_main_module_38      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 38)));
    auto x_main_module_39      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 39)));
    auto x_main_module_40      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 40)));
    auto x_main_module_41      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 41)));
    auto x_main_module_42      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 42)));
    auto x_main_module_43      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 43)));
    auto x_main_module_44      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 44)));
    auto x_main_module_45      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 45)));
    auto x_main_module_46      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 46)));
    auto x_main_module_47      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 47)));
    auto x_main_module_48      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 48)));
    auto x_main_module_49      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 49)));
    auto x_main_module_50      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 50)));
    auto x_main_module_51      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 51)));
    auto x_main_module_52      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 52)));
    auto x_main_module_53      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 53)));
    auto x_main_module_54      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 54)));
    auto x_main_module_55      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 55)));
    auto x_main_module_56      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 56)));
    auto x_main_module_57      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 57)));
    auto x_main_module_58      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 58)));
    auto x_main_module_59      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 59)));
    auto x_main_module_60      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 60)));
    auto x_main_module_61      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 61)));
    auto x_main_module_62      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 62)));
    auto x_main_module_63      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 63)));
    auto x_main_module_64      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 64)));
    auto x_main_module_65      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 65)));
    auto x_main_module_66      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 66)));
    auto x_main_module_67      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 67)));
    auto x_main_module_68      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 68)));
    auto x_main_module_69      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 69)));
    auto x_main_module_70      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 70)));
    auto x_main_module_71      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 71)));
    auto x_main_module_72      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 72)));
    auto x_main_module_73      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 73)));
    auto x_main_module_74      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 74)));
    auto x_main_module_75      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 75)));
    auto x_main_module_76      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 76)));
    auto x_main_module_77      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 77)));
    auto x_main_module_78      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 78)));
    auto x_main_module_79      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 79)));
    auto x_main_module_80      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 80)));
    auto x_main_module_81      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 81)));
    auto x_main_module_82      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 82)));
    auto x_main_module_83      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 83)));
    auto x_main_module_84      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 84)));
    auto x_main_module_85      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 85)));
    auto x_main_module_86      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 86)));
    auto x_main_module_87      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 87)));
    auto x_main_module_88      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 88)));
    auto x_main_module_89      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 89)));
    auto x_main_module_90      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 90)));
    auto x_main_module_91      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 91)));
    auto x_main_module_92      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 92)));
    auto x_main_module_93      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 93)));
    auto x_main_module_94      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 94)));
    auto x_main_module_95      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 95)));
    auto x_main_module_96      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 96)));
    auto x_main_module_97      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 97)));
    auto x_main_module_98      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 98)));
    auto x_main_module_99      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 99)));
    auto x_main_module_100     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 100)));
    auto x_main_module_101     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 101)));
    auto x_main_module_102     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 102)));
    auto x_main_module_103     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 103)));
    auto x_main_module_104     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 104)));
    auto x_main_module_105     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 105)));
    auto x_main_module_106     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 106)));
    auto x_0                   = mmain->add_parameter(
        "0", migraphx::shape{migraphx::shape::float_type, {batch, 3, 224, 224}});
    auto x_main_module_108 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1000}}, 107));
    auto x_main_module_109 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1000, 2048}}, 108));
    auto x_main_module_110 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 109)));
    auto x_main_module_111 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 110));
    auto x_main_module_112 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 111));
    auto x_main_module_113 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 112)));
    auto x_main_module_114 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {2048, 512, 1, 1}}, 113));
    auto x_main_module_115 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 114)));
    auto x_main_module_116 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 115));
    auto x_main_module_117 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 116));
    auto x_main_module_118 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 117)));
    auto x_main_module_119 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 512, 3, 3}}, 118));
    auto x_main_module_120 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 119)));
    auto x_main_module_121 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 120));
    auto x_main_module_122 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 121));
    auto x_main_module_123 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 122)));
    auto x_main_module_124 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 2048, 1, 1}}, 123));
    auto x_main_module_125 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 124)));
    auto x_main_module_126 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 125));
    auto x_main_module_127 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 126));
    auto x_main_module_128 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 127)));
    auto x_main_module_129 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {2048, 512, 1, 1}}, 128));
    auto x_main_module_130 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 129)));
    auto x_main_module_131 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 130));
    auto x_main_module_132 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 131));
    auto x_main_module_133 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 132)));
    auto x_main_module_134 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 512, 3, 3}}, 133));
    auto x_main_module_135 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 134)));
    auto x_main_module_136 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 135));
    auto x_main_module_137 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 136));
    auto x_main_module_138 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 137)));
    auto x_main_module_139 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 2048, 1, 1}}, 138));
    auto x_main_module_140 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 139)));
    auto x_main_module_141 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 140));
    auto x_main_module_142 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 141));
    auto x_main_module_143 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 142)));
    auto x_main_module_144 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {2048, 1024, 1, 1}}, 143));
    auto x_main_module_145 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 144)));
    auto x_main_module_146 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 145));
    auto x_main_module_147 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 146));
    auto x_main_module_148 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 147)));
    auto x_main_module_149 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {2048, 512, 1, 1}}, 148));
    auto x_main_module_150 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 149)));
    auto x_main_module_151 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 150));
    auto x_main_module_152 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 151));
    auto x_main_module_153 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 152)));
    auto x_main_module_154 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 512, 3, 3}}, 153));
    auto x_main_module_155 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 154)));
    auto x_main_module_156 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 155));
    auto x_main_module_157 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 156));
    auto x_main_module_158 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 157)));
    auto x_main_module_159 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 1024, 1, 1}}, 158));
    auto x_main_module_160 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 159)));
    auto x_main_module_161 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 160));
    auto x_main_module_162 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 161));
    auto x_main_module_163 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 162));
    auto x_main_module_164 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 256, 1, 1}}, 163));
    auto x_main_module_165 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 164)));
    auto x_main_module_166 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 165));
    auto x_main_module_167 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 166));
    auto x_main_module_168 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 167)));
    auto x_main_module_169 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 256, 3, 3}}, 168));
    auto x_main_module_170 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 169)));
    auto x_main_module_171 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 170));
    auto x_main_module_172 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 171));
    auto x_main_module_173 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 172)));
    auto x_main_module_174 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 1024, 1, 1}}, 173));
    auto x_main_module_175 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 174)));
    auto x_main_module_176 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 175));
    auto x_main_module_177 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 176));
    auto x_main_module_178 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 177));
    auto x_main_module_179 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 256, 1, 1}}, 178));
    auto x_main_module_180 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 179)));
    auto x_main_module_181 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 180));
    auto x_main_module_182 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 181));
    auto x_main_module_183 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 182)));
    auto x_main_module_184 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 256, 3, 3}}, 183));
    auto x_main_module_185 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 184)));
    auto x_main_module_186 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 185));
    auto x_main_module_187 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 186));
    auto x_main_module_188 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 187)));
    auto x_main_module_189 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 1024, 1, 1}}, 188));
    auto x_main_module_190 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 189)));
    auto x_main_module_191 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 190));
    auto x_main_module_192 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 191));
    auto x_main_module_193 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 192));
    auto x_main_module_194 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 256, 1, 1}}, 193));
    auto x_main_module_195 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 194)));
    auto x_main_module_196 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 195));
    auto x_main_module_197 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 196));
    auto x_main_module_198 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 197)));
    auto x_main_module_199 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 256, 3, 3}}, 198));
    auto x_main_module_200 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 199)));
    auto x_main_module_201 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 200));
    auto x_main_module_202 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 201));
    auto x_main_module_203 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 202)));
    auto x_main_module_204 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 1024, 1, 1}}, 203));
    auto x_main_module_205 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 204)));
    auto x_main_module_206 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 205));
    auto x_main_module_207 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 206));
    auto x_main_module_208 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 207));
    auto x_main_module_209 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 256, 1, 1}}, 208));
    auto x_main_module_210 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 209)));
    auto x_main_module_211 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 210));
    auto x_main_module_212 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 211));
    auto x_main_module_213 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 212)));
    auto x_main_module_214 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 256, 3, 3}}, 213));
    auto x_main_module_215 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 214)));
    auto x_main_module_216 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 215));
    auto x_main_module_217 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 216));
    auto x_main_module_218 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 217)));
    auto x_main_module_219 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 1024, 1, 1}}, 218));
    auto x_main_module_220 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 219)));
    auto x_main_module_221 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 220));
    auto x_main_module_222 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 221));
    auto x_main_module_223 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 222));
    auto x_main_module_224 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 256, 1, 1}}, 223));
    auto x_main_module_225 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 224)));
    auto x_main_module_226 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 225));
    auto x_main_module_227 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 226));
    auto x_main_module_228 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 227)));
    auto x_main_module_229 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 256, 3, 3}}, 228));
    auto x_main_module_230 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 229)));
    auto x_main_module_231 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 230));
    auto x_main_module_232 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 231));
    auto x_main_module_233 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 232)));
    auto x_main_module_234 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 1024, 1, 1}}, 233));
    auto x_main_module_235 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 234)));
    auto x_main_module_236 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 235));
    auto x_main_module_237 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 236));
    auto x_main_module_238 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 237));
    auto x_main_module_239 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 512, 1, 1}}, 238));
    auto x_main_module_240 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 239)));
    auto x_main_module_241 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 240));
    auto x_main_module_242 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 241));
    auto x_main_module_243 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 242));
    auto x_main_module_244 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 256, 1, 1}}, 243));
    auto x_main_module_245 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 244)));
    auto x_main_module_246 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 245));
    auto x_main_module_247 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 246));
    auto x_main_module_248 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 247)));
    auto x_main_module_249 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 256, 3, 3}}, 248));
    auto x_main_module_250 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 249)));
    auto x_main_module_251 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 250));
    auto x_main_module_252 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 251));
    auto x_main_module_253 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 252)));
    auto x_main_module_254 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 512, 1, 1}}, 253));
    auto x_main_module_255 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 254)));
    auto x_main_module_256 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 255));
    auto x_main_module_257 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 256));
    auto x_main_module_258 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 257));
    auto x_main_module_259 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 128, 1, 1}}, 258));
    auto x_main_module_260 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 259)));
    auto x_main_module_261 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 260));
    auto x_main_module_262 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 261));
    auto x_main_module_263 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 262)));
    auto x_main_module_264 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 3, 3}}, 263));
    auto x_main_module_265 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 264)));
    auto x_main_module_266 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 265));
    auto x_main_module_267 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 266));
    auto x_main_module_268 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 267)));
    auto x_main_module_269 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 512, 1, 1}}, 268));
    auto x_main_module_270 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 269)));
    auto x_main_module_271 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 270));
    auto x_main_module_272 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 271));
    auto x_main_module_273 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 272));
    auto x_main_module_274 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 128, 1, 1}}, 273));
    auto x_main_module_275 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 274)));
    auto x_main_module_276 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 275));
    auto x_main_module_277 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 276));
    auto x_main_module_278 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 277)));
    auto x_main_module_279 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 3, 3}}, 278));
    auto x_main_module_280 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 279)));
    auto x_main_module_281 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 280));
    auto x_main_module_282 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 281));
    auto x_main_module_283 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 282)));
    auto x_main_module_284 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 512, 1, 1}}, 283));
    auto x_main_module_285 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 284)));
    auto x_main_module_286 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 285));
    auto x_main_module_287 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 286));
    auto x_main_module_288 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 287));
    auto x_main_module_289 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 128, 1, 1}}, 288));
    auto x_main_module_290 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 289)));
    auto x_main_module_291 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 290));
    auto x_main_module_292 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 291));
    auto x_main_module_293 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 292)));
    auto x_main_module_294 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 3, 3}}, 293));
    auto x_main_module_295 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 294)));
    auto x_main_module_296 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 295));
    auto x_main_module_297 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 296));
    auto x_main_module_298 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 297)));
    auto x_main_module_299 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 512, 1, 1}}, 298));
    auto x_main_module_300 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 299)));
    auto x_main_module_301 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 300));
    auto x_main_module_302 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 301));
    auto x_main_module_303 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 302));
    auto x_main_module_304 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 256, 1, 1}}, 303));
    auto x_main_module_305 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 304)));
    auto x_main_module_306 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 305));
    auto x_main_module_307 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 306));
    auto x_main_module_308 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 307));
    auto x_main_module_309 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 128, 1, 1}}, 308));
    auto x_main_module_310 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 309)));
    auto x_main_module_311 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 310));
    auto x_main_module_312 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 311));
    auto x_main_module_313 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 312)));
    auto x_main_module_314 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 3, 3}}, 313));
    auto x_main_module_315 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 314)));
    auto x_main_module_316 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 315));
    auto x_main_module_317 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 316));
    auto x_main_module_318 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 317)));
    auto x_main_module_319 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 256, 1, 1}}, 318));
    auto x_main_module_320 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 319)));
    auto x_main_module_321 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 320));
    auto x_main_module_322 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 321));
    auto x_main_module_323 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 322));
    auto x_main_module_324 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 64, 1, 1}}, 323));
    auto x_main_module_325 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 324)));
    auto x_main_module_326 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 325));
    auto x_main_module_327 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 326));
    auto x_main_module_328 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 327)));
    auto x_main_module_329 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 64, 3, 3}}, 328));
    auto x_main_module_330 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 329)));
    auto x_main_module_331 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 330));
    auto x_main_module_332 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 331));
    auto x_main_module_333 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 332)));
    auto x_main_module_334 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 256, 1, 1}}, 333));
    auto x_main_module_335 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 334)));
    auto x_main_module_336 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 335));
    auto x_main_module_337 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 336));
    auto x_main_module_338 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 337));
    auto x_main_module_339 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 64, 1, 1}}, 338));
    auto x_main_module_340 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 339)));
    auto x_main_module_341 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 340));
    auto x_main_module_342 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 341));
    auto x_main_module_343 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 342)));
    auto x_main_module_344 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 64, 3, 3}}, 343));
    auto x_main_module_345 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 344)));
    auto x_main_module_346 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 345));
    auto x_main_module_347 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 346));
    auto x_main_module_348 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 347)));
    auto x_main_module_349 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 256, 1, 1}}, 348));
    auto x_main_module_350 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 349)));
    auto x_main_module_351 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 350));
    auto x_main_module_352 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 351));
    auto x_main_module_353 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 352));
    auto x_main_module_354 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 64, 1, 1}}, 353));
    auto x_main_module_355 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 354)));
    auto x_main_module_356 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 355));
    auto x_main_module_357 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 356));
    auto x_main_module_358 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 357));
    auto x_main_module_359 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 64, 1, 1}}, 358));
    auto x_main_module_360 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 359)));
    auto x_main_module_361 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 360));
    auto x_main_module_362 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 361));
    auto x_main_module_363 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 362)));
    auto x_main_module_364 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 64, 3, 3}}, 363));
    auto x_main_module_365 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 364)));
    auto x_main_module_366 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 365));
    auto x_main_module_367 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 366));
    auto x_main_module_368 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 367)));
    auto x_main_module_369 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 64, 1, 1}}, 368));
    auto x_main_module_370 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 369)));
    auto x_main_module_371 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 370));
    auto x_main_module_372 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 371));
    auto x_main_module_373 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 372)));
    auto x_main_module_374 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 3, 7, 7}}, 373));
    auto x_main_module_375 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[3,3,3,3],padding_mode:0,stride:[2,2]}"),
        x_0,
        x_main_module_374);
    auto x_main_module_376 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_373);
    auto x_main_module_377 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_372);
    auto x_main_module_378 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_371);
    auto x_main_module_379 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_370);
    auto x_main_module_380 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,112,112]}"), x_main_module_378);
    auto x_main_module_381 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_375, x_main_module_380);
    auto x_main_module_382 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_105);
    auto x_main_module_383 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_379, x_main_module_382);
    auto x_main_module_384 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_106);
    auto x_main_module_385 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_383, x_main_module_384);
    auto x_main_module_386 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,112,112]}"), x_main_module_385);
    auto x_main_module_387 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_381, x_main_module_386);
    auto x_main_module_388 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,112,112]}"), x_main_module_376);
    auto x_main_module_389 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_387, x_main_module_388);
    auto x_main_module_390 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,112,112]}"), x_main_module_377);
    auto x_main_module_391 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_389, x_main_module_390);
    auto x_main_module_392 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_391);
    auto x_main_module_393 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:1,padding:[1,1,1,1],stride:[2,2]}"),
        x_main_module_392);
    auto x_main_module_394 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_393,
        x_main_module_369);
    auto x_main_module_395 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_368);
    auto x_main_module_396 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_367);
    auto x_main_module_397 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_366);
    auto x_main_module_398 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_365);
    auto x_main_module_399 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,56,56]}"), x_main_module_397);
    auto x_main_module_400 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_394, x_main_module_399);
    auto x_main_module_401 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_103);
    auto x_main_module_402 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_398, x_main_module_401);
    auto x_main_module_403 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_104);
    auto x_main_module_404 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_402, x_main_module_403);
    auto x_main_module_405 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,56,56]}"), x_main_module_404);
    auto x_main_module_406 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_400, x_main_module_405);
    auto x_main_module_407 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,56,56]}"), x_main_module_395);
    auto x_main_module_408 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_406, x_main_module_407);
    auto x_main_module_409 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,56,56]}"), x_main_module_396);
    auto x_main_module_410 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_408, x_main_module_409);
    auto x_main_module_411 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_410);
    auto x_main_module_412 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}"),
        x_main_module_411,
        x_main_module_364);
    auto x_main_module_413 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_363);
    auto x_main_module_414 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_362);
    auto x_main_module_415 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_361);
    auto x_main_module_416 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_360);
    auto x_main_module_417 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,56,56]}"), x_main_module_415);
    auto x_main_module_418 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_412, x_main_module_417);
    auto x_main_module_419 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_101);
    auto x_main_module_420 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_416, x_main_module_419);
    auto x_main_module_421 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_102);
    auto x_main_module_422 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_420, x_main_module_421);
    auto x_main_module_423 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,56,56]}"), x_main_module_422);
    auto x_main_module_424 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_418, x_main_module_423);
    auto x_main_module_425 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,56,56]}"), x_main_module_413);
    auto x_main_module_426 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_424, x_main_module_425);
    auto x_main_module_427 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,56,56]}"), x_main_module_414);
    auto x_main_module_428 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_426, x_main_module_427);
    auto x_main_module_429 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_428);
    auto x_main_module_430 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_429,
        x_main_module_359);
    auto x_main_module_431 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_358);
    auto x_main_module_432 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_357);
    auto x_main_module_433 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_356);
    auto x_main_module_434 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_355);
    auto x_main_module_435 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,56,56]}"), x_main_module_433);
    auto x_main_module_436 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_430, x_main_module_435);
    auto x_main_module_437 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[256,1,1]}"), x_main_module_99);
    auto x_main_module_438 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_434, x_main_module_437);
    auto x_main_module_439 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[256,1,1]}"), x_main_module_100);
    auto x_main_module_440 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_438, x_main_module_439);
    auto x_main_module_441 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,56,56]}"), x_main_module_440);
    auto x_main_module_442 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_436, x_main_module_441);
    auto x_main_module_443 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,56,56]}"), x_main_module_431);
    auto x_main_module_444 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_442, x_main_module_443);
    auto x_main_module_445 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,56,56]}"), x_main_module_432);
    auto x_main_module_446 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_444, x_main_module_445);
    auto x_main_module_447 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_393,
        x_main_module_354);
    auto x_main_module_448 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_353);
    auto x_main_module_449 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_352);
    auto x_main_module_450 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_351);
    auto x_main_module_451 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_350);
    auto x_main_module_452 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,56,56]}"), x_main_module_450);
    auto x_main_module_453 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_447, x_main_module_452);
    auto x_main_module_454 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[256,1,1]}"), x_main_module_97);
    auto x_main_module_455 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_451, x_main_module_454);
    auto x_main_module_456 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[256,1,1]}"), x_main_module_98);
    auto x_main_module_457 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_455, x_main_module_456);
    auto x_main_module_458 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,56,56]}"), x_main_module_457);
    auto x_main_module_459 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_453, x_main_module_458);
    auto x_main_module_460 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,56,56]}"), x_main_module_448);
    auto x_main_module_461 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_459, x_main_module_460);
    auto x_main_module_462 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,56,56]}"), x_main_module_449);
    auto x_main_module_463 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_461, x_main_module_462);
    auto x_main_module_464 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_446, x_main_module_463);
    auto x_main_module_465 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_464);
    auto x_main_module_466 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_465,
        x_main_module_349);
    auto x_main_module_467 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_348);
    auto x_main_module_468 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_347);
    auto x_main_module_469 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_346);
    auto x_main_module_470 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_345);
    auto x_main_module_471 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,56,56]}"), x_main_module_469);
    auto x_main_module_472 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_466, x_main_module_471);
    auto x_main_module_473 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_95);
    auto x_main_module_474 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_470, x_main_module_473);
    auto x_main_module_475 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_96);
    auto x_main_module_476 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_474, x_main_module_475);
    auto x_main_module_477 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,56,56]}"), x_main_module_476);
    auto x_main_module_478 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_472, x_main_module_477);
    auto x_main_module_479 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,56,56]}"), x_main_module_467);
    auto x_main_module_480 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_478, x_main_module_479);
    auto x_main_module_481 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,56,56]}"), x_main_module_468);
    auto x_main_module_482 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_480, x_main_module_481);
    auto x_main_module_483 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_482);
    auto x_main_module_484 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}"),
        x_main_module_483,
        x_main_module_344);
    auto x_main_module_485 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_343);
    auto x_main_module_486 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_342);
    auto x_main_module_487 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_341);
    auto x_main_module_488 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_340);
    auto x_main_module_489 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,56,56]}"), x_main_module_487);
    auto x_main_module_490 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_484, x_main_module_489);
    auto x_main_module_491 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_93);
    auto x_main_module_492 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_488, x_main_module_491);
    auto x_main_module_493 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_94);
    auto x_main_module_494 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_492, x_main_module_493);
    auto x_main_module_495 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,56,56]}"), x_main_module_494);
    auto x_main_module_496 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_490, x_main_module_495);
    auto x_main_module_497 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,56,56]}"), x_main_module_485);
    auto x_main_module_498 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_496, x_main_module_497);
    auto x_main_module_499 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,56,56]}"), x_main_module_486);
    auto x_main_module_500 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_498, x_main_module_499);
    auto x_main_module_501 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_500);
    auto x_main_module_502 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_501,
        x_main_module_339);
    auto x_main_module_503 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_338);
    auto x_main_module_504 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_337);
    auto x_main_module_505 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_336);
    auto x_main_module_506 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_335);
    auto x_main_module_507 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,56,56]}"), x_main_module_505);
    auto x_main_module_508 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_502, x_main_module_507);
    auto x_main_module_509 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[256,1,1]}"), x_main_module_91);
    auto x_main_module_510 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_506, x_main_module_509);
    auto x_main_module_511 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[256,1,1]}"), x_main_module_92);
    auto x_main_module_512 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_510, x_main_module_511);
    auto x_main_module_513 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,56,56]}"), x_main_module_512);
    auto x_main_module_514 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_508, x_main_module_513);
    auto x_main_module_515 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,56,56]}"), x_main_module_503);
    auto x_main_module_516 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_514, x_main_module_515);
    auto x_main_module_517 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,56,56]}"), x_main_module_504);
    auto x_main_module_518 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_516, x_main_module_517);
    auto x_main_module_519 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_518, x_main_module_465);
    auto x_main_module_520 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_519);
    auto x_main_module_521 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_520,
        x_main_module_334);
    auto x_main_module_522 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_333);
    auto x_main_module_523 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_332);
    auto x_main_module_524 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_331);
    auto x_main_module_525 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_330);
    auto x_main_module_526 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,56,56]}"), x_main_module_524);
    auto x_main_module_527 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_521, x_main_module_526);
    auto x_main_module_528 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_89);
    auto x_main_module_529 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_525, x_main_module_528);
    auto x_main_module_530 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_90);
    auto x_main_module_531 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_529, x_main_module_530);
    auto x_main_module_532 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,56,56]}"), x_main_module_531);
    auto x_main_module_533 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_527, x_main_module_532);
    auto x_main_module_534 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,56,56]}"), x_main_module_522);
    auto x_main_module_535 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_533, x_main_module_534);
    auto x_main_module_536 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,56,56]}"), x_main_module_523);
    auto x_main_module_537 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_535, x_main_module_536);
    auto x_main_module_538 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_537);
    auto x_main_module_539 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}"),
        x_main_module_538,
        x_main_module_329);
    auto x_main_module_540 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_328);
    auto x_main_module_541 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_327);
    auto x_main_module_542 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_326);
    auto x_main_module_543 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_325);
    auto x_main_module_544 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,56,56]}"), x_main_module_542);
    auto x_main_module_545 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_539, x_main_module_544);
    auto x_main_module_546 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_87);
    auto x_main_module_547 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_543, x_main_module_546);
    auto x_main_module_548 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_88);
    auto x_main_module_549 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_547, x_main_module_548);
    auto x_main_module_550 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,56,56]}"), x_main_module_549);
    auto x_main_module_551 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_545, x_main_module_550);
    auto x_main_module_552 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,56,56]}"), x_main_module_540);
    auto x_main_module_553 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_551, x_main_module_552);
    auto x_main_module_554 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,56,56]}"), x_main_module_541);
    auto x_main_module_555 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_553, x_main_module_554);
    auto x_main_module_556 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_555);
    auto x_main_module_557 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_556,
        x_main_module_324);
    auto x_main_module_558 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_323);
    auto x_main_module_559 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_322);
    auto x_main_module_560 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_321);
    auto x_main_module_561 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_320);
    auto x_main_module_562 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,56,56]}"), x_main_module_560);
    auto x_main_module_563 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_557, x_main_module_562);
    auto x_main_module_564 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[256,1,1]}"), x_main_module_85);
    auto x_main_module_565 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_561, x_main_module_564);
    auto x_main_module_566 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[256,1,1]}"), x_main_module_86);
    auto x_main_module_567 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_565, x_main_module_566);
    auto x_main_module_568 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,56,56]}"), x_main_module_567);
    auto x_main_module_569 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_563, x_main_module_568);
    auto x_main_module_570 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,56,56]}"), x_main_module_558);
    auto x_main_module_571 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_569, x_main_module_570);
    auto x_main_module_572 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,56,56]}"), x_main_module_559);
    auto x_main_module_573 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_571, x_main_module_572);
    auto x_main_module_574 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_573, x_main_module_520);
    auto x_main_module_575 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_574);
    auto x_main_module_576 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_575,
        x_main_module_319);
    auto x_main_module_577 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_318);
    auto x_main_module_578 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_317);
    auto x_main_module_579 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_316);
    auto x_main_module_580 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_315);
    auto x_main_module_581 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,56,56]}"), x_main_module_579);
    auto x_main_module_582 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_576, x_main_module_581);
    auto x_main_module_583 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[128,1,1]}"), x_main_module_83);
    auto x_main_module_584 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_580, x_main_module_583);
    auto x_main_module_585 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[128,1,1]}"), x_main_module_84);
    auto x_main_module_586 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_584, x_main_module_585);
    auto x_main_module_587 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,56,56]}"), x_main_module_586);
    auto x_main_module_588 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_582, x_main_module_587);
    auto x_main_module_589 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,56,56]}"), x_main_module_577);
    auto x_main_module_590 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_588, x_main_module_589);
    auto x_main_module_591 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,56,56]}"), x_main_module_578);
    auto x_main_module_592 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_590, x_main_module_591);
    auto x_main_module_593 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_592);
    auto x_main_module_594 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[2,2]}"),
        x_main_module_593,
        x_main_module_314);
    auto x_main_module_595 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_313);
    auto x_main_module_596 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_312);
    auto x_main_module_597 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_311);
    auto x_main_module_598 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_310);
    auto x_main_module_599 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,28,28]}"), x_main_module_597);
    auto x_main_module_600 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_594, x_main_module_599);
    auto x_main_module_601 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[128,1,1]}"), x_main_module_81);
    auto x_main_module_602 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_598, x_main_module_601);
    auto x_main_module_603 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[128,1,1]}"), x_main_module_82);
    auto x_main_module_604 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_602, x_main_module_603);
    auto x_main_module_605 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,28,28]}"), x_main_module_604);
    auto x_main_module_606 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_600, x_main_module_605);
    auto x_main_module_607 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,28,28]}"), x_main_module_595);
    auto x_main_module_608 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_606, x_main_module_607);
    auto x_main_module_609 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,28,28]}"), x_main_module_596);
    auto x_main_module_610 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_608, x_main_module_609);
    auto x_main_module_611 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_610);
    auto x_main_module_612 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_611,
        x_main_module_309);
    auto x_main_module_613 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_308);
    auto x_main_module_614 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_307);
    auto x_main_module_615 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_306);
    auto x_main_module_616 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_305);
    auto x_main_module_617 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,28,28]}"), x_main_module_615);
    auto x_main_module_618 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_612, x_main_module_617);
    auto x_main_module_619 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[512,1,1]}"), x_main_module_79);
    auto x_main_module_620 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_616, x_main_module_619);
    auto x_main_module_621 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[512,1,1]}"), x_main_module_80);
    auto x_main_module_622 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_620, x_main_module_621);
    auto x_main_module_623 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,28,28]}"), x_main_module_622);
    auto x_main_module_624 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_618, x_main_module_623);
    auto x_main_module_625 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,28,28]}"), x_main_module_613);
    auto x_main_module_626 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_624, x_main_module_625);
    auto x_main_module_627 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,28,28]}"), x_main_module_614);
    auto x_main_module_628 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_626, x_main_module_627);
    auto x_main_module_629 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[2,2]}"),
        x_main_module_575,
        x_main_module_304);
    auto x_main_module_630 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_303);
    auto x_main_module_631 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_302);
    auto x_main_module_632 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_301);
    auto x_main_module_633 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_300);
    auto x_main_module_634 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,28,28]}"), x_main_module_632);
    auto x_main_module_635 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_629, x_main_module_634);
    auto x_main_module_636 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[512,1,1]}"), x_main_module_77);
    auto x_main_module_637 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_633, x_main_module_636);
    auto x_main_module_638 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[512,1,1]}"), x_main_module_78);
    auto x_main_module_639 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_637, x_main_module_638);
    auto x_main_module_640 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,28,28]}"), x_main_module_639);
    auto x_main_module_641 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_635, x_main_module_640);
    auto x_main_module_642 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,28,28]}"), x_main_module_630);
    auto x_main_module_643 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_641, x_main_module_642);
    auto x_main_module_644 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,28,28]}"), x_main_module_631);
    auto x_main_module_645 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_643, x_main_module_644);
    auto x_main_module_646 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_628, x_main_module_645);
    auto x_main_module_647 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_646);
    auto x_main_module_648 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_647,
        x_main_module_299);
    auto x_main_module_649 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_298);
    auto x_main_module_650 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_297);
    auto x_main_module_651 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_296);
    auto x_main_module_652 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_295);
    auto x_main_module_653 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,28,28]}"), x_main_module_651);
    auto x_main_module_654 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_648, x_main_module_653);
    auto x_main_module_655 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[128,1,1]}"), x_main_module_75);
    auto x_main_module_656 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_652, x_main_module_655);
    auto x_main_module_657 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[128,1,1]}"), x_main_module_76);
    auto x_main_module_658 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_656, x_main_module_657);
    auto x_main_module_659 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,28,28]}"), x_main_module_658);
    auto x_main_module_660 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_654, x_main_module_659);
    auto x_main_module_661 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,28,28]}"), x_main_module_649);
    auto x_main_module_662 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_660, x_main_module_661);
    auto x_main_module_663 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,28,28]}"), x_main_module_650);
    auto x_main_module_664 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_662, x_main_module_663);
    auto x_main_module_665 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_664);
    auto x_main_module_666 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}"),
        x_main_module_665,
        x_main_module_294);
    auto x_main_module_667 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_293);
    auto x_main_module_668 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_292);
    auto x_main_module_669 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_291);
    auto x_main_module_670 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_290);
    auto x_main_module_671 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,28,28]}"), x_main_module_669);
    auto x_main_module_672 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_666, x_main_module_671);
    auto x_main_module_673 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[128,1,1]}"), x_main_module_73);
    auto x_main_module_674 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_670, x_main_module_673);
    auto x_main_module_675 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[128,1,1]}"), x_main_module_74);
    auto x_main_module_676 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_674, x_main_module_675);
    auto x_main_module_677 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,28,28]}"), x_main_module_676);
    auto x_main_module_678 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_672, x_main_module_677);
    auto x_main_module_679 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,28,28]}"), x_main_module_667);
    auto x_main_module_680 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_678, x_main_module_679);
    auto x_main_module_681 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,28,28]}"), x_main_module_668);
    auto x_main_module_682 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_680, x_main_module_681);
    auto x_main_module_683 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_682);
    auto x_main_module_684 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_683,
        x_main_module_289);
    auto x_main_module_685 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_288);
    auto x_main_module_686 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_287);
    auto x_main_module_687 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_286);
    auto x_main_module_688 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_285);
    auto x_main_module_689 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,28,28]}"), x_main_module_687);
    auto x_main_module_690 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_684, x_main_module_689);
    auto x_main_module_691 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[512,1,1]}"), x_main_module_71);
    auto x_main_module_692 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_688, x_main_module_691);
    auto x_main_module_693 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[512,1,1]}"), x_main_module_72);
    auto x_main_module_694 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_692, x_main_module_693);
    auto x_main_module_695 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,28,28]}"), x_main_module_694);
    auto x_main_module_696 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_690, x_main_module_695);
    auto x_main_module_697 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,28,28]}"), x_main_module_685);
    auto x_main_module_698 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_696, x_main_module_697);
    auto x_main_module_699 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,28,28]}"), x_main_module_686);
    auto x_main_module_700 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_698, x_main_module_699);
    auto x_main_module_701 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_700, x_main_module_647);
    auto x_main_module_702 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_701);
    auto x_main_module_703 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_702,
        x_main_module_284);
    auto x_main_module_704 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_283);
    auto x_main_module_705 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_282);
    auto x_main_module_706 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_281);
    auto x_main_module_707 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_280);
    auto x_main_module_708 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,28,28]}"), x_main_module_706);
    auto x_main_module_709 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_703, x_main_module_708);
    auto x_main_module_710 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[128,1,1]}"), x_main_module_69);
    auto x_main_module_711 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_707, x_main_module_710);
    auto x_main_module_712 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[128,1,1]}"), x_main_module_70);
    auto x_main_module_713 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_711, x_main_module_712);
    auto x_main_module_714 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,28,28]}"), x_main_module_713);
    auto x_main_module_715 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_709, x_main_module_714);
    auto x_main_module_716 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,28,28]}"), x_main_module_704);
    auto x_main_module_717 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_715, x_main_module_716);
    auto x_main_module_718 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,28,28]}"), x_main_module_705);
    auto x_main_module_719 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_717, x_main_module_718);
    auto x_main_module_720 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_719);
    auto x_main_module_721 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}"),
        x_main_module_720,
        x_main_module_279);
    auto x_main_module_722 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_278);
    auto x_main_module_723 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_277);
    auto x_main_module_724 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_276);
    auto x_main_module_725 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_275);
    auto x_main_module_726 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,28,28]}"), x_main_module_724);
    auto x_main_module_727 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_721, x_main_module_726);
    auto x_main_module_728 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[128,1,1]}"), x_main_module_67);
    auto x_main_module_729 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_725, x_main_module_728);
    auto x_main_module_730 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[128,1,1]}"), x_main_module_68);
    auto x_main_module_731 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_729, x_main_module_730);
    auto x_main_module_732 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,28,28]}"), x_main_module_731);
    auto x_main_module_733 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_727, x_main_module_732);
    auto x_main_module_734 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,28,28]}"), x_main_module_722);
    auto x_main_module_735 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_733, x_main_module_734);
    auto x_main_module_736 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,28,28]}"), x_main_module_723);
    auto x_main_module_737 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_735, x_main_module_736);
    auto x_main_module_738 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_737);
    auto x_main_module_739 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_738,
        x_main_module_274);
    auto x_main_module_740 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_273);
    auto x_main_module_741 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_272);
    auto x_main_module_742 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_271);
    auto x_main_module_743 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_270);
    auto x_main_module_744 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,28,28]}"), x_main_module_742);
    auto x_main_module_745 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_739, x_main_module_744);
    auto x_main_module_746 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[512,1,1]}"), x_main_module_65);
    auto x_main_module_747 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_743, x_main_module_746);
    auto x_main_module_748 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[512,1,1]}"), x_main_module_66);
    auto x_main_module_749 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_747, x_main_module_748);
    auto x_main_module_750 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,28,28]}"), x_main_module_749);
    auto x_main_module_751 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_745, x_main_module_750);
    auto x_main_module_752 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,28,28]}"), x_main_module_740);
    auto x_main_module_753 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_751, x_main_module_752);
    auto x_main_module_754 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,28,28]}"), x_main_module_741);
    auto x_main_module_755 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_753, x_main_module_754);
    auto x_main_module_756 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_755, x_main_module_702);
    auto x_main_module_757 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_756);
    auto x_main_module_758 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_757,
        x_main_module_269);
    auto x_main_module_759 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_268);
    auto x_main_module_760 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_267);
    auto x_main_module_761 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_266);
    auto x_main_module_762 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_265);
    auto x_main_module_763 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,28,28]}"), x_main_module_761);
    auto x_main_module_764 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_758, x_main_module_763);
    auto x_main_module_765 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[128,1,1]}"), x_main_module_63);
    auto x_main_module_766 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_762, x_main_module_765);
    auto x_main_module_767 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[128,1,1]}"), x_main_module_64);
    auto x_main_module_768 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_766, x_main_module_767);
    auto x_main_module_769 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,28,28]}"), x_main_module_768);
    auto x_main_module_770 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_764, x_main_module_769);
    auto x_main_module_771 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,28,28]}"), x_main_module_759);
    auto x_main_module_772 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_770, x_main_module_771);
    auto x_main_module_773 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,28,28]}"), x_main_module_760);
    auto x_main_module_774 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_772, x_main_module_773);
    auto x_main_module_775 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_774);
    auto x_main_module_776 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}"),
        x_main_module_775,
        x_main_module_264);
    auto x_main_module_777 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_263);
    auto x_main_module_778 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_262);
    auto x_main_module_779 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_261);
    auto x_main_module_780 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_260);
    auto x_main_module_781 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,28,28]}"), x_main_module_779);
    auto x_main_module_782 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_776, x_main_module_781);
    auto x_main_module_783 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[128,1,1]}"), x_main_module_61);
    auto x_main_module_784 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_780, x_main_module_783);
    auto x_main_module_785 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[128,1,1]}"), x_main_module_62);
    auto x_main_module_786 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_784, x_main_module_785);
    auto x_main_module_787 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,28,28]}"), x_main_module_786);
    auto x_main_module_788 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_782, x_main_module_787);
    auto x_main_module_789 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,28,28]}"), x_main_module_777);
    auto x_main_module_790 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_788, x_main_module_789);
    auto x_main_module_791 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,28,28]}"), x_main_module_778);
    auto x_main_module_792 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_790, x_main_module_791);
    auto x_main_module_793 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_792);
    auto x_main_module_794 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_793,
        x_main_module_259);
    auto x_main_module_795 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_258);
    auto x_main_module_796 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_257);
    auto x_main_module_797 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_256);
    auto x_main_module_798 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_255);
    auto x_main_module_799 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,28,28]}"), x_main_module_797);
    auto x_main_module_800 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_794, x_main_module_799);
    auto x_main_module_801 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[512,1,1]}"), x_main_module_59);
    auto x_main_module_802 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_798, x_main_module_801);
    auto x_main_module_803 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[512,1,1]}"), x_main_module_60);
    auto x_main_module_804 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_802, x_main_module_803);
    auto x_main_module_805 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,28,28]}"), x_main_module_804);
    auto x_main_module_806 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_800, x_main_module_805);
    auto x_main_module_807 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,28,28]}"), x_main_module_795);
    auto x_main_module_808 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_806, x_main_module_807);
    auto x_main_module_809 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,28,28]}"), x_main_module_796);
    auto x_main_module_810 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_808, x_main_module_809);
    auto x_main_module_811 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_810, x_main_module_757);
    auto x_main_module_812 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_811);
    auto x_main_module_813 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_812,
        x_main_module_254);
    auto x_main_module_814 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_253);
    auto x_main_module_815 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_252);
    auto x_main_module_816 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_251);
    auto x_main_module_817 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_250);
    auto x_main_module_818 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,28,28]}"), x_main_module_816);
    auto x_main_module_819 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_813, x_main_module_818);
    auto x_main_module_820 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[256,1,1]}"), x_main_module_57);
    auto x_main_module_821 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_817, x_main_module_820);
    auto x_main_module_822 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[256,1,1]}"), x_main_module_58);
    auto x_main_module_823 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_821, x_main_module_822);
    auto x_main_module_824 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,28,28]}"), x_main_module_823);
    auto x_main_module_825 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_819, x_main_module_824);
    auto x_main_module_826 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,28,28]}"), x_main_module_814);
    auto x_main_module_827 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_825, x_main_module_826);
    auto x_main_module_828 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,28,28]}"), x_main_module_815);
    auto x_main_module_829 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_827, x_main_module_828);
    auto x_main_module_830 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_829);
    auto x_main_module_831 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[2,2]}"),
        x_main_module_830,
        x_main_module_249);
    auto x_main_module_832 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_248);
    auto x_main_module_833 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_247);
    auto x_main_module_834 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_246);
    auto x_main_module_835 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_245);
    auto x_main_module_836 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_834);
    auto x_main_module_837 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_831, x_main_module_836);
    auto x_main_module_838 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[256,1,1]}"), x_main_module_55);
    auto x_main_module_839 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_835, x_main_module_838);
    auto x_main_module_840 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[256,1,1]}"), x_main_module_56);
    auto x_main_module_841 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_839, x_main_module_840);
    auto x_main_module_842 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_841);
    auto x_main_module_843 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_837, x_main_module_842);
    auto x_main_module_844 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_832);
    auto x_main_module_845 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_843, x_main_module_844);
    auto x_main_module_846 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_833);
    auto x_main_module_847 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_845, x_main_module_846);
    auto x_main_module_848 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_847);
    auto x_main_module_849 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_848,
        x_main_module_244);
    auto x_main_module_850 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_243);
    auto x_main_module_851 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_242);
    auto x_main_module_852 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_241);
    auto x_main_module_853 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_240);
    auto x_main_module_854 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1024,14,14]}"), x_main_module_852);
    auto x_main_module_855 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_849, x_main_module_854);
    auto x_main_module_856 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1024,1,1]}"), x_main_module_53);
    auto x_main_module_857 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_853, x_main_module_856);
    auto x_main_module_858 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1024,1,1]}"), x_main_module_54);
    auto x_main_module_859 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_857, x_main_module_858);
    auto x_main_module_860 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1024,14,14]}"), x_main_module_859);
    auto x_main_module_861 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_855, x_main_module_860);
    auto x_main_module_862 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1024,14,14]}"), x_main_module_850);
    auto x_main_module_863 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_861, x_main_module_862);
    auto x_main_module_864 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1024,14,14]}"), x_main_module_851);
    auto x_main_module_865 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_863, x_main_module_864);
    auto x_main_module_866 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[2,2]}"),
        x_main_module_812,
        x_main_module_239);
    auto x_main_module_867 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_238);
    auto x_main_module_868 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_237);
    auto x_main_module_869 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_236);
    auto x_main_module_870 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_235);
    auto x_main_module_871 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1024,14,14]}"), x_main_module_869);
    auto x_main_module_872 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_866, x_main_module_871);
    auto x_main_module_873 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1024,1,1]}"), x_main_module_51);
    auto x_main_module_874 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_870, x_main_module_873);
    auto x_main_module_875 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1024,1,1]}"), x_main_module_52);
    auto x_main_module_876 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_874, x_main_module_875);
    auto x_main_module_877 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1024,14,14]}"), x_main_module_876);
    auto x_main_module_878 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_872, x_main_module_877);
    auto x_main_module_879 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1024,14,14]}"), x_main_module_867);
    auto x_main_module_880 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_878, x_main_module_879);
    auto x_main_module_881 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1024,14,14]}"), x_main_module_868);
    auto x_main_module_882 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_880, x_main_module_881);
    auto x_main_module_883 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_865, x_main_module_882);
    auto x_main_module_884 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_883);
    auto x_main_module_885 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_884,
        x_main_module_234);
    auto x_main_module_886 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_233);
    auto x_main_module_887 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_232);
    auto x_main_module_888 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_231);
    auto x_main_module_889 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_230);
    auto x_main_module_890 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_888);
    auto x_main_module_891 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_885, x_main_module_890);
    auto x_main_module_892 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[256,1,1]}"), x_main_module_49);
    auto x_main_module_893 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_889, x_main_module_892);
    auto x_main_module_894 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[256,1,1]}"), x_main_module_50);
    auto x_main_module_895 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_893, x_main_module_894);
    auto x_main_module_896 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_895);
    auto x_main_module_897 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_891, x_main_module_896);
    auto x_main_module_898 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_886);
    auto x_main_module_899 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_897, x_main_module_898);
    auto x_main_module_900 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_887);
    auto x_main_module_901 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_899, x_main_module_900);
    auto x_main_module_902 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_901);
    auto x_main_module_903 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}"),
        x_main_module_902,
        x_main_module_229);
    auto x_main_module_904 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_228);
    auto x_main_module_905 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_227);
    auto x_main_module_906 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_226);
    auto x_main_module_907 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_225);
    auto x_main_module_908 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_906);
    auto x_main_module_909 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_903, x_main_module_908);
    auto x_main_module_910 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[256,1,1]}"), x_main_module_47);
    auto x_main_module_911 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_907, x_main_module_910);
    auto x_main_module_912 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[256,1,1]}"), x_main_module_48);
    auto x_main_module_913 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_911, x_main_module_912);
    auto x_main_module_914 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_913);
    auto x_main_module_915 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_909, x_main_module_914);
    auto x_main_module_916 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_904);
    auto x_main_module_917 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_915, x_main_module_916);
    auto x_main_module_918 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_905);
    auto x_main_module_919 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_917, x_main_module_918);
    auto x_main_module_920 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_919);
    auto x_main_module_921 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_920,
        x_main_module_224);
    auto x_main_module_922 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_223);
    auto x_main_module_923 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_222);
    auto x_main_module_924 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_221);
    auto x_main_module_925 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_220);
    auto x_main_module_926 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1024,14,14]}"), x_main_module_924);
    auto x_main_module_927 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_921, x_main_module_926);
    auto x_main_module_928 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1024,1,1]}"), x_main_module_45);
    auto x_main_module_929 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_925, x_main_module_928);
    auto x_main_module_930 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1024,1,1]}"), x_main_module_46);
    auto x_main_module_931 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_929, x_main_module_930);
    auto x_main_module_932 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1024,14,14]}"), x_main_module_931);
    auto x_main_module_933 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_927, x_main_module_932);
    auto x_main_module_934 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1024,14,14]}"), x_main_module_922);
    auto x_main_module_935 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_933, x_main_module_934);
    auto x_main_module_936 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1024,14,14]}"), x_main_module_923);
    auto x_main_module_937 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_935, x_main_module_936);
    auto x_main_module_938 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_937, x_main_module_884);
    auto x_main_module_939 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_938);
    auto x_main_module_940 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_939,
        x_main_module_219);
    auto x_main_module_941 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_218);
    auto x_main_module_942 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_217);
    auto x_main_module_943 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_216);
    auto x_main_module_944 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_215);
    auto x_main_module_945 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_943);
    auto x_main_module_946 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_940, x_main_module_945);
    auto x_main_module_947 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[256,1,1]}"), x_main_module_43);
    auto x_main_module_948 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_944, x_main_module_947);
    auto x_main_module_949 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[256,1,1]}"), x_main_module_44);
    auto x_main_module_950 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_948, x_main_module_949);
    auto x_main_module_951 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_950);
    auto x_main_module_952 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_946, x_main_module_951);
    auto x_main_module_953 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_941);
    auto x_main_module_954 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_952, x_main_module_953);
    auto x_main_module_955 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_942);
    auto x_main_module_956 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_954, x_main_module_955);
    auto x_main_module_957 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_956);
    auto x_main_module_958 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}"),
        x_main_module_957,
        x_main_module_214);
    auto x_main_module_959 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_213);
    auto x_main_module_960 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_212);
    auto x_main_module_961 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_211);
    auto x_main_module_962 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_210);
    auto x_main_module_963 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_961);
    auto x_main_module_964 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_958, x_main_module_963);
    auto x_main_module_965 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[256,1,1]}"), x_main_module_41);
    auto x_main_module_966 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_962, x_main_module_965);
    auto x_main_module_967 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[256,1,1]}"), x_main_module_42);
    auto x_main_module_968 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_966, x_main_module_967);
    auto x_main_module_969 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_968);
    auto x_main_module_970 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_964, x_main_module_969);
    auto x_main_module_971 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_959);
    auto x_main_module_972 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_970, x_main_module_971);
    auto x_main_module_973 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_960);
    auto x_main_module_974 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_972, x_main_module_973);
    auto x_main_module_975 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_974);
    auto x_main_module_976 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_975,
        x_main_module_209);
    auto x_main_module_977 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_208);
    auto x_main_module_978 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_207);
    auto x_main_module_979 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_206);
    auto x_main_module_980 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_205);
    auto x_main_module_981 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1024,14,14]}"), x_main_module_979);
    auto x_main_module_982 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_976, x_main_module_981);
    auto x_main_module_983 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1024,1,1]}"), x_main_module_39);
    auto x_main_module_984 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_980, x_main_module_983);
    auto x_main_module_985 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1024,1,1]}"), x_main_module_40);
    auto x_main_module_986 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_984, x_main_module_985);
    auto x_main_module_987 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1024,14,14]}"), x_main_module_986);
    auto x_main_module_988 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_982, x_main_module_987);
    auto x_main_module_989 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1024,14,14]}"), x_main_module_977);
    auto x_main_module_990 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_988, x_main_module_989);
    auto x_main_module_991 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1024,14,14]}"), x_main_module_978);
    auto x_main_module_992 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_990, x_main_module_991);
    auto x_main_module_993 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_992, x_main_module_939);
    auto x_main_module_994 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_993);
    auto x_main_module_995 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_994,
        x_main_module_204);
    auto x_main_module_996 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_203);
    auto x_main_module_997 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_202);
    auto x_main_module_998 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_201);
    auto x_main_module_999 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_200);
    auto x_main_module_1000 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_998);
    auto x_main_module_1001 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_995, x_main_module_1000);
    auto x_main_module_1002 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[256,1,1]}"), x_main_module_37);
    auto x_main_module_1003 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_999, x_main_module_1002);
    auto x_main_module_1004 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[256,1,1]}"), x_main_module_38);
    auto x_main_module_1005 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1003, x_main_module_1004);
    auto x_main_module_1006 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_1005);
    auto x_main_module_1007 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1001, x_main_module_1006);
    auto x_main_module_1008 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_996);
    auto x_main_module_1009 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1007, x_main_module_1008);
    auto x_main_module_1010 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_997);
    auto x_main_module_1011 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1009, x_main_module_1010);
    auto x_main_module_1012 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1011);
    auto x_main_module_1013 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}"),
        x_main_module_1012,
        x_main_module_199);
    auto x_main_module_1014 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_198);
    auto x_main_module_1015 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_197);
    auto x_main_module_1016 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_196);
    auto x_main_module_1017 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_195);
    auto x_main_module_1018 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_1016);
    auto x_main_module_1019 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1013, x_main_module_1018);
    auto x_main_module_1020 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[256,1,1]}"), x_main_module_35);
    auto x_main_module_1021 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1017, x_main_module_1020);
    auto x_main_module_1022 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[256,1,1]}"), x_main_module_36);
    auto x_main_module_1023 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1021, x_main_module_1022);
    auto x_main_module_1024 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_1023);
    auto x_main_module_1025 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1019, x_main_module_1024);
    auto x_main_module_1026 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_1014);
    auto x_main_module_1027 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1025, x_main_module_1026);
    auto x_main_module_1028 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_1015);
    auto x_main_module_1029 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1027, x_main_module_1028);
    auto x_main_module_1030 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1029);
    auto x_main_module_1031 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1030,
        x_main_module_194);
    auto x_main_module_1032 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_193);
    auto x_main_module_1033 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_192);
    auto x_main_module_1034 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_191);
    auto x_main_module_1035 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_190);
    auto x_main_module_1036 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1024,14,14]}"), x_main_module_1034);
    auto x_main_module_1037 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1031, x_main_module_1036);
    auto x_main_module_1038 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1024,1,1]}"), x_main_module_33);
    auto x_main_module_1039 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1035, x_main_module_1038);
    auto x_main_module_1040 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1024,1,1]}"), x_main_module_34);
    auto x_main_module_1041 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1039, x_main_module_1040);
    auto x_main_module_1042 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1024,14,14]}"), x_main_module_1041);
    auto x_main_module_1043 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1037, x_main_module_1042);
    auto x_main_module_1044 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1024,14,14]}"), x_main_module_1032);
    auto x_main_module_1045 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1043, x_main_module_1044);
    auto x_main_module_1046 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1024,14,14]}"), x_main_module_1033);
    auto x_main_module_1047 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1045, x_main_module_1046);
    auto x_main_module_1048 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1047, x_main_module_994);
    auto x_main_module_1049 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1048);
    auto x_main_module_1050 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1049,
        x_main_module_189);
    auto x_main_module_1051 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_188);
    auto x_main_module_1052 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_187);
    auto x_main_module_1053 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_186);
    auto x_main_module_1054 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_185);
    auto x_main_module_1055 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_1053);
    auto x_main_module_1056 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1050, x_main_module_1055);
    auto x_main_module_1057 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[256,1,1]}"), x_main_module_31);
    auto x_main_module_1058 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1054, x_main_module_1057);
    auto x_main_module_1059 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[256,1,1]}"), x_main_module_32);
    auto x_main_module_1060 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1058, x_main_module_1059);
    auto x_main_module_1061 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_1060);
    auto x_main_module_1062 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1056, x_main_module_1061);
    auto x_main_module_1063 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_1051);
    auto x_main_module_1064 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1062, x_main_module_1063);
    auto x_main_module_1065 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_1052);
    auto x_main_module_1066 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1064, x_main_module_1065);
    auto x_main_module_1067 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1066);
    auto x_main_module_1068 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}"),
        x_main_module_1067,
        x_main_module_184);
    auto x_main_module_1069 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_183);
    auto x_main_module_1070 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_182);
    auto x_main_module_1071 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_181);
    auto x_main_module_1072 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_180);
    auto x_main_module_1073 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_1071);
    auto x_main_module_1074 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1068, x_main_module_1073);
    auto x_main_module_1075 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[256,1,1]}"), x_main_module_29);
    auto x_main_module_1076 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1072, x_main_module_1075);
    auto x_main_module_1077 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[256,1,1]}"), x_main_module_30);
    auto x_main_module_1078 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1076, x_main_module_1077);
    auto x_main_module_1079 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_1078);
    auto x_main_module_1080 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1074, x_main_module_1079);
    auto x_main_module_1081 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_1069);
    auto x_main_module_1082 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1080, x_main_module_1081);
    auto x_main_module_1083 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_1070);
    auto x_main_module_1084 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1082, x_main_module_1083);
    auto x_main_module_1085 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1084);
    auto x_main_module_1086 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1085,
        x_main_module_179);
    auto x_main_module_1087 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_178);
    auto x_main_module_1088 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_177);
    auto x_main_module_1089 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_176);
    auto x_main_module_1090 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_175);
    auto x_main_module_1091 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1024,14,14]}"), x_main_module_1089);
    auto x_main_module_1092 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1086, x_main_module_1091);
    auto x_main_module_1093 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1024,1,1]}"), x_main_module_27);
    auto x_main_module_1094 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1090, x_main_module_1093);
    auto x_main_module_1095 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1024,1,1]}"), x_main_module_28);
    auto x_main_module_1096 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1094, x_main_module_1095);
    auto x_main_module_1097 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1024,14,14]}"), x_main_module_1096);
    auto x_main_module_1098 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1092, x_main_module_1097);
    auto x_main_module_1099 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1024,14,14]}"), x_main_module_1087);
    auto x_main_module_1100 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1098, x_main_module_1099);
    auto x_main_module_1101 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1024,14,14]}"), x_main_module_1088);
    auto x_main_module_1102 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1100, x_main_module_1101);
    auto x_main_module_1103 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1102, x_main_module_1049);
    auto x_main_module_1104 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1103);
    auto x_main_module_1105 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1104,
        x_main_module_174);
    auto x_main_module_1106 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_173);
    auto x_main_module_1107 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_172);
    auto x_main_module_1108 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_171);
    auto x_main_module_1109 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_170);
    auto x_main_module_1110 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_1108);
    auto x_main_module_1111 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1105, x_main_module_1110);
    auto x_main_module_1112 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[256,1,1]}"), x_main_module_25);
    auto x_main_module_1113 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1109, x_main_module_1112);
    auto x_main_module_1114 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[256,1,1]}"), x_main_module_26);
    auto x_main_module_1115 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1113, x_main_module_1114);
    auto x_main_module_1116 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_1115);
    auto x_main_module_1117 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1111, x_main_module_1116);
    auto x_main_module_1118 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_1106);
    auto x_main_module_1119 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1117, x_main_module_1118);
    auto x_main_module_1120 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_1107);
    auto x_main_module_1121 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1119, x_main_module_1120);
    auto x_main_module_1122 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1121);
    auto x_main_module_1123 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}"),
        x_main_module_1122,
        x_main_module_169);
    auto x_main_module_1124 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_168);
    auto x_main_module_1125 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_167);
    auto x_main_module_1126 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_166);
    auto x_main_module_1127 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_165);
    auto x_main_module_1128 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_1126);
    auto x_main_module_1129 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1123, x_main_module_1128);
    auto x_main_module_1130 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[256,1,1]}"), x_main_module_23);
    auto x_main_module_1131 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1127, x_main_module_1130);
    auto x_main_module_1132 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[256,1,1]}"), x_main_module_24);
    auto x_main_module_1133 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1131, x_main_module_1132);
    auto x_main_module_1134 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_1133);
    auto x_main_module_1135 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1129, x_main_module_1134);
    auto x_main_module_1136 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_1124);
    auto x_main_module_1137 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1135, x_main_module_1136);
    auto x_main_module_1138 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,256,14,14]}"), x_main_module_1125);
    auto x_main_module_1139 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1137, x_main_module_1138);
    auto x_main_module_1140 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1139);
    auto x_main_module_1141 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1140,
        x_main_module_164);
    auto x_main_module_1142 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_163);
    auto x_main_module_1143 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_162);
    auto x_main_module_1144 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_161);
    auto x_main_module_1145 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_160);
    auto x_main_module_1146 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1024,14,14]}"), x_main_module_1144);
    auto x_main_module_1147 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1141, x_main_module_1146);
    auto x_main_module_1148 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1024,1,1]}"), x_main_module_21);
    auto x_main_module_1149 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1145, x_main_module_1148);
    auto x_main_module_1150 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1024,1,1]}"), x_main_module_22);
    auto x_main_module_1151 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1149, x_main_module_1150);
    auto x_main_module_1152 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1024,14,14]}"), x_main_module_1151);
    auto x_main_module_1153 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1147, x_main_module_1152);
    auto x_main_module_1154 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1024,14,14]}"), x_main_module_1142);
    auto x_main_module_1155 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1153, x_main_module_1154);
    auto x_main_module_1156 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1024,14,14]}"), x_main_module_1143);
    auto x_main_module_1157 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1155, x_main_module_1156);
    auto x_main_module_1158 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1157, x_main_module_1104);
    auto x_main_module_1159 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1158);
    auto x_main_module_1160 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1159,
        x_main_module_159);
    auto x_main_module_1161 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_158);
    auto x_main_module_1162 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_157);
    auto x_main_module_1163 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_156);
    auto x_main_module_1164 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_155);
    auto x_main_module_1165 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,14,14]}"), x_main_module_1163);
    auto x_main_module_1166 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1160, x_main_module_1165);
    auto x_main_module_1167 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[512,1,1]}"), x_main_module_19);
    auto x_main_module_1168 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1164, x_main_module_1167);
    auto x_main_module_1169 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[512,1,1]}"), x_main_module_20);
    auto x_main_module_1170 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1168, x_main_module_1169);
    auto x_main_module_1171 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,14,14]}"), x_main_module_1170);
    auto x_main_module_1172 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1166, x_main_module_1171);
    auto x_main_module_1173 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,14,14]}"), x_main_module_1161);
    auto x_main_module_1174 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1172, x_main_module_1173);
    auto x_main_module_1175 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,14,14]}"), x_main_module_1162);
    auto x_main_module_1176 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1174, x_main_module_1175);
    auto x_main_module_1177 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1176);
    auto x_main_module_1178 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[2,2]}"),
        x_main_module_1177,
        x_main_module_154);
    auto x_main_module_1179 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_153);
    auto x_main_module_1180 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_152);
    auto x_main_module_1181 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_151);
    auto x_main_module_1182 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_150);
    auto x_main_module_1183 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,7,7]}"), x_main_module_1181);
    auto x_main_module_1184 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1178, x_main_module_1183);
    auto x_main_module_1185 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[512,1,1]}"), x_main_module_17);
    auto x_main_module_1186 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1182, x_main_module_1185);
    auto x_main_module_1187 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[512,1,1]}"), x_main_module_18);
    auto x_main_module_1188 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1186, x_main_module_1187);
    auto x_main_module_1189 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,7,7]}"), x_main_module_1188);
    auto x_main_module_1190 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1184, x_main_module_1189);
    auto x_main_module_1191 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,7,7]}"), x_main_module_1179);
    auto x_main_module_1192 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1190, x_main_module_1191);
    auto x_main_module_1193 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,7,7]}"), x_main_module_1180);
    auto x_main_module_1194 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1192, x_main_module_1193);
    auto x_main_module_1195 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1194);
    auto x_main_module_1196 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1195,
        x_main_module_149);
    auto x_main_module_1197 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_148);
    auto x_main_module_1198 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_147);
    auto x_main_module_1199 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_146);
    auto x_main_module_1200 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_145);
    auto x_main_module_1201 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,2048,7,7]}"), x_main_module_1199);
    auto x_main_module_1202 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1196, x_main_module_1201);
    auto x_main_module_1203 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[2048,1,1]}"), x_main_module_15);
    auto x_main_module_1204 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1200, x_main_module_1203);
    auto x_main_module_1205 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[2048,1,1]}"), x_main_module_16);
    auto x_main_module_1206 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1204, x_main_module_1205);
    auto x_main_module_1207 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,2048,7,7]}"), x_main_module_1206);
    auto x_main_module_1208 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1202, x_main_module_1207);
    auto x_main_module_1209 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,2048,7,7]}"), x_main_module_1197);
    auto x_main_module_1210 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1208, x_main_module_1209);
    auto x_main_module_1211 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,2048,7,7]}"), x_main_module_1198);
    auto x_main_module_1212 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1210, x_main_module_1211);
    auto x_main_module_1213 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[2,2]}"),
        x_main_module_1159,
        x_main_module_144);
    auto x_main_module_1214 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_143);
    auto x_main_module_1215 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_142);
    auto x_main_module_1216 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_141);
    auto x_main_module_1217 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_140);
    auto x_main_module_1218 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,2048,7,7]}"), x_main_module_1216);
    auto x_main_module_1219 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1213, x_main_module_1218);
    auto x_main_module_1220 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[2048,1,1]}"), x_main_module_13);
    auto x_main_module_1221 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1217, x_main_module_1220);
    auto x_main_module_1222 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[2048,1,1]}"), x_main_module_14);
    auto x_main_module_1223 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1221, x_main_module_1222);
    auto x_main_module_1224 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,2048,7,7]}"), x_main_module_1223);
    auto x_main_module_1225 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1219, x_main_module_1224);
    auto x_main_module_1226 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,2048,7,7]}"), x_main_module_1214);
    auto x_main_module_1227 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1225, x_main_module_1226);
    auto x_main_module_1228 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,2048,7,7]}"), x_main_module_1215);
    auto x_main_module_1229 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1227, x_main_module_1228);
    auto x_main_module_1230 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1212, x_main_module_1229);
    auto x_main_module_1231 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1230);
    auto x_main_module_1232 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1231,
        x_main_module_139);
    auto x_main_module_1233 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_138);
    auto x_main_module_1234 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_137);
    auto x_main_module_1235 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_136);
    auto x_main_module_1236 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_135);
    auto x_main_module_1237 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,7,7]}"), x_main_module_1235);
    auto x_main_module_1238 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1232, x_main_module_1237);
    auto x_main_module_1239 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[512,1,1]}"), x_main_module_11);
    auto x_main_module_1240 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1236, x_main_module_1239);
    auto x_main_module_1241 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[512,1,1]}"), x_main_module_12);
    auto x_main_module_1242 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1240, x_main_module_1241);
    auto x_main_module_1243 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,7,7]}"), x_main_module_1242);
    auto x_main_module_1244 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1238, x_main_module_1243);
    auto x_main_module_1245 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,7,7]}"), x_main_module_1233);
    auto x_main_module_1246 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1244, x_main_module_1245);
    auto x_main_module_1247 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,7,7]}"), x_main_module_1234);
    auto x_main_module_1248 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1246, x_main_module_1247);
    auto x_main_module_1249 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1248);
    auto x_main_module_1250 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}"),
        x_main_module_1249,
        x_main_module_134);
    auto x_main_module_1251 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_133);
    auto x_main_module_1252 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_132);
    auto x_main_module_1253 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_131);
    auto x_main_module_1254 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_130);
    auto x_main_module_1255 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,7,7]}"), x_main_module_1253);
    auto x_main_module_1256 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1250, x_main_module_1255);
    auto x_main_module_1257 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[512,1,1]}"), x_main_module_9);
    auto x_main_module_1258 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1254, x_main_module_1257);
    auto x_main_module_1259 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[512,1,1]}"), x_main_module_10);
    auto x_main_module_1260 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1258, x_main_module_1259);
    auto x_main_module_1261 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,7,7]}"), x_main_module_1260);
    auto x_main_module_1262 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1256, x_main_module_1261);
    auto x_main_module_1263 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,7,7]}"), x_main_module_1251);
    auto x_main_module_1264 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1262, x_main_module_1263);
    auto x_main_module_1265 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,7,7]}"), x_main_module_1252);
    auto x_main_module_1266 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1264, x_main_module_1265);
    auto x_main_module_1267 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1266);
    auto x_main_module_1268 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1267,
        x_main_module_129);
    auto x_main_module_1269 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_128);
    auto x_main_module_1270 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_127);
    auto x_main_module_1271 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_126);
    auto x_main_module_1272 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_125);
    auto x_main_module_1273 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,2048,7,7]}"), x_main_module_1271);
    auto x_main_module_1274 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1268, x_main_module_1273);
    auto x_main_module_1275 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[2048,1,1]}"), x_main_module_7);
    auto x_main_module_1276 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1272, x_main_module_1275);
    auto x_main_module_1277 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[2048,1,1]}"), x_main_module_8);
    auto x_main_module_1278 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1276, x_main_module_1277);
    auto x_main_module_1279 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,2048,7,7]}"), x_main_module_1278);
    auto x_main_module_1280 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1274, x_main_module_1279);
    auto x_main_module_1281 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,2048,7,7]}"), x_main_module_1269);
    auto x_main_module_1282 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1280, x_main_module_1281);
    auto x_main_module_1283 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,2048,7,7]}"), x_main_module_1270);
    auto x_main_module_1284 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1282, x_main_module_1283);
    auto x_main_module_1285 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1284, x_main_module_1231);
    auto x_main_module_1286 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1285);
    auto x_main_module_1287 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1286,
        x_main_module_124);
    auto x_main_module_1288 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_123);
    auto x_main_module_1289 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_122);
    auto x_main_module_1290 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_121);
    auto x_main_module_1291 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_120);
    auto x_main_module_1292 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,7,7]}"), x_main_module_1290);
    auto x_main_module_1293 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1287, x_main_module_1292);
    auto x_main_module_1294 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[512,1,1]}"), x_main_module_5);
    auto x_main_module_1295 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1291, x_main_module_1294);
    auto x_main_module_1296 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[512,1,1]}"), x_main_module_6);
    auto x_main_module_1297 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1295, x_main_module_1296);
    auto x_main_module_1298 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,7,7]}"), x_main_module_1297);
    auto x_main_module_1299 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1293, x_main_module_1298);
    auto x_main_module_1300 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,7,7]}"), x_main_module_1288);
    auto x_main_module_1301 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1299, x_main_module_1300);
    auto x_main_module_1302 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,7,7]}"), x_main_module_1289);
    auto x_main_module_1303 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1301, x_main_module_1302);
    auto x_main_module_1304 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1303);
    auto x_main_module_1305 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}"),
        x_main_module_1304,
        x_main_module_119);
    auto x_main_module_1306 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_118);
    auto x_main_module_1307 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_117);
    auto x_main_module_1308 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_116);
    auto x_main_module_1309 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_115);
    auto x_main_module_1310 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,7,7]}"), x_main_module_1308);
    auto x_main_module_1311 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1305, x_main_module_1310);
    auto x_main_module_1312 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[512,1,1]}"), x_main_module_3);
    auto x_main_module_1313 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1309, x_main_module_1312);
    auto x_main_module_1314 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[512,1,1]}"), x_main_module_4);
    auto x_main_module_1315 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1313, x_main_module_1314);
    auto x_main_module_1316 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,7,7]}"), x_main_module_1315);
    auto x_main_module_1317 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1311, x_main_module_1316);
    auto x_main_module_1318 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,7,7]}"), x_main_module_1306);
    auto x_main_module_1319 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1317, x_main_module_1318);
    auto x_main_module_1320 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,512,7,7]}"), x_main_module_1307);
    auto x_main_module_1321 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1319, x_main_module_1320);
    auto x_main_module_1322 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1321);
    auto x_main_module_1323 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1322,
        x_main_module_114);
    auto x_main_module_1324 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_113);
    auto x_main_module_1325 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_112);
    auto x_main_module_1326 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_111);
    auto x_main_module_1327 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_110);
    auto x_main_module_1328 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,2048,7,7]}"), x_main_module_1326);
    auto x_main_module_1329 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1323, x_main_module_1328);
    auto x_main_module_1330 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[2048,1,1]}"), x_main_module_1);
    auto x_main_module_1331 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1327, x_main_module_1330);
    auto x_main_module_1332 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[2048,1,1]}"), x_main_module_2);
    auto x_main_module_1333 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1331, x_main_module_1332);
    auto x_main_module_1334 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,2048,7,7]}"), x_main_module_1333);
    auto x_main_module_1335 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1329, x_main_module_1334);
    auto x_main_module_1336 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,2048,7,7]}"), x_main_module_1324);
    auto x_main_module_1337 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1335, x_main_module_1336);
    auto x_main_module_1338 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,2048,7,7]}"), x_main_module_1325);
    auto x_main_module_1339 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1337, x_main_module_1338);
    auto x_main_module_1340 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1339, x_main_module_1286);
    auto x_main_module_1341 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1340);
    auto x_main_module_1342 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[7,7],lp_order:2,mode:0,padding:[0,0,0,0],stride:[1,1]}"),
        x_main_module_1341);
    auto x_main_module_1343 =
        mmain->add_instruction(migraphx::make_json_op("flatten", "{axis:1}"), x_main_module_1342);
    auto x_main_module_1344 = mmain->add_instruction(
        migraphx::make_json_op("transpose", "{permutation:[1,0]}"), x_main_module_109);
    auto x_main_module_1345 =
        mmain->add_instruction(migraphx::make_op("dot"), x_main_module_1343, x_main_module_1344);
    auto x_main_module_1346 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1000]}"), x_main_module_108);
    auto x_main_module_1347 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1000]}"), x_main_module_0);
    auto x_main_module_1348 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1346, x_main_module_1347);
    auto x_main_module_1349 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1345, x_main_module_1348);
    mmain->add_return({x_main_module_1349});

    return p;
}
} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx
