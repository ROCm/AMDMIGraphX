
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
    auto x_main_module_107     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 107)));
    auto x_main_module_108     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 108)));
    auto x_main_module_109     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 109)));
    auto x_main_module_110     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 110)));
    auto x_main_module_111     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 111)));
    auto x_main_module_112     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 112)));
    auto x_main_module_113     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 113)));
    auto x_main_module_114     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 114)));
    auto x_main_module_115     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 115)));
    auto x_main_module_116     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 116)));
    auto x_main_module_117     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 117)));
    auto x_main_module_118     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 118)));
    auto x_main_module_119     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 119)));
    auto x_main_module_120     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 120)));
    auto x_main_module_121     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 121)));
    auto x_main_module_122     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 122)));
    auto x_main_module_123     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 123)));
    auto x_main_module_124     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 124)));
    auto x_main_module_125     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 125)));
    auto x_main_module_126     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 126)));
    auto x_main_module_127     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 127)));
    auto x_main_module_128     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 128)));
    auto x_main_module_129     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 129)));
    auto x_main_module_130     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 130)));
    auto x_main_module_131     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 131)));
    auto x_main_module_132     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 132)));
    auto x_main_module_133     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 133)));
    auto x_main_module_134     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 134)));
    auto x_main_module_135     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 135)));
    auto x_main_module_136     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 136)));
    auto x_main_module_137     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 137)));
    auto x_main_module_138     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 138)));
    auto x_main_module_139     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 139)));
    auto x_main_module_140     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 140)));
    auto x_main_module_141     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 141)));
    auto x_main_module_142     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 142)));
    auto x_main_module_143     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 143)));
    auto x_main_module_144     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 144)));
    auto x_main_module_145     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 145)));
    auto x_main_module_146     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 146)));
    auto x_main_module_147     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 147)));
    auto x_main_module_148     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 148)));
    auto x_main_module_149     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 149)));
    auto x_main_module_150     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 150)));
    auto x_main_module_151     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 151)));
    auto x_main_module_152     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 152)));
    auto x_main_module_153     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 153)));
    auto x_main_module_154     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 154)));
    auto x_main_module_155     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 155)));
    auto x_main_module_156     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 156)));
    auto x_main_module_157     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 157)));
    auto x_main_module_158     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 158)));
    auto x_main_module_159     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 159)));
    auto x_main_module_160     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 160)));
    auto x_main_module_161     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 161)));
    auto x_main_module_162     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 162)));
    auto x_main_module_163     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 163)));
    auto x_main_module_164     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 164)));
    auto x_main_module_165     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 165)));
    auto x_main_module_166     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 166)));
    auto x_main_module_167     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 167)));
    auto x_main_module_168     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 168)));
    auto x_main_module_169     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 169)));
    auto x_main_module_170     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 170)));
    auto x_main_module_171     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 171)));
    auto x_main_module_172     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 172)));
    auto x_main_module_173     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 173)));
    auto x_main_module_174     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 174)));
    auto x_main_module_175     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 175)));
    auto x_main_module_176     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 176)));
    auto x_main_module_177     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 177)));
    auto x_main_module_178     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 178)));
    auto x_main_module_179     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 179)));
    auto x_main_module_180     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 180)));
    auto x_main_module_181     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 181)));
    auto x_main_module_182     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 182)));
    auto x_main_module_183     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 183)));
    auto x_main_module_184     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 184)));
    auto x_main_module_185     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 185)));
    auto x_main_module_186     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 186)));
    auto x_main_module_187     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 187)));
    auto x_main_module_188     = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 188)));
    auto x_0                   = mmain->add_parameter(
        "0", migraphx::shape{migraphx::shape::float_type, {batch, 3, 299, 299}});
    auto x_main_module_190 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1000}}, 189));
    auto x_main_module_191 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1000, 2048}}, 190));
    auto x_main_module_192 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 191)));
    auto x_main_module_193 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 192));
    auto x_main_module_194 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 193));
    auto x_main_module_195 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 194)));
    auto x_main_module_196 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 2048, 1, 1}}, 195));
    auto x_main_module_197 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 196)));
    auto x_main_module_198 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 197));
    auto x_main_module_199 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 198));
    auto x_main_module_200 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 199)));
    auto x_main_module_201 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 3, 1}}, 200));
    auto x_main_module_202 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 201)));
    auto x_main_module_203 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 202));
    auto x_main_module_204 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 203));
    auto x_main_module_205 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 204)));
    auto x_main_module_206 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 1, 3}}, 205));
    auto x_main_module_207 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 206)));
    auto x_main_module_208 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 207));
    auto x_main_module_209 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 208));
    auto x_main_module_210 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 209)));
    auto x_main_module_211 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 448, 3, 3}}, 210));
    auto x_main_module_212 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 211)));
    auto x_main_module_213 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 212));
    auto x_main_module_214 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 213));
    auto x_main_module_215 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 214)));
    auto x_main_module_216 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {448, 2048, 1, 1}}, 215));
    auto x_main_module_217 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 216)));
    auto x_main_module_218 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 217)));
    auto x_main_module_219 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 218));
    auto x_main_module_220 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 219)));
    auto x_main_module_221 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 3, 1}}, 220));
    auto x_main_module_222 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 221)));
    auto x_main_module_223 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 222)));
    auto x_main_module_224 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 223));
    auto x_main_module_225 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 224)));
    auto x_main_module_226 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 1, 3}}, 225));
    auto x_main_module_227 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 226)));
    auto x_main_module_228 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 227));
    auto x_main_module_229 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 228));
    auto x_main_module_230 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 229)));
    auto x_main_module_231 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 2048, 1, 1}}, 230));
    auto x_main_module_232 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 231)));
    auto x_main_module_233 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 232));
    auto x_main_module_234 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 233));
    auto x_main_module_235 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 234)));
    auto x_main_module_236 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {320, 2048, 1, 1}}, 235));
    auto x_main_module_237 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 236)));
    auto x_main_module_238 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 237));
    auto x_main_module_239 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 238));
    auto x_main_module_240 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 239)));
    auto x_main_module_241 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 1280, 1, 1}}, 240));
    auto x_main_module_242 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 241)));
    auto x_main_module_243 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 242));
    auto x_main_module_244 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 243));
    auto x_main_module_245 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 244)));
    auto x_main_module_246 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 3, 1}}, 245));
    auto x_main_module_247 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 246)));
    auto x_main_module_248 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 247));
    auto x_main_module_249 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 248));
    auto x_main_module_250 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 249)));
    auto x_main_module_251 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 1, 3}}, 250));
    auto x_main_module_252 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 251)));
    auto x_main_module_253 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 252));
    auto x_main_module_254 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 253));
    auto x_main_module_255 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 254)));
    auto x_main_module_256 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 448, 3, 3}}, 255));
    auto x_main_module_257 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 256)));
    auto x_main_module_258 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 257));
    auto x_main_module_259 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 258));
    auto x_main_module_260 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 259)));
    auto x_main_module_261 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {448, 1280, 1, 1}}, 260));
    auto x_main_module_262 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 261)));
    auto x_main_module_263 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 262));
    auto x_main_module_264 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 263));
    auto x_main_module_265 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 264)));
    auto x_main_module_266 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 3, 1}}, 265));
    auto x_main_module_267 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 266)));
    auto x_main_module_268 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 267));
    auto x_main_module_269 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 268));
    auto x_main_module_270 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 269)));
    auto x_main_module_271 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 1, 3}}, 270));
    auto x_main_module_272 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 271)));
    auto x_main_module_273 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 272));
    auto x_main_module_274 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 273));
    auto x_main_module_275 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 274)));
    auto x_main_module_276 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 1280, 1, 1}}, 275));
    auto x_main_module_277 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 276)));
    auto x_main_module_278 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 277));
    auto x_main_module_279 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 278));
    auto x_main_module_280 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 279)));
    auto x_main_module_281 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {320, 1280, 1, 1}}, 280));
    auto x_main_module_282 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 281)));
    auto x_main_module_283 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 282));
    auto x_main_module_284 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 283));
    auto x_main_module_285 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 284)));
    auto x_main_module_286 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 3, 3}}, 285));
    auto x_main_module_287 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 286)));
    auto x_main_module_288 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 287));
    auto x_main_module_289 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 288));
    auto x_main_module_290 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 289)));
    auto x_main_module_291 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 7, 1}}, 290));
    auto x_main_module_292 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 291)));
    auto x_main_module_293 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 292));
    auto x_main_module_294 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 293));
    auto x_main_module_295 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 294)));
    auto x_main_module_296 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 1, 7}}, 295));
    auto x_main_module_297 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 296)));
    auto x_main_module_298 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 297));
    auto x_main_module_299 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 298));
    auto x_main_module_300 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 299)));
    auto x_main_module_301 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 300));
    auto x_main_module_302 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 301)));
    auto x_main_module_303 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 302));
    auto x_main_module_304 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 303));
    auto x_main_module_305 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 304)));
    auto x_main_module_306 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {320, 192, 3, 3}}, 305));
    auto x_main_module_307 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 306)));
    auto x_main_module_308 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 307));
    auto x_main_module_309 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 308));
    auto x_main_module_310 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 309)));
    auto x_main_module_311 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 310));
    auto x_main_module_312 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 311)));
    auto x_main_module_313 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 312));
    auto x_main_module_314 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 313));
    auto x_main_module_315 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 314)));
    auto x_main_module_316 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 315));
    auto x_main_module_317 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 316)));
    auto x_main_module_318 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 317));
    auto x_main_module_319 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 318));
    auto x_main_module_320 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 319)));
    auto x_main_module_321 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 1, 7}}, 320));
    auto x_main_module_322 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 321)));
    auto x_main_module_323 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 322));
    auto x_main_module_324 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 323));
    auto x_main_module_325 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 324)));
    auto x_main_module_326 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 7, 1}}, 325));
    auto x_main_module_327 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 326)));
    auto x_main_module_328 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 327));
    auto x_main_module_329 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 328));
    auto x_main_module_330 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 329)));
    auto x_main_module_331 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 1, 7}}, 330));
    auto x_main_module_332 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 331)));
    auto x_main_module_333 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 332));
    auto x_main_module_334 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 333));
    auto x_main_module_335 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 334)));
    auto x_main_module_336 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 7, 1}}, 335));
    auto x_main_module_337 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 336)));
    auto x_main_module_338 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 337));
    auto x_main_module_339 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 338));
    auto x_main_module_340 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 339)));
    auto x_main_module_341 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 340));
    auto x_main_module_342 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 341)));
    auto x_main_module_343 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 342));
    auto x_main_module_344 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 343));
    auto x_main_module_345 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 344)));
    auto x_main_module_346 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 7, 1}}, 345));
    auto x_main_module_347 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 346)));
    auto x_main_module_348 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 347));
    auto x_main_module_349 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 348));
    auto x_main_module_350 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 349)));
    auto x_main_module_351 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 1, 7}}, 350));
    auto x_main_module_352 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 351)));
    auto x_main_module_353 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 352));
    auto x_main_module_354 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 353));
    auto x_main_module_355 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 354)));
    auto x_main_module_356 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 355));
    auto x_main_module_357 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 356)));
    auto x_main_module_358 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 357));
    auto x_main_module_359 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 358));
    auto x_main_module_360 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 359)));
    auto x_main_module_361 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 360));
    auto x_main_module_362 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 361)));
    auto x_main_module_363 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 362));
    auto x_main_module_364 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 363));
    auto x_main_module_365 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 364)));
    auto x_main_module_366 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 365));
    auto x_main_module_367 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 366)));
    auto x_main_module_368 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 367));
    auto x_main_module_369 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 368));
    auto x_main_module_370 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 369)));
    auto x_main_module_371 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 160, 1, 7}}, 370));
    auto x_main_module_372 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 371)));
    auto x_main_module_373 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 372));
    auto x_main_module_374 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 373));
    auto x_main_module_375 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 374)));
    auto x_main_module_376 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 7, 1}}, 375));
    auto x_main_module_377 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 376)));
    auto x_main_module_378 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 377));
    auto x_main_module_379 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 378));
    auto x_main_module_380 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 379)));
    auto x_main_module_381 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 1, 7}}, 380));
    auto x_main_module_382 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 381)));
    auto x_main_module_383 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 382));
    auto x_main_module_384 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 383));
    auto x_main_module_385 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 384)));
    auto x_main_module_386 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 7, 1}}, 385));
    auto x_main_module_387 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 386)));
    auto x_main_module_388 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 387));
    auto x_main_module_389 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 388));
    auto x_main_module_390 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 389)));
    auto x_main_module_391 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 768, 1, 1}}, 390));
    auto x_main_module_392 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 391)));
    auto x_main_module_393 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 392));
    auto x_main_module_394 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 393));
    auto x_main_module_395 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 394)));
    auto x_main_module_396 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 160, 7, 1}}, 395));
    auto x_main_module_397 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 396)));
    auto x_main_module_398 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 397));
    auto x_main_module_399 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 398));
    auto x_main_module_400 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 399)));
    auto x_main_module_401 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 1, 7}}, 400));
    auto x_main_module_402 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 401)));
    auto x_main_module_403 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 402));
    auto x_main_module_404 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 403));
    auto x_main_module_405 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 404)));
    auto x_main_module_406 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 768, 1, 1}}, 405));
    auto x_main_module_407 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 406)));
    auto x_main_module_408 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 407));
    auto x_main_module_409 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 408));
    auto x_main_module_410 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 409)));
    auto x_main_module_411 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 410));
    auto x_main_module_412 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 411)));
    auto x_main_module_413 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 412));
    auto x_main_module_414 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 413));
    auto x_main_module_415 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 414)));
    auto x_main_module_416 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 415));
    auto x_main_module_417 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 416)));
    auto x_main_module_418 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 417));
    auto x_main_module_419 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 418));
    auto x_main_module_420 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 419)));
    auto x_main_module_421 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 160, 1, 7}}, 420));
    auto x_main_module_422 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 421)));
    auto x_main_module_423 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 422));
    auto x_main_module_424 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 423));
    auto x_main_module_425 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 424)));
    auto x_main_module_426 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 7, 1}}, 425));
    auto x_main_module_427 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 426)));
    auto x_main_module_428 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 427));
    auto x_main_module_429 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 428));
    auto x_main_module_430 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 429)));
    auto x_main_module_431 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 1, 7}}, 430));
    auto x_main_module_432 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 431)));
    auto x_main_module_433 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 432));
    auto x_main_module_434 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 433));
    auto x_main_module_435 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 434)));
    auto x_main_module_436 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 7, 1}}, 435));
    auto x_main_module_437 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 436)));
    auto x_main_module_438 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 437));
    auto x_main_module_439 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 438));
    auto x_main_module_440 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 439)));
    auto x_main_module_441 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 768, 1, 1}}, 440));
    auto x_main_module_442 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 441)));
    auto x_main_module_443 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 442));
    auto x_main_module_444 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 443));
    auto x_main_module_445 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 444)));
    auto x_main_module_446 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 160, 7, 1}}, 445));
    auto x_main_module_447 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 446)));
    auto x_main_module_448 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 447));
    auto x_main_module_449 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 448));
    auto x_main_module_450 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 449)));
    auto x_main_module_451 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 1, 7}}, 450));
    auto x_main_module_452 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 451)));
    auto x_main_module_453 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 452));
    auto x_main_module_454 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 453));
    auto x_main_module_455 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 454)));
    auto x_main_module_456 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 768, 1, 1}}, 455));
    auto x_main_module_457 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 456)));
    auto x_main_module_458 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 457));
    auto x_main_module_459 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 458));
    auto x_main_module_460 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 459)));
    auto x_main_module_461 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 460));
    auto x_main_module_462 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 461)));
    auto x_main_module_463 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 462));
    auto x_main_module_464 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 463));
    auto x_main_module_465 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 464)));
    auto x_main_module_466 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 465));
    auto x_main_module_467 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 466)));
    auto x_main_module_468 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 467));
    auto x_main_module_469 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 468));
    auto x_main_module_470 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 469)));
    auto x_main_module_471 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 128, 1, 7}}, 470));
    auto x_main_module_472 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 471)));
    auto x_main_module_473 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 472));
    auto x_main_module_474 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 473));
    auto x_main_module_475 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 474)));
    auto x_main_module_476 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 7, 1}}, 475));
    auto x_main_module_477 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 476)));
    auto x_main_module_478 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 477));
    auto x_main_module_479 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 478));
    auto x_main_module_480 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 479)));
    auto x_main_module_481 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 1, 7}}, 480));
    auto x_main_module_482 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 481)));
    auto x_main_module_483 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 482));
    auto x_main_module_484 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 483));
    auto x_main_module_485 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 484)));
    auto x_main_module_486 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 7, 1}}, 485));
    auto x_main_module_487 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 486)));
    auto x_main_module_488 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 487));
    auto x_main_module_489 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 488));
    auto x_main_module_490 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 489)));
    auto x_main_module_491 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 768, 1, 1}}, 490));
    auto x_main_module_492 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 491)));
    auto x_main_module_493 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 492));
    auto x_main_module_494 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 493));
    auto x_main_module_495 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 494)));
    auto x_main_module_496 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 128, 7, 1}}, 495));
    auto x_main_module_497 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 496)));
    auto x_main_module_498 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 497));
    auto x_main_module_499 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 498));
    auto x_main_module_500 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 499)));
    auto x_main_module_501 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 1, 7}}, 500));
    auto x_main_module_502 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 501)));
    auto x_main_module_503 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 502));
    auto x_main_module_504 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 503));
    auto x_main_module_505 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 504)));
    auto x_main_module_506 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 768, 1, 1}}, 505));
    auto x_main_module_507 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 506)));
    auto x_main_module_508 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 507));
    auto x_main_module_509 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 508));
    auto x_main_module_510 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 509)));
    auto x_main_module_511 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 510));
    auto x_main_module_512 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 511)));
    auto x_main_module_513 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 512));
    auto x_main_module_514 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 513));
    auto x_main_module_515 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 514)));
    auto x_main_module_516 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 96, 3, 3}}, 515));
    auto x_main_module_517 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 516)));
    auto x_main_module_518 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 517));
    auto x_main_module_519 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 518));
    auto x_main_module_520 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 519)));
    auto x_main_module_521 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 64, 3, 3}}, 520));
    auto x_main_module_522 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 521)));
    auto x_main_module_523 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 522));
    auto x_main_module_524 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 523));
    auto x_main_module_525 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 524)));
    auto x_main_module_526 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 288, 1, 1}}, 525));
    auto x_main_module_527 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 526)));
    auto x_main_module_528 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 527));
    auto x_main_module_529 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 528));
    auto x_main_module_530 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 529)));
    auto x_main_module_531 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 288, 3, 3}}, 530));
    auto x_main_module_532 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 531)));
    auto x_main_module_533 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 532));
    auto x_main_module_534 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 533));
    auto x_main_module_535 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 534)));
    auto x_main_module_536 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 288, 1, 1}}, 535));
    auto x_main_module_537 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 536)));
    auto x_main_module_538 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 537));
    auto x_main_module_539 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 538));
    auto x_main_module_540 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 539)));
    auto x_main_module_541 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 96, 3, 3}}, 540));
    auto x_main_module_542 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 541)));
    auto x_main_module_543 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 542));
    auto x_main_module_544 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 543));
    auto x_main_module_545 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 544)));
    auto x_main_module_546 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 64, 3, 3}}, 545));
    auto x_main_module_547 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 546)));
    auto x_main_module_548 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 547));
    auto x_main_module_549 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 548));
    auto x_main_module_550 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 549)));
    auto x_main_module_551 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 288, 1, 1}}, 550));
    auto x_main_module_552 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 551)));
    auto x_main_module_553 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 552));
    auto x_main_module_554 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 553));
    auto x_main_module_555 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 554)));
    auto x_main_module_556 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 48, 5, 5}}, 555));
    auto x_main_module_557 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 556)));
    auto x_main_module_558 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 557));
    auto x_main_module_559 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 558));
    auto x_main_module_560 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 559)));
    auto x_main_module_561 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {48, 288, 1, 1}}, 560));
    auto x_main_module_562 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 561)));
    auto x_main_module_563 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 562));
    auto x_main_module_564 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 563));
    auto x_main_module_565 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 564)));
    auto x_main_module_566 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 288, 1, 1}}, 565));
    auto x_main_module_567 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 566)));
    auto x_main_module_568 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 567));
    auto x_main_module_569 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 568));
    auto x_main_module_570 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 569)));
    auto x_main_module_571 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 256, 1, 1}}, 570));
    auto x_main_module_572 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 571)));
    auto x_main_module_573 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 572));
    auto x_main_module_574 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 573));
    auto x_main_module_575 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 574)));
    auto x_main_module_576 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 96, 3, 3}}, 575));
    auto x_main_module_577 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 576)));
    auto x_main_module_578 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 577));
    auto x_main_module_579 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 578));
    auto x_main_module_580 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 579)));
    auto x_main_module_581 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 64, 3, 3}}, 580));
    auto x_main_module_582 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 581)));
    auto x_main_module_583 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 582));
    auto x_main_module_584 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 583));
    auto x_main_module_585 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 584)));
    auto x_main_module_586 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 256, 1, 1}}, 585));
    auto x_main_module_587 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 586)));
    auto x_main_module_588 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 587));
    auto x_main_module_589 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 588));
    auto x_main_module_590 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 589)));
    auto x_main_module_591 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 48, 5, 5}}, 590));
    auto x_main_module_592 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 591)));
    auto x_main_module_593 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 592));
    auto x_main_module_594 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 593));
    auto x_main_module_595 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 594)));
    auto x_main_module_596 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {48, 256, 1, 1}}, 595));
    auto x_main_module_597 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 596)));
    auto x_main_module_598 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 597));
    auto x_main_module_599 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 598));
    auto x_main_module_600 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 599)));
    auto x_main_module_601 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 256, 1, 1}}, 600));
    auto x_main_module_602 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 601)));
    auto x_main_module_603 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 602));
    auto x_main_module_604 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 603));
    auto x_main_module_605 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 604)));
    auto x_main_module_606 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {32, 192, 1, 1}}, 605));
    auto x_main_module_607 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 606)));
    auto x_main_module_608 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 607));
    auto x_main_module_609 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 608));
    auto x_main_module_610 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 609)));
    auto x_main_module_611 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 96, 3, 3}}, 610));
    auto x_main_module_612 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 611)));
    auto x_main_module_613 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 612));
    auto x_main_module_614 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 613));
    auto x_main_module_615 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 614)));
    auto x_main_module_616 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 64, 3, 3}}, 615));
    auto x_main_module_617 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 616)));
    auto x_main_module_618 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 617));
    auto x_main_module_619 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 618));
    auto x_main_module_620 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 619)));
    auto x_main_module_621 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 192, 1, 1}}, 620));
    auto x_main_module_622 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 621)));
    auto x_main_module_623 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 622));
    auto x_main_module_624 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 623));
    auto x_main_module_625 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 624)));
    auto x_main_module_626 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 48, 5, 5}}, 625));
    auto x_main_module_627 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 626)));
    auto x_main_module_628 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 627));
    auto x_main_module_629 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 628));
    auto x_main_module_630 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 629)));
    auto x_main_module_631 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {48, 192, 1, 1}}, 630));
    auto x_main_module_632 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 631)));
    auto x_main_module_633 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 632));
    auto x_main_module_634 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 633));
    auto x_main_module_635 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 634)));
    auto x_main_module_636 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 192, 1, 1}}, 635));
    auto x_main_module_637 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 636)));
    auto x_main_module_638 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 637));
    auto x_main_module_639 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 638));
    auto x_main_module_640 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 639)));
    auto x_main_module_641 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 80, 3, 3}}, 640));
    auto x_main_module_642 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {80}}, 641)));
    auto x_main_module_643 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {80}}, 642));
    auto x_main_module_644 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {80}}, 643));
    auto x_main_module_645 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {80}}, 644)));
    auto x_main_module_646 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {80, 64, 1, 1}}, 645));
    auto x_main_module_647 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 646)));
    auto x_main_module_648 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 647));
    auto x_main_module_649 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 648));
    auto x_main_module_650 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 649)));
    auto x_main_module_651 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 32, 3, 3}}, 650));
    auto x_main_module_652 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 651)));
    auto x_main_module_653 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 652));
    auto x_main_module_654 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 653));
    auto x_main_module_655 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 654)));
    auto x_main_module_656 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {32, 32, 3, 3}}, 655));
    auto x_main_module_657 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 656)));
    auto x_main_module_658 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 657));
    auto x_main_module_659 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 658));
    auto x_main_module_660 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 659)));
    auto x_main_module_661 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {32, 3, 3, 3}}, 660));
    auto x_main_module_662 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[2,2]}"),
        x_0,
        x_main_module_661);
    auto x_main_module_663 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_660);
    auto x_main_module_664 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_659);
    auto x_main_module_665 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_658);
    auto x_main_module_666 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_657);
    auto x_main_module_667 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,32,149,149]}"), x_main_module_665);
    auto x_main_module_668 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_662, x_main_module_667);
    auto x_main_module_669 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[32,1,1]}"), x_main_module_187);
    auto x_main_module_670 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_666, x_main_module_669);
    auto x_main_module_671 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[32,1,1]}"), x_main_module_188);
    auto x_main_module_672 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_670, x_main_module_671);
    auto x_main_module_673 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,32,149,149]}"), x_main_module_672);
    auto x_main_module_674 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_668, x_main_module_673);
    auto x_main_module_675 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,32,149,149]}"), x_main_module_663);
    auto x_main_module_676 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_674, x_main_module_675);
    auto x_main_module_677 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,32,149,149]}"), x_main_module_664);
    auto x_main_module_678 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_676, x_main_module_677);
    auto x_main_module_679 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_678);
    auto x_main_module_680 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_679,
        x_main_module_656);
    auto x_main_module_681 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_655);
    auto x_main_module_682 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_654);
    auto x_main_module_683 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_653);
    auto x_main_module_684 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_652);
    auto x_main_module_685 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,32,147,147]}"), x_main_module_683);
    auto x_main_module_686 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_680, x_main_module_685);
    auto x_main_module_687 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[32,1,1]}"), x_main_module_185);
    auto x_main_module_688 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_684, x_main_module_687);
    auto x_main_module_689 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[32,1,1]}"), x_main_module_186);
    auto x_main_module_690 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_688, x_main_module_689);
    auto x_main_module_691 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,32,147,147]}"), x_main_module_690);
    auto x_main_module_692 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_686, x_main_module_691);
    auto x_main_module_693 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,32,147,147]}"), x_main_module_681);
    auto x_main_module_694 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_692, x_main_module_693);
    auto x_main_module_695 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,32,147,147]}"), x_main_module_682);
    auto x_main_module_696 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_694, x_main_module_695);
    auto x_main_module_697 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_696);
    auto x_main_module_698 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}"),
        x_main_module_697,
        x_main_module_651);
    auto x_main_module_699 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_650);
    auto x_main_module_700 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_649);
    auto x_main_module_701 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_648);
    auto x_main_module_702 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_647);
    auto x_main_module_703 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,147,147]}"), x_main_module_701);
    auto x_main_module_704 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_698, x_main_module_703);
    auto x_main_module_705 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_183);
    auto x_main_module_706 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_702, x_main_module_705);
    auto x_main_module_707 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_184);
    auto x_main_module_708 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_706, x_main_module_707);
    auto x_main_module_709 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,147,147]}"), x_main_module_708);
    auto x_main_module_710 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_704, x_main_module_709);
    auto x_main_module_711 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,147,147]}"), x_main_module_699);
    auto x_main_module_712 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_710, x_main_module_711);
    auto x_main_module_713 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,147,147]}"), x_main_module_700);
    auto x_main_module_714 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_712, x_main_module_713);
    auto x_main_module_715 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_714);
    auto x_main_module_716 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:1,padding:[0,0,0,0],stride:[2,2]}"),
        x_main_module_715);
    auto x_main_module_717 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_716,
        x_main_module_646);
    auto x_main_module_718 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_645);
    auto x_main_module_719 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_644);
    auto x_main_module_720 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_643);
    auto x_main_module_721 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_642);
    auto x_main_module_722 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,80,73,73]}"), x_main_module_720);
    auto x_main_module_723 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_717, x_main_module_722);
    auto x_main_module_724 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[80,1,1]}"), x_main_module_181);
    auto x_main_module_725 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_721, x_main_module_724);
    auto x_main_module_726 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[80,1,1]}"), x_main_module_182);
    auto x_main_module_727 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_725, x_main_module_726);
    auto x_main_module_728 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,80,73,73]}"), x_main_module_727);
    auto x_main_module_729 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_723, x_main_module_728);
    auto x_main_module_730 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,80,73,73]}"), x_main_module_718);
    auto x_main_module_731 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_729, x_main_module_730);
    auto x_main_module_732 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,80,73,73]}"), x_main_module_719);
    auto x_main_module_733 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_731, x_main_module_732);
    auto x_main_module_734 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_733);
    auto x_main_module_735 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_734,
        x_main_module_641);
    auto x_main_module_736 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_640);
    auto x_main_module_737 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_639);
    auto x_main_module_738 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_638);
    auto x_main_module_739 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_637);
    auto x_main_module_740 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,71,71]}"), x_main_module_738);
    auto x_main_module_741 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_735, x_main_module_740);
    auto x_main_module_742 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_179);
    auto x_main_module_743 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_739, x_main_module_742);
    auto x_main_module_744 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_180);
    auto x_main_module_745 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_743, x_main_module_744);
    auto x_main_module_746 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,71,71]}"), x_main_module_745);
    auto x_main_module_747 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_741, x_main_module_746);
    auto x_main_module_748 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,71,71]}"), x_main_module_736);
    auto x_main_module_749 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_747, x_main_module_748);
    auto x_main_module_750 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,71,71]}"), x_main_module_737);
    auto x_main_module_751 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_749, x_main_module_750);
    auto x_main_module_752 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_751);
    auto x_main_module_753 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:1,padding:[0,0,0,0],stride:[2,2]}"),
        x_main_module_752);
    auto x_main_module_754 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_753,
        x_main_module_636);
    auto x_main_module_755 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_635);
    auto x_main_module_756 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_634);
    auto x_main_module_757 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_633);
    auto x_main_module_758 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_632);
    auto x_main_module_759 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_757);
    auto x_main_module_760 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_754, x_main_module_759);
    auto x_main_module_761 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_177);
    auto x_main_module_762 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_758, x_main_module_761);
    auto x_main_module_763 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_178);
    auto x_main_module_764 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_762, x_main_module_763);
    auto x_main_module_765 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_764);
    auto x_main_module_766 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_760, x_main_module_765);
    auto x_main_module_767 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_755);
    auto x_main_module_768 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_766, x_main_module_767);
    auto x_main_module_769 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_756);
    auto x_main_module_770 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_768, x_main_module_769);
    auto x_main_module_771 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_770);
    auto x_main_module_772 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_753,
        x_main_module_631);
    auto x_main_module_773 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_630);
    auto x_main_module_774 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_629);
    auto x_main_module_775 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_628);
    auto x_main_module_776 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_627);
    auto x_main_module_777 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,48,35,35]}"), x_main_module_775);
    auto x_main_module_778 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_772, x_main_module_777);
    auto x_main_module_779 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[48,1,1]}"), x_main_module_175);
    auto x_main_module_780 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_776, x_main_module_779);
    auto x_main_module_781 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[48,1,1]}"), x_main_module_176);
    auto x_main_module_782 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_780, x_main_module_781);
    auto x_main_module_783 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,48,35,35]}"), x_main_module_782);
    auto x_main_module_784 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_778, x_main_module_783);
    auto x_main_module_785 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,48,35,35]}"), x_main_module_773);
    auto x_main_module_786 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_784, x_main_module_785);
    auto x_main_module_787 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,48,35,35]}"), x_main_module_774);
    auto x_main_module_788 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_786, x_main_module_787);
    auto x_main_module_789 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_788);
    auto x_main_module_790 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[2,2,2,2],padding_mode:0,stride:[1,1]}"),
        x_main_module_789,
        x_main_module_626);
    auto x_main_module_791 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_625);
    auto x_main_module_792 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_624);
    auto x_main_module_793 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_623);
    auto x_main_module_794 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_622);
    auto x_main_module_795 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_793);
    auto x_main_module_796 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_790, x_main_module_795);
    auto x_main_module_797 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_173);
    auto x_main_module_798 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_794, x_main_module_797);
    auto x_main_module_799 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_174);
    auto x_main_module_800 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_798, x_main_module_799);
    auto x_main_module_801 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_800);
    auto x_main_module_802 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_796, x_main_module_801);
    auto x_main_module_803 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_791);
    auto x_main_module_804 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_802, x_main_module_803);
    auto x_main_module_805 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_792);
    auto x_main_module_806 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_804, x_main_module_805);
    auto x_main_module_807 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_806);
    auto x_main_module_808 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_753,
        x_main_module_621);
    auto x_main_module_809 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_620);
    auto x_main_module_810 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_619);
    auto x_main_module_811 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_618);
    auto x_main_module_812 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_617);
    auto x_main_module_813 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_811);
    auto x_main_module_814 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_808, x_main_module_813);
    auto x_main_module_815 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_171);
    auto x_main_module_816 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_812, x_main_module_815);
    auto x_main_module_817 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_172);
    auto x_main_module_818 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_816, x_main_module_817);
    auto x_main_module_819 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_818);
    auto x_main_module_820 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_814, x_main_module_819);
    auto x_main_module_821 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_809);
    auto x_main_module_822 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_820, x_main_module_821);
    auto x_main_module_823 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_810);
    auto x_main_module_824 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_822, x_main_module_823);
    auto x_main_module_825 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_824);
    auto x_main_module_826 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}"),
        x_main_module_825,
        x_main_module_616);
    auto x_main_module_827 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_615);
    auto x_main_module_828 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_614);
    auto x_main_module_829 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_613);
    auto x_main_module_830 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_612);
    auto x_main_module_831 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,96,35,35]}"), x_main_module_829);
    auto x_main_module_832 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_826, x_main_module_831);
    auto x_main_module_833 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[96,1,1]}"), x_main_module_169);
    auto x_main_module_834 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_830, x_main_module_833);
    auto x_main_module_835 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[96,1,1]}"), x_main_module_170);
    auto x_main_module_836 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_834, x_main_module_835);
    auto x_main_module_837 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,96,35,35]}"), x_main_module_836);
    auto x_main_module_838 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_832, x_main_module_837);
    auto x_main_module_839 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,96,35,35]}"), x_main_module_827);
    auto x_main_module_840 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_838, x_main_module_839);
    auto x_main_module_841 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,96,35,35]}"), x_main_module_828);
    auto x_main_module_842 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_840, x_main_module_841);
    auto x_main_module_843 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_842);
    auto x_main_module_844 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}"),
        x_main_module_843,
        x_main_module_611);
    auto x_main_module_845 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_610);
    auto x_main_module_846 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_609);
    auto x_main_module_847 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_608);
    auto x_main_module_848 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_607);
    auto x_main_module_849 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,96,35,35]}"), x_main_module_847);
    auto x_main_module_850 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_844, x_main_module_849);
    auto x_main_module_851 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[96,1,1]}"), x_main_module_167);
    auto x_main_module_852 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_848, x_main_module_851);
    auto x_main_module_853 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[96,1,1]}"), x_main_module_168);
    auto x_main_module_854 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_852, x_main_module_853);
    auto x_main_module_855 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,96,35,35]}"), x_main_module_854);
    auto x_main_module_856 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_850, x_main_module_855);
    auto x_main_module_857 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,96,35,35]}"), x_main_module_845);
    auto x_main_module_858 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_856, x_main_module_857);
    auto x_main_module_859 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,96,35,35]}"), x_main_module_846);
    auto x_main_module_860 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_858, x_main_module_859);
    auto x_main_module_861 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_860);
    auto x_main_module_862 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:0,padding:[1,1,1,1],stride:[1,1]}"),
        x_main_module_753);
    auto x_main_module_863 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_862,
        x_main_module_606);
    auto x_main_module_864 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_605);
    auto x_main_module_865 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_604);
    auto x_main_module_866 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_603);
    auto x_main_module_867 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_602);
    auto x_main_module_868 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,32,35,35]}"), x_main_module_866);
    auto x_main_module_869 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_863, x_main_module_868);
    auto x_main_module_870 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[32,1,1]}"), x_main_module_165);
    auto x_main_module_871 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_867, x_main_module_870);
    auto x_main_module_872 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[32,1,1]}"), x_main_module_166);
    auto x_main_module_873 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_871, x_main_module_872);
    auto x_main_module_874 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,32,35,35]}"), x_main_module_873);
    auto x_main_module_875 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_869, x_main_module_874);
    auto x_main_module_876 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,32,35,35]}"), x_main_module_864);
    auto x_main_module_877 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_875, x_main_module_876);
    auto x_main_module_878 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,32,35,35]}"), x_main_module_865);
    auto x_main_module_879 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_877, x_main_module_878);
    auto x_main_module_880 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_879);
    auto x_main_module_881 = mmain->add_instruction(migraphx::make_json_op("concat", "{axis:1}"),
                                                    x_main_module_771,
                                                    x_main_module_807,
                                                    x_main_module_861,
                                                    x_main_module_880);
    auto x_main_module_882 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_881,
        x_main_module_601);
    auto x_main_module_883 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_600);
    auto x_main_module_884 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_599);
    auto x_main_module_885 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_598);
    auto x_main_module_886 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_597);
    auto x_main_module_887 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_885);
    auto x_main_module_888 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_882, x_main_module_887);
    auto x_main_module_889 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_163);
    auto x_main_module_890 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_886, x_main_module_889);
    auto x_main_module_891 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_164);
    auto x_main_module_892 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_890, x_main_module_891);
    auto x_main_module_893 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_892);
    auto x_main_module_894 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_888, x_main_module_893);
    auto x_main_module_895 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_883);
    auto x_main_module_896 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_894, x_main_module_895);
    auto x_main_module_897 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_884);
    auto x_main_module_898 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_896, x_main_module_897);
    auto x_main_module_899 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_898);
    auto x_main_module_900 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_881,
        x_main_module_596);
    auto x_main_module_901 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_595);
    auto x_main_module_902 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_594);
    auto x_main_module_903 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_593);
    auto x_main_module_904 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_592);
    auto x_main_module_905 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,48,35,35]}"), x_main_module_903);
    auto x_main_module_906 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_900, x_main_module_905);
    auto x_main_module_907 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[48,1,1]}"), x_main_module_161);
    auto x_main_module_908 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_904, x_main_module_907);
    auto x_main_module_909 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[48,1,1]}"), x_main_module_162);
    auto x_main_module_910 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_908, x_main_module_909);
    auto x_main_module_911 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,48,35,35]}"), x_main_module_910);
    auto x_main_module_912 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_906, x_main_module_911);
    auto x_main_module_913 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,48,35,35]}"), x_main_module_901);
    auto x_main_module_914 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_912, x_main_module_913);
    auto x_main_module_915 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,48,35,35]}"), x_main_module_902);
    auto x_main_module_916 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_914, x_main_module_915);
    auto x_main_module_917 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_916);
    auto x_main_module_918 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[2,2,2,2],padding_mode:0,stride:[1,1]}"),
        x_main_module_917,
        x_main_module_591);
    auto x_main_module_919 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_590);
    auto x_main_module_920 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_589);
    auto x_main_module_921 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_588);
    auto x_main_module_922 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_587);
    auto x_main_module_923 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_921);
    auto x_main_module_924 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_918, x_main_module_923);
    auto x_main_module_925 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_159);
    auto x_main_module_926 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_922, x_main_module_925);
    auto x_main_module_927 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_160);
    auto x_main_module_928 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_926, x_main_module_927);
    auto x_main_module_929 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_928);
    auto x_main_module_930 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_924, x_main_module_929);
    auto x_main_module_931 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_919);
    auto x_main_module_932 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_930, x_main_module_931);
    auto x_main_module_933 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_920);
    auto x_main_module_934 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_932, x_main_module_933);
    auto x_main_module_935 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_934);
    auto x_main_module_936 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_881,
        x_main_module_586);
    auto x_main_module_937 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_585);
    auto x_main_module_938 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_584);
    auto x_main_module_939 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_583);
    auto x_main_module_940 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_582);
    auto x_main_module_941 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_939);
    auto x_main_module_942 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_936, x_main_module_941);
    auto x_main_module_943 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_157);
    auto x_main_module_944 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_940, x_main_module_943);
    auto x_main_module_945 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_158);
    auto x_main_module_946 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_944, x_main_module_945);
    auto x_main_module_947 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_946);
    auto x_main_module_948 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_942, x_main_module_947);
    auto x_main_module_949 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_937);
    auto x_main_module_950 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_948, x_main_module_949);
    auto x_main_module_951 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_938);
    auto x_main_module_952 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_950, x_main_module_951);
    auto x_main_module_953 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_952);
    auto x_main_module_954 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}"),
        x_main_module_953,
        x_main_module_581);
    auto x_main_module_955 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_580);
    auto x_main_module_956 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_579);
    auto x_main_module_957 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_578);
    auto x_main_module_958 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_577);
    auto x_main_module_959 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,96,35,35]}"), x_main_module_957);
    auto x_main_module_960 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_954, x_main_module_959);
    auto x_main_module_961 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[96,1,1]}"), x_main_module_155);
    auto x_main_module_962 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_958, x_main_module_961);
    auto x_main_module_963 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[96,1,1]}"), x_main_module_156);
    auto x_main_module_964 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_962, x_main_module_963);
    auto x_main_module_965 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,96,35,35]}"), x_main_module_964);
    auto x_main_module_966 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_960, x_main_module_965);
    auto x_main_module_967 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,96,35,35]}"), x_main_module_955);
    auto x_main_module_968 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_966, x_main_module_967);
    auto x_main_module_969 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,96,35,35]}"), x_main_module_956);
    auto x_main_module_970 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_968, x_main_module_969);
    auto x_main_module_971 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_970);
    auto x_main_module_972 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}"),
        x_main_module_971,
        x_main_module_576);
    auto x_main_module_973 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_575);
    auto x_main_module_974 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_574);
    auto x_main_module_975 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_573);
    auto x_main_module_976 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_572);
    auto x_main_module_977 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,96,35,35]}"), x_main_module_975);
    auto x_main_module_978 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_972, x_main_module_977);
    auto x_main_module_979 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[96,1,1]}"), x_main_module_153);
    auto x_main_module_980 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_976, x_main_module_979);
    auto x_main_module_981 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[96,1,1]}"), x_main_module_154);
    auto x_main_module_982 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_980, x_main_module_981);
    auto x_main_module_983 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,96,35,35]}"), x_main_module_982);
    auto x_main_module_984 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_978, x_main_module_983);
    auto x_main_module_985 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,96,35,35]}"), x_main_module_973);
    auto x_main_module_986 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_984, x_main_module_985);
    auto x_main_module_987 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,96,35,35]}"), x_main_module_974);
    auto x_main_module_988 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_986, x_main_module_987);
    auto x_main_module_989 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_988);
    auto x_main_module_990 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:0,padding:[1,1,1,1],stride:[1,1]}"),
        x_main_module_881);
    auto x_main_module_991 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_990,
        x_main_module_571);
    auto x_main_module_992 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_570);
    auto x_main_module_993 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_569);
    auto x_main_module_994 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_568);
    auto x_main_module_995 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_567);
    auto x_main_module_996 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_994);
    auto x_main_module_997 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_991, x_main_module_996);
    auto x_main_module_998 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_151);
    auto x_main_module_999 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_995, x_main_module_998);
    auto x_main_module_1000 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_152);
    auto x_main_module_1001 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_999, x_main_module_1000);
    auto x_main_module_1002 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_1001);
    auto x_main_module_1003 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_997, x_main_module_1002);
    auto x_main_module_1004 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_992);
    auto x_main_module_1005 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1003, x_main_module_1004);
    auto x_main_module_1006 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_993);
    auto x_main_module_1007 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1005, x_main_module_1006);
    auto x_main_module_1008 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1007);
    auto x_main_module_1009 = mmain->add_instruction(migraphx::make_json_op("concat", "{axis:1}"),
                                                     x_main_module_899,
                                                     x_main_module_935,
                                                     x_main_module_989,
                                                     x_main_module_1008);
    auto x_main_module_1010 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1009,
        x_main_module_566);
    auto x_main_module_1011 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_565);
    auto x_main_module_1012 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_564);
    auto x_main_module_1013 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_563);
    auto x_main_module_1014 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_562);
    auto x_main_module_1015 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_1013);
    auto x_main_module_1016 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1010, x_main_module_1015);
    auto x_main_module_1017 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_149);
    auto x_main_module_1018 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1014, x_main_module_1017);
    auto x_main_module_1019 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_150);
    auto x_main_module_1020 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1018, x_main_module_1019);
    auto x_main_module_1021 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_1020);
    auto x_main_module_1022 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1016, x_main_module_1021);
    auto x_main_module_1023 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_1011);
    auto x_main_module_1024 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1022, x_main_module_1023);
    auto x_main_module_1025 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_1012);
    auto x_main_module_1026 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1024, x_main_module_1025);
    auto x_main_module_1027 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1026);
    auto x_main_module_1028 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1009,
        x_main_module_561);
    auto x_main_module_1029 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_560);
    auto x_main_module_1030 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_559);
    auto x_main_module_1031 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_558);
    auto x_main_module_1032 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_557);
    auto x_main_module_1033 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,48,35,35]}"), x_main_module_1031);
    auto x_main_module_1034 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1028, x_main_module_1033);
    auto x_main_module_1035 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[48,1,1]}"), x_main_module_147);
    auto x_main_module_1036 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1032, x_main_module_1035);
    auto x_main_module_1037 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[48,1,1]}"), x_main_module_148);
    auto x_main_module_1038 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1036, x_main_module_1037);
    auto x_main_module_1039 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,48,35,35]}"), x_main_module_1038);
    auto x_main_module_1040 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1034, x_main_module_1039);
    auto x_main_module_1041 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,48,35,35]}"), x_main_module_1029);
    auto x_main_module_1042 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1040, x_main_module_1041);
    auto x_main_module_1043 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,48,35,35]}"), x_main_module_1030);
    auto x_main_module_1044 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1042, x_main_module_1043);
    auto x_main_module_1045 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1044);
    auto x_main_module_1046 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[2,2,2,2],padding_mode:0,stride:[1,1]}"),
        x_main_module_1045,
        x_main_module_556);
    auto x_main_module_1047 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_555);
    auto x_main_module_1048 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_554);
    auto x_main_module_1049 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_553);
    auto x_main_module_1050 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_552);
    auto x_main_module_1051 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_1049);
    auto x_main_module_1052 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1046, x_main_module_1051);
    auto x_main_module_1053 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_145);
    auto x_main_module_1054 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1050, x_main_module_1053);
    auto x_main_module_1055 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_146);
    auto x_main_module_1056 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1054, x_main_module_1055);
    auto x_main_module_1057 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_1056);
    auto x_main_module_1058 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1052, x_main_module_1057);
    auto x_main_module_1059 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_1047);
    auto x_main_module_1060 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1058, x_main_module_1059);
    auto x_main_module_1061 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_1048);
    auto x_main_module_1062 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1060, x_main_module_1061);
    auto x_main_module_1063 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1062);
    auto x_main_module_1064 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1009,
        x_main_module_551);
    auto x_main_module_1065 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_550);
    auto x_main_module_1066 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_549);
    auto x_main_module_1067 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_548);
    auto x_main_module_1068 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_547);
    auto x_main_module_1069 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_1067);
    auto x_main_module_1070 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1064, x_main_module_1069);
    auto x_main_module_1071 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_143);
    auto x_main_module_1072 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1068, x_main_module_1071);
    auto x_main_module_1073 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_144);
    auto x_main_module_1074 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1072, x_main_module_1073);
    auto x_main_module_1075 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_1074);
    auto x_main_module_1076 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1070, x_main_module_1075);
    auto x_main_module_1077 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_1065);
    auto x_main_module_1078 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1076, x_main_module_1077);
    auto x_main_module_1079 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_1066);
    auto x_main_module_1080 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1078, x_main_module_1079);
    auto x_main_module_1081 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1080);
    auto x_main_module_1082 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}"),
        x_main_module_1081,
        x_main_module_546);
    auto x_main_module_1083 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_545);
    auto x_main_module_1084 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_544);
    auto x_main_module_1085 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_543);
    auto x_main_module_1086 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_542);
    auto x_main_module_1087 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,96,35,35]}"), x_main_module_1085);
    auto x_main_module_1088 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1082, x_main_module_1087);
    auto x_main_module_1089 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[96,1,1]}"), x_main_module_141);
    auto x_main_module_1090 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1086, x_main_module_1089);
    auto x_main_module_1091 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[96,1,1]}"), x_main_module_142);
    auto x_main_module_1092 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1090, x_main_module_1091);
    auto x_main_module_1093 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,96,35,35]}"), x_main_module_1092);
    auto x_main_module_1094 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1088, x_main_module_1093);
    auto x_main_module_1095 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,96,35,35]}"), x_main_module_1083);
    auto x_main_module_1096 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1094, x_main_module_1095);
    auto x_main_module_1097 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,96,35,35]}"), x_main_module_1084);
    auto x_main_module_1098 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1096, x_main_module_1097);
    auto x_main_module_1099 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1098);
    auto x_main_module_1100 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}"),
        x_main_module_1099,
        x_main_module_541);
    auto x_main_module_1101 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_540);
    auto x_main_module_1102 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_539);
    auto x_main_module_1103 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_538);
    auto x_main_module_1104 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_537);
    auto x_main_module_1105 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,96,35,35]}"), x_main_module_1103);
    auto x_main_module_1106 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1100, x_main_module_1105);
    auto x_main_module_1107 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[96,1,1]}"), x_main_module_139);
    auto x_main_module_1108 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1104, x_main_module_1107);
    auto x_main_module_1109 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[96,1,1]}"), x_main_module_140);
    auto x_main_module_1110 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1108, x_main_module_1109);
    auto x_main_module_1111 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,96,35,35]}"), x_main_module_1110);
    auto x_main_module_1112 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1106, x_main_module_1111);
    auto x_main_module_1113 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,96,35,35]}"), x_main_module_1101);
    auto x_main_module_1114 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1112, x_main_module_1113);
    auto x_main_module_1115 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,96,35,35]}"), x_main_module_1102);
    auto x_main_module_1116 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1114, x_main_module_1115);
    auto x_main_module_1117 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1116);
    auto x_main_module_1118 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:0,padding:[1,1,1,1],stride:[1,1]}"),
        x_main_module_1009);
    auto x_main_module_1119 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1118,
        x_main_module_536);
    auto x_main_module_1120 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_535);
    auto x_main_module_1121 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_534);
    auto x_main_module_1122 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_533);
    auto x_main_module_1123 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_532);
    auto x_main_module_1124 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_1122);
    auto x_main_module_1125 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1119, x_main_module_1124);
    auto x_main_module_1126 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_137);
    auto x_main_module_1127 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1123, x_main_module_1126);
    auto x_main_module_1128 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_138);
    auto x_main_module_1129 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1127, x_main_module_1128);
    auto x_main_module_1130 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_1129);
    auto x_main_module_1131 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1125, x_main_module_1130);
    auto x_main_module_1132 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_1120);
    auto x_main_module_1133 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1131, x_main_module_1132);
    auto x_main_module_1134 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_1121);
    auto x_main_module_1135 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1133, x_main_module_1134);
    auto x_main_module_1136 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1135);
    auto x_main_module_1137 = mmain->add_instruction(migraphx::make_json_op("concat", "{axis:1}"),
                                                     x_main_module_1027,
                                                     x_main_module_1063,
                                                     x_main_module_1117,
                                                     x_main_module_1136);
    auto x_main_module_1138 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[2,2]}"),
        x_main_module_1137,
        x_main_module_531);
    auto x_main_module_1139 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_530);
    auto x_main_module_1140 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_529);
    auto x_main_module_1141 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_528);
    auto x_main_module_1142 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_527);
    auto x_main_module_1143 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,17,17]}"), x_main_module_1141);
    auto x_main_module_1144 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1138, x_main_module_1143);
    auto x_main_module_1145 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[384,1,1]}"), x_main_module_135);
    auto x_main_module_1146 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1142, x_main_module_1145);
    auto x_main_module_1147 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[384,1,1]}"), x_main_module_136);
    auto x_main_module_1148 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1146, x_main_module_1147);
    auto x_main_module_1149 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,17,17]}"), x_main_module_1148);
    auto x_main_module_1150 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1144, x_main_module_1149);
    auto x_main_module_1151 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,17,17]}"), x_main_module_1139);
    auto x_main_module_1152 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1150, x_main_module_1151);
    auto x_main_module_1153 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,17,17]}"), x_main_module_1140);
    auto x_main_module_1154 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1152, x_main_module_1153);
    auto x_main_module_1155 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1154);
    auto x_main_module_1156 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1137,
        x_main_module_526);
    auto x_main_module_1157 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_525);
    auto x_main_module_1158 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_524);
    auto x_main_module_1159 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_523);
    auto x_main_module_1160 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_522);
    auto x_main_module_1161 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_1159);
    auto x_main_module_1162 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1156, x_main_module_1161);
    auto x_main_module_1163 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_133);
    auto x_main_module_1164 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1160, x_main_module_1163);
    auto x_main_module_1165 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[64,1,1]}"), x_main_module_134);
    auto x_main_module_1166 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1164, x_main_module_1165);
    auto x_main_module_1167 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_1166);
    auto x_main_module_1168 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1162, x_main_module_1167);
    auto x_main_module_1169 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_1157);
    auto x_main_module_1170 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1168, x_main_module_1169);
    auto x_main_module_1171 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,64,35,35]}"), x_main_module_1158);
    auto x_main_module_1172 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1170, x_main_module_1171);
    auto x_main_module_1173 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1172);
    auto x_main_module_1174 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}"),
        x_main_module_1173,
        x_main_module_521);
    auto x_main_module_1175 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_520);
    auto x_main_module_1176 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_519);
    auto x_main_module_1177 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_518);
    auto x_main_module_1178 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_517);
    auto x_main_module_1179 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,96,35,35]}"), x_main_module_1177);
    auto x_main_module_1180 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1174, x_main_module_1179);
    auto x_main_module_1181 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[96,1,1]}"), x_main_module_131);
    auto x_main_module_1182 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1178, x_main_module_1181);
    auto x_main_module_1183 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[96,1,1]}"), x_main_module_132);
    auto x_main_module_1184 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1182, x_main_module_1183);
    auto x_main_module_1185 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,96,35,35]}"), x_main_module_1184);
    auto x_main_module_1186 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1180, x_main_module_1185);
    auto x_main_module_1187 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,96,35,35]}"), x_main_module_1175);
    auto x_main_module_1188 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1186, x_main_module_1187);
    auto x_main_module_1189 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,96,35,35]}"), x_main_module_1176);
    auto x_main_module_1190 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1188, x_main_module_1189);
    auto x_main_module_1191 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1190);
    auto x_main_module_1192 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[2,2]}"),
        x_main_module_1191,
        x_main_module_516);
    auto x_main_module_1193 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_515);
    auto x_main_module_1194 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_514);
    auto x_main_module_1195 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_513);
    auto x_main_module_1196 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_512);
    auto x_main_module_1197 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,96,17,17]}"), x_main_module_1195);
    auto x_main_module_1198 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1192, x_main_module_1197);
    auto x_main_module_1199 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[96,1,1]}"), x_main_module_129);
    auto x_main_module_1200 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1196, x_main_module_1199);
    auto x_main_module_1201 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[96,1,1]}"), x_main_module_130);
    auto x_main_module_1202 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1200, x_main_module_1201);
    auto x_main_module_1203 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,96,17,17]}"), x_main_module_1202);
    auto x_main_module_1204 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1198, x_main_module_1203);
    auto x_main_module_1205 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,96,17,17]}"), x_main_module_1193);
    auto x_main_module_1206 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1204, x_main_module_1205);
    auto x_main_module_1207 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,96,17,17]}"), x_main_module_1194);
    auto x_main_module_1208 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1206, x_main_module_1207);
    auto x_main_module_1209 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1208);
    auto x_main_module_1210 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:1,padding:[0,0,0,0],stride:[2,2]}"),
        x_main_module_1137);
    auto x_main_module_1211 = mmain->add_instruction(migraphx::make_json_op("concat", "{axis:1}"),
                                                     x_main_module_1155,
                                                     x_main_module_1209,
                                                     x_main_module_1210);
    auto x_main_module_1212 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1211,
        x_main_module_511);
    auto x_main_module_1213 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_510);
    auto x_main_module_1214 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_509);
    auto x_main_module_1215 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_508);
    auto x_main_module_1216 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_507);
    auto x_main_module_1217 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1215);
    auto x_main_module_1218 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1212, x_main_module_1217);
    auto x_main_module_1219 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_127);
    auto x_main_module_1220 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1216, x_main_module_1219);
    auto x_main_module_1221 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_128);
    auto x_main_module_1222 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1220, x_main_module_1221);
    auto x_main_module_1223 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1222);
    auto x_main_module_1224 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1218, x_main_module_1223);
    auto x_main_module_1225 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1213);
    auto x_main_module_1226 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1224, x_main_module_1225);
    auto x_main_module_1227 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1214);
    auto x_main_module_1228 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1226, x_main_module_1227);
    auto x_main_module_1229 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1228);
    auto x_main_module_1230 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1211,
        x_main_module_506);
    auto x_main_module_1231 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_505);
    auto x_main_module_1232 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_504);
    auto x_main_module_1233 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_503);
    auto x_main_module_1234 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_502);
    auto x_main_module_1235 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,17,17]}"), x_main_module_1233);
    auto x_main_module_1236 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1230, x_main_module_1235);
    auto x_main_module_1237 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[128,1,1]}"), x_main_module_125);
    auto x_main_module_1238 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1234, x_main_module_1237);
    auto x_main_module_1239 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[128,1,1]}"), x_main_module_126);
    auto x_main_module_1240 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1238, x_main_module_1239);
    auto x_main_module_1241 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,17,17]}"), x_main_module_1240);
    auto x_main_module_1242 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1236, x_main_module_1241);
    auto x_main_module_1243 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,17,17]}"), x_main_module_1231);
    auto x_main_module_1244 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1242, x_main_module_1243);
    auto x_main_module_1245 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,17,17]}"), x_main_module_1232);
    auto x_main_module_1246 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1244, x_main_module_1245);
    auto x_main_module_1247 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1246);
    auto x_main_module_1248 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,1]}"),
        x_main_module_1247,
        x_main_module_501);
    auto x_main_module_1249 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_500);
    auto x_main_module_1250 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_499);
    auto x_main_module_1251 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_498);
    auto x_main_module_1252 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_497);
    auto x_main_module_1253 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,17,17]}"), x_main_module_1251);
    auto x_main_module_1254 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1248, x_main_module_1253);
    auto x_main_module_1255 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[128,1,1]}"), x_main_module_123);
    auto x_main_module_1256 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1252, x_main_module_1255);
    auto x_main_module_1257 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[128,1,1]}"), x_main_module_124);
    auto x_main_module_1258 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1256, x_main_module_1257);
    auto x_main_module_1259 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,17,17]}"), x_main_module_1258);
    auto x_main_module_1260 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1254, x_main_module_1259);
    auto x_main_module_1261 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,17,17]}"), x_main_module_1249);
    auto x_main_module_1262 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1260, x_main_module_1261);
    auto x_main_module_1263 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,17,17]}"), x_main_module_1250);
    auto x_main_module_1264 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1262, x_main_module_1263);
    auto x_main_module_1265 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1264);
    auto x_main_module_1266 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1265,
        x_main_module_496);
    auto x_main_module_1267 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_495);
    auto x_main_module_1268 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_494);
    auto x_main_module_1269 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_493);
    auto x_main_module_1270 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_492);
    auto x_main_module_1271 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1269);
    auto x_main_module_1272 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1266, x_main_module_1271);
    auto x_main_module_1273 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_121);
    auto x_main_module_1274 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1270, x_main_module_1273);
    auto x_main_module_1275 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_122);
    auto x_main_module_1276 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1274, x_main_module_1275);
    auto x_main_module_1277 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1276);
    auto x_main_module_1278 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1272, x_main_module_1277);
    auto x_main_module_1279 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1267);
    auto x_main_module_1280 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1278, x_main_module_1279);
    auto x_main_module_1281 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1268);
    auto x_main_module_1282 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1280, x_main_module_1281);
    auto x_main_module_1283 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1282);
    auto x_main_module_1284 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1211,
        x_main_module_491);
    auto x_main_module_1285 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_490);
    auto x_main_module_1286 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_489);
    auto x_main_module_1287 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_488);
    auto x_main_module_1288 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_487);
    auto x_main_module_1289 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,17,17]}"), x_main_module_1287);
    auto x_main_module_1290 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1284, x_main_module_1289);
    auto x_main_module_1291 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[128,1,1]}"), x_main_module_119);
    auto x_main_module_1292 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1288, x_main_module_1291);
    auto x_main_module_1293 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[128,1,1]}"), x_main_module_120);
    auto x_main_module_1294 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1292, x_main_module_1293);
    auto x_main_module_1295 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,17,17]}"), x_main_module_1294);
    auto x_main_module_1296 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1290, x_main_module_1295);
    auto x_main_module_1297 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,17,17]}"), x_main_module_1285);
    auto x_main_module_1298 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1296, x_main_module_1297);
    auto x_main_module_1299 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,17,17]}"), x_main_module_1286);
    auto x_main_module_1300 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1298, x_main_module_1299);
    auto x_main_module_1301 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1300);
    auto x_main_module_1302 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1301,
        x_main_module_486);
    auto x_main_module_1303 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_485);
    auto x_main_module_1304 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_484);
    auto x_main_module_1305 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_483);
    auto x_main_module_1306 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_482);
    auto x_main_module_1307 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,17,17]}"), x_main_module_1305);
    auto x_main_module_1308 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1302, x_main_module_1307);
    auto x_main_module_1309 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[128,1,1]}"), x_main_module_117);
    auto x_main_module_1310 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1306, x_main_module_1309);
    auto x_main_module_1311 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[128,1,1]}"), x_main_module_118);
    auto x_main_module_1312 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1310, x_main_module_1311);
    auto x_main_module_1313 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,17,17]}"), x_main_module_1312);
    auto x_main_module_1314 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1308, x_main_module_1313);
    auto x_main_module_1315 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,17,17]}"), x_main_module_1303);
    auto x_main_module_1316 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1314, x_main_module_1315);
    auto x_main_module_1317 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,17,17]}"), x_main_module_1304);
    auto x_main_module_1318 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1316, x_main_module_1317);
    auto x_main_module_1319 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1318);
    auto x_main_module_1320 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,1]}"),
        x_main_module_1319,
        x_main_module_481);
    auto x_main_module_1321 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_480);
    auto x_main_module_1322 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_479);
    auto x_main_module_1323 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_478);
    auto x_main_module_1324 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_477);
    auto x_main_module_1325 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,17,17]}"), x_main_module_1323);
    auto x_main_module_1326 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1320, x_main_module_1325);
    auto x_main_module_1327 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[128,1,1]}"), x_main_module_115);
    auto x_main_module_1328 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1324, x_main_module_1327);
    auto x_main_module_1329 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[128,1,1]}"), x_main_module_116);
    auto x_main_module_1330 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1328, x_main_module_1329);
    auto x_main_module_1331 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,17,17]}"), x_main_module_1330);
    auto x_main_module_1332 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1326, x_main_module_1331);
    auto x_main_module_1333 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,17,17]}"), x_main_module_1321);
    auto x_main_module_1334 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1332, x_main_module_1333);
    auto x_main_module_1335 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,17,17]}"), x_main_module_1322);
    auto x_main_module_1336 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1334, x_main_module_1335);
    auto x_main_module_1337 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1336);
    auto x_main_module_1338 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1337,
        x_main_module_476);
    auto x_main_module_1339 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_475);
    auto x_main_module_1340 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_474);
    auto x_main_module_1341 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_473);
    auto x_main_module_1342 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_472);
    auto x_main_module_1343 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,17,17]}"), x_main_module_1341);
    auto x_main_module_1344 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1338, x_main_module_1343);
    auto x_main_module_1345 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[128,1,1]}"), x_main_module_113);
    auto x_main_module_1346 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1342, x_main_module_1345);
    auto x_main_module_1347 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[128,1,1]}"), x_main_module_114);
    auto x_main_module_1348 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1346, x_main_module_1347);
    auto x_main_module_1349 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,17,17]}"), x_main_module_1348);
    auto x_main_module_1350 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1344, x_main_module_1349);
    auto x_main_module_1351 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,17,17]}"), x_main_module_1339);
    auto x_main_module_1352 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1350, x_main_module_1351);
    auto x_main_module_1353 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,128,17,17]}"), x_main_module_1340);
    auto x_main_module_1354 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1352, x_main_module_1353);
    auto x_main_module_1355 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1354);
    auto x_main_module_1356 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,1]}"),
        x_main_module_1355,
        x_main_module_471);
    auto x_main_module_1357 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_470);
    auto x_main_module_1358 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_469);
    auto x_main_module_1359 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_468);
    auto x_main_module_1360 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_467);
    auto x_main_module_1361 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1359);
    auto x_main_module_1362 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1356, x_main_module_1361);
    auto x_main_module_1363 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_111);
    auto x_main_module_1364 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1360, x_main_module_1363);
    auto x_main_module_1365 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_112);
    auto x_main_module_1366 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1364, x_main_module_1365);
    auto x_main_module_1367 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1366);
    auto x_main_module_1368 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1362, x_main_module_1367);
    auto x_main_module_1369 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1357);
    auto x_main_module_1370 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1368, x_main_module_1369);
    auto x_main_module_1371 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1358);
    auto x_main_module_1372 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1370, x_main_module_1371);
    auto x_main_module_1373 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1372);
    auto x_main_module_1374 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:0,padding:[1,1,1,1],stride:[1,1]}"),
        x_main_module_1211);
    auto x_main_module_1375 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1374,
        x_main_module_466);
    auto x_main_module_1376 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_465);
    auto x_main_module_1377 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_464);
    auto x_main_module_1378 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_463);
    auto x_main_module_1379 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_462);
    auto x_main_module_1380 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1378);
    auto x_main_module_1381 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1375, x_main_module_1380);
    auto x_main_module_1382 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_109);
    auto x_main_module_1383 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1379, x_main_module_1382);
    auto x_main_module_1384 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_110);
    auto x_main_module_1385 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1383, x_main_module_1384);
    auto x_main_module_1386 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1385);
    auto x_main_module_1387 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1381, x_main_module_1386);
    auto x_main_module_1388 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1376);
    auto x_main_module_1389 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1387, x_main_module_1388);
    auto x_main_module_1390 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1377);
    auto x_main_module_1391 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1389, x_main_module_1390);
    auto x_main_module_1392 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1391);
    auto x_main_module_1393 = mmain->add_instruction(migraphx::make_json_op("concat", "{axis:1}"),
                                                     x_main_module_1229,
                                                     x_main_module_1283,
                                                     x_main_module_1373,
                                                     x_main_module_1392);
    auto x_main_module_1394 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1393,
        x_main_module_461);
    auto x_main_module_1395 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_460);
    auto x_main_module_1396 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_459);
    auto x_main_module_1397 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_458);
    auto x_main_module_1398 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_457);
    auto x_main_module_1399 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1397);
    auto x_main_module_1400 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1394, x_main_module_1399);
    auto x_main_module_1401 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_107);
    auto x_main_module_1402 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1398, x_main_module_1401);
    auto x_main_module_1403 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_108);
    auto x_main_module_1404 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1402, x_main_module_1403);
    auto x_main_module_1405 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1404);
    auto x_main_module_1406 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1400, x_main_module_1405);
    auto x_main_module_1407 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1395);
    auto x_main_module_1408 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1406, x_main_module_1407);
    auto x_main_module_1409 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1396);
    auto x_main_module_1410 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1408, x_main_module_1409);
    auto x_main_module_1411 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1410);
    auto x_main_module_1412 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1393,
        x_main_module_456);
    auto x_main_module_1413 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_455);
    auto x_main_module_1414 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_454);
    auto x_main_module_1415 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_453);
    auto x_main_module_1416 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_452);
    auto x_main_module_1417 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1415);
    auto x_main_module_1418 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1412, x_main_module_1417);
    auto x_main_module_1419 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[160,1,1]}"), x_main_module_105);
    auto x_main_module_1420 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1416, x_main_module_1419);
    auto x_main_module_1421 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[160,1,1]}"), x_main_module_106);
    auto x_main_module_1422 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1420, x_main_module_1421);
    auto x_main_module_1423 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1422);
    auto x_main_module_1424 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1418, x_main_module_1423);
    auto x_main_module_1425 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1413);
    auto x_main_module_1426 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1424, x_main_module_1425);
    auto x_main_module_1427 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1414);
    auto x_main_module_1428 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1426, x_main_module_1427);
    auto x_main_module_1429 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1428);
    auto x_main_module_1430 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,1]}"),
        x_main_module_1429,
        x_main_module_451);
    auto x_main_module_1431 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_450);
    auto x_main_module_1432 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_449);
    auto x_main_module_1433 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_448);
    auto x_main_module_1434 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_447);
    auto x_main_module_1435 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1433);
    auto x_main_module_1436 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1430, x_main_module_1435);
    auto x_main_module_1437 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[160,1,1]}"), x_main_module_103);
    auto x_main_module_1438 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1434, x_main_module_1437);
    auto x_main_module_1439 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[160,1,1]}"), x_main_module_104);
    auto x_main_module_1440 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1438, x_main_module_1439);
    auto x_main_module_1441 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1440);
    auto x_main_module_1442 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1436, x_main_module_1441);
    auto x_main_module_1443 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1431);
    auto x_main_module_1444 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1442, x_main_module_1443);
    auto x_main_module_1445 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1432);
    auto x_main_module_1446 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1444, x_main_module_1445);
    auto x_main_module_1447 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1446);
    auto x_main_module_1448 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1447,
        x_main_module_446);
    auto x_main_module_1449 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_445);
    auto x_main_module_1450 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_444);
    auto x_main_module_1451 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_443);
    auto x_main_module_1452 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_442);
    auto x_main_module_1453 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1451);
    auto x_main_module_1454 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1448, x_main_module_1453);
    auto x_main_module_1455 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_101);
    auto x_main_module_1456 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1452, x_main_module_1455);
    auto x_main_module_1457 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_102);
    auto x_main_module_1458 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1456, x_main_module_1457);
    auto x_main_module_1459 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1458);
    auto x_main_module_1460 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1454, x_main_module_1459);
    auto x_main_module_1461 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1449);
    auto x_main_module_1462 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1460, x_main_module_1461);
    auto x_main_module_1463 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1450);
    auto x_main_module_1464 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1462, x_main_module_1463);
    auto x_main_module_1465 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1464);
    auto x_main_module_1466 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1393,
        x_main_module_441);
    auto x_main_module_1467 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_440);
    auto x_main_module_1468 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_439);
    auto x_main_module_1469 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_438);
    auto x_main_module_1470 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_437);
    auto x_main_module_1471 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1469);
    auto x_main_module_1472 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1466, x_main_module_1471);
    auto x_main_module_1473 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[160,1,1]}"), x_main_module_99);
    auto x_main_module_1474 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1470, x_main_module_1473);
    auto x_main_module_1475 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[160,1,1]}"), x_main_module_100);
    auto x_main_module_1476 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1474, x_main_module_1475);
    auto x_main_module_1477 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1476);
    auto x_main_module_1478 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1472, x_main_module_1477);
    auto x_main_module_1479 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1467);
    auto x_main_module_1480 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1478, x_main_module_1479);
    auto x_main_module_1481 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1468);
    auto x_main_module_1482 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1480, x_main_module_1481);
    auto x_main_module_1483 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1482);
    auto x_main_module_1484 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1483,
        x_main_module_436);
    auto x_main_module_1485 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_435);
    auto x_main_module_1486 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_434);
    auto x_main_module_1487 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_433);
    auto x_main_module_1488 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_432);
    auto x_main_module_1489 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1487);
    auto x_main_module_1490 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1484, x_main_module_1489);
    auto x_main_module_1491 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[160,1,1]}"), x_main_module_97);
    auto x_main_module_1492 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1488, x_main_module_1491);
    auto x_main_module_1493 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[160,1,1]}"), x_main_module_98);
    auto x_main_module_1494 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1492, x_main_module_1493);
    auto x_main_module_1495 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1494);
    auto x_main_module_1496 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1490, x_main_module_1495);
    auto x_main_module_1497 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1485);
    auto x_main_module_1498 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1496, x_main_module_1497);
    auto x_main_module_1499 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1486);
    auto x_main_module_1500 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1498, x_main_module_1499);
    auto x_main_module_1501 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1500);
    auto x_main_module_1502 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,1]}"),
        x_main_module_1501,
        x_main_module_431);
    auto x_main_module_1503 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_430);
    auto x_main_module_1504 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_429);
    auto x_main_module_1505 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_428);
    auto x_main_module_1506 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_427);
    auto x_main_module_1507 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1505);
    auto x_main_module_1508 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1502, x_main_module_1507);
    auto x_main_module_1509 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[160,1,1]}"), x_main_module_95);
    auto x_main_module_1510 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1506, x_main_module_1509);
    auto x_main_module_1511 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[160,1,1]}"), x_main_module_96);
    auto x_main_module_1512 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1510, x_main_module_1511);
    auto x_main_module_1513 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1512);
    auto x_main_module_1514 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1508, x_main_module_1513);
    auto x_main_module_1515 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1503);
    auto x_main_module_1516 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1514, x_main_module_1515);
    auto x_main_module_1517 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1504);
    auto x_main_module_1518 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1516, x_main_module_1517);
    auto x_main_module_1519 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1518);
    auto x_main_module_1520 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1519,
        x_main_module_426);
    auto x_main_module_1521 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_425);
    auto x_main_module_1522 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_424);
    auto x_main_module_1523 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_423);
    auto x_main_module_1524 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_422);
    auto x_main_module_1525 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1523);
    auto x_main_module_1526 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1520, x_main_module_1525);
    auto x_main_module_1527 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[160,1,1]}"), x_main_module_93);
    auto x_main_module_1528 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1524, x_main_module_1527);
    auto x_main_module_1529 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[160,1,1]}"), x_main_module_94);
    auto x_main_module_1530 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1528, x_main_module_1529);
    auto x_main_module_1531 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1530);
    auto x_main_module_1532 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1526, x_main_module_1531);
    auto x_main_module_1533 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1521);
    auto x_main_module_1534 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1532, x_main_module_1533);
    auto x_main_module_1535 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1522);
    auto x_main_module_1536 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1534, x_main_module_1535);
    auto x_main_module_1537 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1536);
    auto x_main_module_1538 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,1]}"),
        x_main_module_1537,
        x_main_module_421);
    auto x_main_module_1539 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_420);
    auto x_main_module_1540 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_419);
    auto x_main_module_1541 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_418);
    auto x_main_module_1542 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_417);
    auto x_main_module_1543 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1541);
    auto x_main_module_1544 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1538, x_main_module_1543);
    auto x_main_module_1545 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_91);
    auto x_main_module_1546 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1542, x_main_module_1545);
    auto x_main_module_1547 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_92);
    auto x_main_module_1548 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1546, x_main_module_1547);
    auto x_main_module_1549 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1548);
    auto x_main_module_1550 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1544, x_main_module_1549);
    auto x_main_module_1551 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1539);
    auto x_main_module_1552 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1550, x_main_module_1551);
    auto x_main_module_1553 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1540);
    auto x_main_module_1554 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1552, x_main_module_1553);
    auto x_main_module_1555 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1554);
    auto x_main_module_1556 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:0,padding:[1,1,1,1],stride:[1,1]}"),
        x_main_module_1393);
    auto x_main_module_1557 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1556,
        x_main_module_416);
    auto x_main_module_1558 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_415);
    auto x_main_module_1559 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_414);
    auto x_main_module_1560 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_413);
    auto x_main_module_1561 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_412);
    auto x_main_module_1562 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1560);
    auto x_main_module_1563 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1557, x_main_module_1562);
    auto x_main_module_1564 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_89);
    auto x_main_module_1565 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1561, x_main_module_1564);
    auto x_main_module_1566 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_90);
    auto x_main_module_1567 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1565, x_main_module_1566);
    auto x_main_module_1568 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1567);
    auto x_main_module_1569 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1563, x_main_module_1568);
    auto x_main_module_1570 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1558);
    auto x_main_module_1571 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1569, x_main_module_1570);
    auto x_main_module_1572 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1559);
    auto x_main_module_1573 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1571, x_main_module_1572);
    auto x_main_module_1574 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1573);
    auto x_main_module_1575 = mmain->add_instruction(migraphx::make_json_op("concat", "{axis:1}"),
                                                     x_main_module_1411,
                                                     x_main_module_1465,
                                                     x_main_module_1555,
                                                     x_main_module_1574);
    auto x_main_module_1576 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1575,
        x_main_module_411);
    auto x_main_module_1577 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_410);
    auto x_main_module_1578 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_409);
    auto x_main_module_1579 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_408);
    auto x_main_module_1580 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_407);
    auto x_main_module_1581 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1579);
    auto x_main_module_1582 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1576, x_main_module_1581);
    auto x_main_module_1583 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_87);
    auto x_main_module_1584 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1580, x_main_module_1583);
    auto x_main_module_1585 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_88);
    auto x_main_module_1586 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1584, x_main_module_1585);
    auto x_main_module_1587 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1586);
    auto x_main_module_1588 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1582, x_main_module_1587);
    auto x_main_module_1589 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1577);
    auto x_main_module_1590 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1588, x_main_module_1589);
    auto x_main_module_1591 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1578);
    auto x_main_module_1592 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1590, x_main_module_1591);
    auto x_main_module_1593 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1592);
    auto x_main_module_1594 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1575,
        x_main_module_406);
    auto x_main_module_1595 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_405);
    auto x_main_module_1596 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_404);
    auto x_main_module_1597 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_403);
    auto x_main_module_1598 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_402);
    auto x_main_module_1599 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1597);
    auto x_main_module_1600 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1594, x_main_module_1599);
    auto x_main_module_1601 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[160,1,1]}"), x_main_module_85);
    auto x_main_module_1602 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1598, x_main_module_1601);
    auto x_main_module_1603 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[160,1,1]}"), x_main_module_86);
    auto x_main_module_1604 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1602, x_main_module_1603);
    auto x_main_module_1605 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1604);
    auto x_main_module_1606 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1600, x_main_module_1605);
    auto x_main_module_1607 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1595);
    auto x_main_module_1608 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1606, x_main_module_1607);
    auto x_main_module_1609 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1596);
    auto x_main_module_1610 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1608, x_main_module_1609);
    auto x_main_module_1611 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1610);
    auto x_main_module_1612 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,1]}"),
        x_main_module_1611,
        x_main_module_401);
    auto x_main_module_1613 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_400);
    auto x_main_module_1614 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_399);
    auto x_main_module_1615 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_398);
    auto x_main_module_1616 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_397);
    auto x_main_module_1617 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1615);
    auto x_main_module_1618 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1612, x_main_module_1617);
    auto x_main_module_1619 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[160,1,1]}"), x_main_module_83);
    auto x_main_module_1620 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1616, x_main_module_1619);
    auto x_main_module_1621 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[160,1,1]}"), x_main_module_84);
    auto x_main_module_1622 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1620, x_main_module_1621);
    auto x_main_module_1623 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1622);
    auto x_main_module_1624 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1618, x_main_module_1623);
    auto x_main_module_1625 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1613);
    auto x_main_module_1626 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1624, x_main_module_1625);
    auto x_main_module_1627 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1614);
    auto x_main_module_1628 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1626, x_main_module_1627);
    auto x_main_module_1629 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1628);
    auto x_main_module_1630 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1629,
        x_main_module_396);
    auto x_main_module_1631 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_395);
    auto x_main_module_1632 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_394);
    auto x_main_module_1633 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_393);
    auto x_main_module_1634 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_392);
    auto x_main_module_1635 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1633);
    auto x_main_module_1636 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1630, x_main_module_1635);
    auto x_main_module_1637 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_81);
    auto x_main_module_1638 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1634, x_main_module_1637);
    auto x_main_module_1639 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_82);
    auto x_main_module_1640 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1638, x_main_module_1639);
    auto x_main_module_1641 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1640);
    auto x_main_module_1642 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1636, x_main_module_1641);
    auto x_main_module_1643 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1631);
    auto x_main_module_1644 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1642, x_main_module_1643);
    auto x_main_module_1645 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1632);
    auto x_main_module_1646 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1644, x_main_module_1645);
    auto x_main_module_1647 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1646);
    auto x_main_module_1648 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1575,
        x_main_module_391);
    auto x_main_module_1649 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_390);
    auto x_main_module_1650 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_389);
    auto x_main_module_1651 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_388);
    auto x_main_module_1652 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_387);
    auto x_main_module_1653 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1651);
    auto x_main_module_1654 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1648, x_main_module_1653);
    auto x_main_module_1655 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[160,1,1]}"), x_main_module_79);
    auto x_main_module_1656 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1652, x_main_module_1655);
    auto x_main_module_1657 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[160,1,1]}"), x_main_module_80);
    auto x_main_module_1658 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1656, x_main_module_1657);
    auto x_main_module_1659 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1658);
    auto x_main_module_1660 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1654, x_main_module_1659);
    auto x_main_module_1661 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1649);
    auto x_main_module_1662 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1660, x_main_module_1661);
    auto x_main_module_1663 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1650);
    auto x_main_module_1664 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1662, x_main_module_1663);
    auto x_main_module_1665 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1664);
    auto x_main_module_1666 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1665,
        x_main_module_386);
    auto x_main_module_1667 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_385);
    auto x_main_module_1668 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_384);
    auto x_main_module_1669 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_383);
    auto x_main_module_1670 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_382);
    auto x_main_module_1671 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1669);
    auto x_main_module_1672 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1666, x_main_module_1671);
    auto x_main_module_1673 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[160,1,1]}"), x_main_module_77);
    auto x_main_module_1674 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1670, x_main_module_1673);
    auto x_main_module_1675 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[160,1,1]}"), x_main_module_78);
    auto x_main_module_1676 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1674, x_main_module_1675);
    auto x_main_module_1677 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1676);
    auto x_main_module_1678 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1672, x_main_module_1677);
    auto x_main_module_1679 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1667);
    auto x_main_module_1680 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1678, x_main_module_1679);
    auto x_main_module_1681 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1668);
    auto x_main_module_1682 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1680, x_main_module_1681);
    auto x_main_module_1683 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1682);
    auto x_main_module_1684 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,1]}"),
        x_main_module_1683,
        x_main_module_381);
    auto x_main_module_1685 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_380);
    auto x_main_module_1686 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_379);
    auto x_main_module_1687 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_378);
    auto x_main_module_1688 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_377);
    auto x_main_module_1689 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1687);
    auto x_main_module_1690 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1684, x_main_module_1689);
    auto x_main_module_1691 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[160,1,1]}"), x_main_module_75);
    auto x_main_module_1692 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1688, x_main_module_1691);
    auto x_main_module_1693 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[160,1,1]}"), x_main_module_76);
    auto x_main_module_1694 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1692, x_main_module_1693);
    auto x_main_module_1695 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1694);
    auto x_main_module_1696 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1690, x_main_module_1695);
    auto x_main_module_1697 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1685);
    auto x_main_module_1698 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1696, x_main_module_1697);
    auto x_main_module_1699 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1686);
    auto x_main_module_1700 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1698, x_main_module_1699);
    auto x_main_module_1701 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1700);
    auto x_main_module_1702 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1701,
        x_main_module_376);
    auto x_main_module_1703 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_375);
    auto x_main_module_1704 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_374);
    auto x_main_module_1705 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_373);
    auto x_main_module_1706 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_372);
    auto x_main_module_1707 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1705);
    auto x_main_module_1708 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1702, x_main_module_1707);
    auto x_main_module_1709 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[160,1,1]}"), x_main_module_73);
    auto x_main_module_1710 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1706, x_main_module_1709);
    auto x_main_module_1711 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[160,1,1]}"), x_main_module_74);
    auto x_main_module_1712 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1710, x_main_module_1711);
    auto x_main_module_1713 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1712);
    auto x_main_module_1714 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1708, x_main_module_1713);
    auto x_main_module_1715 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1703);
    auto x_main_module_1716 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1714, x_main_module_1715);
    auto x_main_module_1717 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,160,17,17]}"), x_main_module_1704);
    auto x_main_module_1718 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1716, x_main_module_1717);
    auto x_main_module_1719 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1718);
    auto x_main_module_1720 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,1]}"),
        x_main_module_1719,
        x_main_module_371);
    auto x_main_module_1721 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_370);
    auto x_main_module_1722 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_369);
    auto x_main_module_1723 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_368);
    auto x_main_module_1724 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_367);
    auto x_main_module_1725 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1723);
    auto x_main_module_1726 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1720, x_main_module_1725);
    auto x_main_module_1727 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_71);
    auto x_main_module_1728 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1724, x_main_module_1727);
    auto x_main_module_1729 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_72);
    auto x_main_module_1730 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1728, x_main_module_1729);
    auto x_main_module_1731 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1730);
    auto x_main_module_1732 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1726, x_main_module_1731);
    auto x_main_module_1733 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1721);
    auto x_main_module_1734 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1732, x_main_module_1733);
    auto x_main_module_1735 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1722);
    auto x_main_module_1736 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1734, x_main_module_1735);
    auto x_main_module_1737 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1736);
    auto x_main_module_1738 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:0,padding:[1,1,1,1],stride:[1,1]}"),
        x_main_module_1575);
    auto x_main_module_1739 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1738,
        x_main_module_366);
    auto x_main_module_1740 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_365);
    auto x_main_module_1741 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_364);
    auto x_main_module_1742 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_363);
    auto x_main_module_1743 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_362);
    auto x_main_module_1744 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1742);
    auto x_main_module_1745 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1739, x_main_module_1744);
    auto x_main_module_1746 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_69);
    auto x_main_module_1747 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1743, x_main_module_1746);
    auto x_main_module_1748 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_70);
    auto x_main_module_1749 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1747, x_main_module_1748);
    auto x_main_module_1750 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1749);
    auto x_main_module_1751 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1745, x_main_module_1750);
    auto x_main_module_1752 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1740);
    auto x_main_module_1753 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1751, x_main_module_1752);
    auto x_main_module_1754 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1741);
    auto x_main_module_1755 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1753, x_main_module_1754);
    auto x_main_module_1756 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1755);
    auto x_main_module_1757 = mmain->add_instruction(migraphx::make_json_op("concat", "{axis:1}"),
                                                     x_main_module_1593,
                                                     x_main_module_1647,
                                                     x_main_module_1737,
                                                     x_main_module_1756);
    auto x_main_module_1758 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1757,
        x_main_module_361);
    auto x_main_module_1759 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_360);
    auto x_main_module_1760 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_359);
    auto x_main_module_1761 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_358);
    auto x_main_module_1762 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_357);
    auto x_main_module_1763 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1761);
    auto x_main_module_1764 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1758, x_main_module_1763);
    auto x_main_module_1765 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_67);
    auto x_main_module_1766 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1762, x_main_module_1765);
    auto x_main_module_1767 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_68);
    auto x_main_module_1768 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1766, x_main_module_1767);
    auto x_main_module_1769 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1768);
    auto x_main_module_1770 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1764, x_main_module_1769);
    auto x_main_module_1771 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1759);
    auto x_main_module_1772 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1770, x_main_module_1771);
    auto x_main_module_1773 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1760);
    auto x_main_module_1774 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1772, x_main_module_1773);
    auto x_main_module_1775 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1774);
    auto x_main_module_1776 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1757,
        x_main_module_356);
    auto x_main_module_1777 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_355);
    auto x_main_module_1778 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_354);
    auto x_main_module_1779 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_353);
    auto x_main_module_1780 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_352);
    auto x_main_module_1781 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1779);
    auto x_main_module_1782 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1776, x_main_module_1781);
    auto x_main_module_1783 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_65);
    auto x_main_module_1784 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1780, x_main_module_1783);
    auto x_main_module_1785 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_66);
    auto x_main_module_1786 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1784, x_main_module_1785);
    auto x_main_module_1787 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1786);
    auto x_main_module_1788 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1782, x_main_module_1787);
    auto x_main_module_1789 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1777);
    auto x_main_module_1790 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1788, x_main_module_1789);
    auto x_main_module_1791 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1778);
    auto x_main_module_1792 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1790, x_main_module_1791);
    auto x_main_module_1793 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1792);
    auto x_main_module_1794 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,1]}"),
        x_main_module_1793,
        x_main_module_351);
    auto x_main_module_1795 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_350);
    auto x_main_module_1796 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_349);
    auto x_main_module_1797 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_348);
    auto x_main_module_1798 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_347);
    auto x_main_module_1799 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1797);
    auto x_main_module_1800 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1794, x_main_module_1799);
    auto x_main_module_1801 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_63);
    auto x_main_module_1802 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1798, x_main_module_1801);
    auto x_main_module_1803 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_64);
    auto x_main_module_1804 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1802, x_main_module_1803);
    auto x_main_module_1805 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1804);
    auto x_main_module_1806 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1800, x_main_module_1805);
    auto x_main_module_1807 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1795);
    auto x_main_module_1808 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1806, x_main_module_1807);
    auto x_main_module_1809 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1796);
    auto x_main_module_1810 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1808, x_main_module_1809);
    auto x_main_module_1811 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1810);
    auto x_main_module_1812 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1811,
        x_main_module_346);
    auto x_main_module_1813 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_345);
    auto x_main_module_1814 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_344);
    auto x_main_module_1815 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_343);
    auto x_main_module_1816 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_342);
    auto x_main_module_1817 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1815);
    auto x_main_module_1818 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1812, x_main_module_1817);
    auto x_main_module_1819 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_61);
    auto x_main_module_1820 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1816, x_main_module_1819);
    auto x_main_module_1821 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_62);
    auto x_main_module_1822 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1820, x_main_module_1821);
    auto x_main_module_1823 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1822);
    auto x_main_module_1824 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1818, x_main_module_1823);
    auto x_main_module_1825 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1813);
    auto x_main_module_1826 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1824, x_main_module_1825);
    auto x_main_module_1827 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1814);
    auto x_main_module_1828 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1826, x_main_module_1827);
    auto x_main_module_1829 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1828);
    auto x_main_module_1830 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1757,
        x_main_module_341);
    auto x_main_module_1831 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_340);
    auto x_main_module_1832 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_339);
    auto x_main_module_1833 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_338);
    auto x_main_module_1834 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_337);
    auto x_main_module_1835 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1833);
    auto x_main_module_1836 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1830, x_main_module_1835);
    auto x_main_module_1837 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_59);
    auto x_main_module_1838 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1834, x_main_module_1837);
    auto x_main_module_1839 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_60);
    auto x_main_module_1840 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1838, x_main_module_1839);
    auto x_main_module_1841 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1840);
    auto x_main_module_1842 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1836, x_main_module_1841);
    auto x_main_module_1843 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1831);
    auto x_main_module_1844 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1842, x_main_module_1843);
    auto x_main_module_1845 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1832);
    auto x_main_module_1846 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1844, x_main_module_1845);
    auto x_main_module_1847 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1846);
    auto x_main_module_1848 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1847,
        x_main_module_336);
    auto x_main_module_1849 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_335);
    auto x_main_module_1850 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_334);
    auto x_main_module_1851 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_333);
    auto x_main_module_1852 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_332);
    auto x_main_module_1853 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1851);
    auto x_main_module_1854 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1848, x_main_module_1853);
    auto x_main_module_1855 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_57);
    auto x_main_module_1856 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1852, x_main_module_1855);
    auto x_main_module_1857 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_58);
    auto x_main_module_1858 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1856, x_main_module_1857);
    auto x_main_module_1859 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1858);
    auto x_main_module_1860 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1854, x_main_module_1859);
    auto x_main_module_1861 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1849);
    auto x_main_module_1862 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1860, x_main_module_1861);
    auto x_main_module_1863 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1850);
    auto x_main_module_1864 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1862, x_main_module_1863);
    auto x_main_module_1865 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1864);
    auto x_main_module_1866 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,1]}"),
        x_main_module_1865,
        x_main_module_331);
    auto x_main_module_1867 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_330);
    auto x_main_module_1868 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_329);
    auto x_main_module_1869 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_328);
    auto x_main_module_1870 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_327);
    auto x_main_module_1871 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1869);
    auto x_main_module_1872 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1866, x_main_module_1871);
    auto x_main_module_1873 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_55);
    auto x_main_module_1874 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1870, x_main_module_1873);
    auto x_main_module_1875 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_56);
    auto x_main_module_1876 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1874, x_main_module_1875);
    auto x_main_module_1877 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1876);
    auto x_main_module_1878 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1872, x_main_module_1877);
    auto x_main_module_1879 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1867);
    auto x_main_module_1880 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1878, x_main_module_1879);
    auto x_main_module_1881 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1868);
    auto x_main_module_1882 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1880, x_main_module_1881);
    auto x_main_module_1883 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1882);
    auto x_main_module_1884 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1883,
        x_main_module_326);
    auto x_main_module_1885 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_325);
    auto x_main_module_1886 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_324);
    auto x_main_module_1887 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_323);
    auto x_main_module_1888 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_322);
    auto x_main_module_1889 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1887);
    auto x_main_module_1890 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1884, x_main_module_1889);
    auto x_main_module_1891 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_53);
    auto x_main_module_1892 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1888, x_main_module_1891);
    auto x_main_module_1893 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_54);
    auto x_main_module_1894 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1892, x_main_module_1893);
    auto x_main_module_1895 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1894);
    auto x_main_module_1896 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1890, x_main_module_1895);
    auto x_main_module_1897 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1885);
    auto x_main_module_1898 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1896, x_main_module_1897);
    auto x_main_module_1899 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1886);
    auto x_main_module_1900 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1898, x_main_module_1899);
    auto x_main_module_1901 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1900);
    auto x_main_module_1902 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,1]}"),
        x_main_module_1901,
        x_main_module_321);
    auto x_main_module_1903 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_320);
    auto x_main_module_1904 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_319);
    auto x_main_module_1905 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_318);
    auto x_main_module_1906 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_317);
    auto x_main_module_1907 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1905);
    auto x_main_module_1908 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1902, x_main_module_1907);
    auto x_main_module_1909 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_51);
    auto x_main_module_1910 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1906, x_main_module_1909);
    auto x_main_module_1911 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_52);
    auto x_main_module_1912 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1910, x_main_module_1911);
    auto x_main_module_1913 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1912);
    auto x_main_module_1914 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1908, x_main_module_1913);
    auto x_main_module_1915 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1903);
    auto x_main_module_1916 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1914, x_main_module_1915);
    auto x_main_module_1917 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1904);
    auto x_main_module_1918 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1916, x_main_module_1917);
    auto x_main_module_1919 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1918);
    auto x_main_module_1920 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:0,padding:[1,1,1,1],stride:[1,1]}"),
        x_main_module_1757);
    auto x_main_module_1921 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1920,
        x_main_module_316);
    auto x_main_module_1922 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_315);
    auto x_main_module_1923 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_314);
    auto x_main_module_1924 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_313);
    auto x_main_module_1925 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_312);
    auto x_main_module_1926 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1924);
    auto x_main_module_1927 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1921, x_main_module_1926);
    auto x_main_module_1928 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_49);
    auto x_main_module_1929 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1925, x_main_module_1928);
    auto x_main_module_1930 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_50);
    auto x_main_module_1931 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1929, x_main_module_1930);
    auto x_main_module_1932 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1931);
    auto x_main_module_1933 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1927, x_main_module_1932);
    auto x_main_module_1934 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1922);
    auto x_main_module_1935 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1933, x_main_module_1934);
    auto x_main_module_1936 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1923);
    auto x_main_module_1937 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1935, x_main_module_1936);
    auto x_main_module_1938 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1937);
    auto x_main_module_1939 = mmain->add_instruction(migraphx::make_json_op("concat", "{axis:1}"),
                                                     x_main_module_1775,
                                                     x_main_module_1829,
                                                     x_main_module_1919,
                                                     x_main_module_1938);
    auto x_main_module_1940 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1939,
        x_main_module_311);
    auto x_main_module_1941 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_310);
    auto x_main_module_1942 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_309);
    auto x_main_module_1943 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_308);
    auto x_main_module_1944 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_307);
    auto x_main_module_1945 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1943);
    auto x_main_module_1946 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1940, x_main_module_1945);
    auto x_main_module_1947 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_47);
    auto x_main_module_1948 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1944, x_main_module_1947);
    auto x_main_module_1949 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_48);
    auto x_main_module_1950 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1948, x_main_module_1949);
    auto x_main_module_1951 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1950);
    auto x_main_module_1952 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1946, x_main_module_1951);
    auto x_main_module_1953 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1941);
    auto x_main_module_1954 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1952, x_main_module_1953);
    auto x_main_module_1955 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1942);
    auto x_main_module_1956 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1954, x_main_module_1955);
    auto x_main_module_1957 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1956);
    auto x_main_module_1958 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[2,2]}"),
        x_main_module_1957,
        x_main_module_306);
    auto x_main_module_1959 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_305);
    auto x_main_module_1960 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_304);
    auto x_main_module_1961 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_303);
    auto x_main_module_1962 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_302);
    auto x_main_module_1963 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,320,8,8]}"), x_main_module_1961);
    auto x_main_module_1964 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1958, x_main_module_1963);
    auto x_main_module_1965 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[320,1,1]}"), x_main_module_45);
    auto x_main_module_1966 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1962, x_main_module_1965);
    auto x_main_module_1967 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[320,1,1]}"), x_main_module_46);
    auto x_main_module_1968 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1966, x_main_module_1967);
    auto x_main_module_1969 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,320,8,8]}"), x_main_module_1968);
    auto x_main_module_1970 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1964, x_main_module_1969);
    auto x_main_module_1971 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,320,8,8]}"), x_main_module_1959);
    auto x_main_module_1972 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1970, x_main_module_1971);
    auto x_main_module_1973 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,320,8,8]}"), x_main_module_1960);
    auto x_main_module_1974 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1972, x_main_module_1973);
    auto x_main_module_1975 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1974);
    auto x_main_module_1976 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_1939,
        x_main_module_301);
    auto x_main_module_1977 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_300);
    auto x_main_module_1978 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_299);
    auto x_main_module_1979 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_298);
    auto x_main_module_1980 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_297);
    auto x_main_module_1981 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1979);
    auto x_main_module_1982 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1976, x_main_module_1981);
    auto x_main_module_1983 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_43);
    auto x_main_module_1984 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1980, x_main_module_1983);
    auto x_main_module_1985 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_44);
    auto x_main_module_1986 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_1984, x_main_module_1985);
    auto x_main_module_1987 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1986);
    auto x_main_module_1988 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_1982, x_main_module_1987);
    auto x_main_module_1989 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1977);
    auto x_main_module_1990 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_1988, x_main_module_1989);
    auto x_main_module_1991 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1978);
    auto x_main_module_1992 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1990, x_main_module_1991);
    auto x_main_module_1993 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_1992);
    auto x_main_module_1994 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,3,0,3],padding_mode:0,stride:[1,1]}"),
        x_main_module_1993,
        x_main_module_296);
    auto x_main_module_1995 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_295);
    auto x_main_module_1996 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_294);
    auto x_main_module_1997 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_293);
    auto x_main_module_1998 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_292);
    auto x_main_module_1999 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1997);
    auto x_main_module_2000 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_1994, x_main_module_1999);
    auto x_main_module_2001 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_41);
    auto x_main_module_2002 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_1998, x_main_module_2001);
    auto x_main_module_2003 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_42);
    auto x_main_module_2004 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_2002, x_main_module_2003);
    auto x_main_module_2005 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_2004);
    auto x_main_module_2006 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_2000, x_main_module_2005);
    auto x_main_module_2007 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1995);
    auto x_main_module_2008 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_2006, x_main_module_2007);
    auto x_main_module_2009 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_1996);
    auto x_main_module_2010 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2008, x_main_module_2009);
    auto x_main_module_2011 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_2010);
    auto x_main_module_2012 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[3,0,3,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_2011,
        x_main_module_291);
    auto x_main_module_2013 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_290);
    auto x_main_module_2014 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_289);
    auto x_main_module_2015 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_288);
    auto x_main_module_2016 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_287);
    auto x_main_module_2017 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_2015);
    auto x_main_module_2018 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_2012, x_main_module_2017);
    auto x_main_module_2019 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_39);
    auto x_main_module_2020 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2016, x_main_module_2019);
    auto x_main_module_2021 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_40);
    auto x_main_module_2022 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_2020, x_main_module_2021);
    auto x_main_module_2023 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_2022);
    auto x_main_module_2024 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_2018, x_main_module_2023);
    auto x_main_module_2025 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_2013);
    auto x_main_module_2026 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_2024, x_main_module_2025);
    auto x_main_module_2027 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,17,17]}"), x_main_module_2014);
    auto x_main_module_2028 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2026, x_main_module_2027);
    auto x_main_module_2029 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_2028);
    auto x_main_module_2030 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[2,2]}"),
        x_main_module_2029,
        x_main_module_286);
    auto x_main_module_2031 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_285);
    auto x_main_module_2032 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_284);
    auto x_main_module_2033 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_283);
    auto x_main_module_2034 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_282);
    auto x_main_module_2035 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,8,8]}"), x_main_module_2033);
    auto x_main_module_2036 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_2030, x_main_module_2035);
    auto x_main_module_2037 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_37);
    auto x_main_module_2038 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2034, x_main_module_2037);
    auto x_main_module_2039 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_38);
    auto x_main_module_2040 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_2038, x_main_module_2039);
    auto x_main_module_2041 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,8,8]}"), x_main_module_2040);
    auto x_main_module_2042 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_2036, x_main_module_2041);
    auto x_main_module_2043 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,8,8]}"), x_main_module_2031);
    auto x_main_module_2044 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_2042, x_main_module_2043);
    auto x_main_module_2045 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,8,8]}"), x_main_module_2032);
    auto x_main_module_2046 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2044, x_main_module_2045);
    auto x_main_module_2047 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_2046);
    auto x_main_module_2048 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:1,padding:[0,0,0,0],stride:[2,2]}"),
        x_main_module_1939);
    auto x_main_module_2049 = mmain->add_instruction(migraphx::make_json_op("concat", "{axis:1}"),
                                                     x_main_module_1975,
                                                     x_main_module_2047,
                                                     x_main_module_2048);
    auto x_main_module_2050 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_2049,
        x_main_module_281);
    auto x_main_module_2051 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_280);
    auto x_main_module_2052 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_279);
    auto x_main_module_2053 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_278);
    auto x_main_module_2054 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_277);
    auto x_main_module_2055 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,320,8,8]}"), x_main_module_2053);
    auto x_main_module_2056 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_2050, x_main_module_2055);
    auto x_main_module_2057 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[320,1,1]}"), x_main_module_35);
    auto x_main_module_2058 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2054, x_main_module_2057);
    auto x_main_module_2059 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[320,1,1]}"), x_main_module_36);
    auto x_main_module_2060 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_2058, x_main_module_2059);
    auto x_main_module_2061 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,320,8,8]}"), x_main_module_2060);
    auto x_main_module_2062 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_2056, x_main_module_2061);
    auto x_main_module_2063 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,320,8,8]}"), x_main_module_2051);
    auto x_main_module_2064 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_2062, x_main_module_2063);
    auto x_main_module_2065 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,320,8,8]}"), x_main_module_2052);
    auto x_main_module_2066 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2064, x_main_module_2065);
    auto x_main_module_2067 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_2066);
    auto x_main_module_2068 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_2049,
        x_main_module_276);
    auto x_main_module_2069 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_275);
    auto x_main_module_2070 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_274);
    auto x_main_module_2071 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_273);
    auto x_main_module_2072 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_272);
    auto x_main_module_2073 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2071);
    auto x_main_module_2074 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_2068, x_main_module_2073);
    auto x_main_module_2075 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[384,1,1]}"), x_main_module_33);
    auto x_main_module_2076 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2072, x_main_module_2075);
    auto x_main_module_2077 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[384,1,1]}"), x_main_module_34);
    auto x_main_module_2078 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_2076, x_main_module_2077);
    auto x_main_module_2079 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2078);
    auto x_main_module_2080 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_2074, x_main_module_2079);
    auto x_main_module_2081 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2069);
    auto x_main_module_2082 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_2080, x_main_module_2081);
    auto x_main_module_2083 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2070);
    auto x_main_module_2084 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2082, x_main_module_2083);
    auto x_main_module_2085 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_2084);
    auto x_main_module_2086 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,1,0,1],padding_mode:0,stride:[1,1]}"),
        x_main_module_2085,
        x_main_module_271);
    auto x_main_module_2087 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_270);
    auto x_main_module_2088 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_269);
    auto x_main_module_2089 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_268);
    auto x_main_module_2090 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_267);
    auto x_main_module_2091 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2089);
    auto x_main_module_2092 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_2086, x_main_module_2091);
    auto x_main_module_2093 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[384,1,1]}"), x_main_module_31);
    auto x_main_module_2094 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2090, x_main_module_2093);
    auto x_main_module_2095 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[384,1,1]}"), x_main_module_32);
    auto x_main_module_2096 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_2094, x_main_module_2095);
    auto x_main_module_2097 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2096);
    auto x_main_module_2098 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_2092, x_main_module_2097);
    auto x_main_module_2099 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2087);
    auto x_main_module_2100 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_2098, x_main_module_2099);
    auto x_main_module_2101 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2088);
    auto x_main_module_2102 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2100, x_main_module_2101);
    auto x_main_module_2103 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_2102);
    auto x_main_module_2104 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[1,0,1,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_2085,
        x_main_module_266);
    auto x_main_module_2105 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_265);
    auto x_main_module_2106 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_264);
    auto x_main_module_2107 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_263);
    auto x_main_module_2108 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_262);
    auto x_main_module_2109 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2107);
    auto x_main_module_2110 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_2104, x_main_module_2109);
    auto x_main_module_2111 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[384,1,1]}"), x_main_module_29);
    auto x_main_module_2112 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2108, x_main_module_2111);
    auto x_main_module_2113 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[384,1,1]}"), x_main_module_30);
    auto x_main_module_2114 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_2112, x_main_module_2113);
    auto x_main_module_2115 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2114);
    auto x_main_module_2116 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_2110, x_main_module_2115);
    auto x_main_module_2117 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2105);
    auto x_main_module_2118 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_2116, x_main_module_2117);
    auto x_main_module_2119 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2106);
    auto x_main_module_2120 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2118, x_main_module_2119);
    auto x_main_module_2121 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_2120);
    auto x_main_module_2122 = mmain->add_instruction(
        migraphx::make_json_op("concat", "{axis:1}"), x_main_module_2103, x_main_module_2121);
    auto x_main_module_2123 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_2049,
        x_main_module_261);
    auto x_main_module_2124 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_260);
    auto x_main_module_2125 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_259);
    auto x_main_module_2126 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_258);
    auto x_main_module_2127 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_257);
    auto x_main_module_2128 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,448,8,8]}"), x_main_module_2126);
    auto x_main_module_2129 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_2123, x_main_module_2128);
    auto x_main_module_2130 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[448,1,1]}"), x_main_module_27);
    auto x_main_module_2131 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2127, x_main_module_2130);
    auto x_main_module_2132 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[448,1,1]}"), x_main_module_28);
    auto x_main_module_2133 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_2131, x_main_module_2132);
    auto x_main_module_2134 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,448,8,8]}"), x_main_module_2133);
    auto x_main_module_2135 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_2129, x_main_module_2134);
    auto x_main_module_2136 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,448,8,8]}"), x_main_module_2124);
    auto x_main_module_2137 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_2135, x_main_module_2136);
    auto x_main_module_2138 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,448,8,8]}"), x_main_module_2125);
    auto x_main_module_2139 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2137, x_main_module_2138);
    auto x_main_module_2140 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_2139);
    auto x_main_module_2141 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}"),
        x_main_module_2140,
        x_main_module_256);
    auto x_main_module_2142 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_255);
    auto x_main_module_2143 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_254);
    auto x_main_module_2144 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_253);
    auto x_main_module_2145 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_252);
    auto x_main_module_2146 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2144);
    auto x_main_module_2147 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_2141, x_main_module_2146);
    auto x_main_module_2148 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[384,1,1]}"), x_main_module_25);
    auto x_main_module_2149 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2145, x_main_module_2148);
    auto x_main_module_2150 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[384,1,1]}"), x_main_module_26);
    auto x_main_module_2151 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_2149, x_main_module_2150);
    auto x_main_module_2152 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2151);
    auto x_main_module_2153 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_2147, x_main_module_2152);
    auto x_main_module_2154 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2142);
    auto x_main_module_2155 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_2153, x_main_module_2154);
    auto x_main_module_2156 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2143);
    auto x_main_module_2157 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2155, x_main_module_2156);
    auto x_main_module_2158 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_2157);
    auto x_main_module_2159 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,1,0,1],padding_mode:0,stride:[1,1]}"),
        x_main_module_2158,
        x_main_module_251);
    auto x_main_module_2160 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_250);
    auto x_main_module_2161 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_249);
    auto x_main_module_2162 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_248);
    auto x_main_module_2163 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_247);
    auto x_main_module_2164 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2162);
    auto x_main_module_2165 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_2159, x_main_module_2164);
    auto x_main_module_2166 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[384,1,1]}"), x_main_module_23);
    auto x_main_module_2167 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2163, x_main_module_2166);
    auto x_main_module_2168 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[384,1,1]}"), x_main_module_24);
    auto x_main_module_2169 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_2167, x_main_module_2168);
    auto x_main_module_2170 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2169);
    auto x_main_module_2171 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_2165, x_main_module_2170);
    auto x_main_module_2172 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2160);
    auto x_main_module_2173 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_2171, x_main_module_2172);
    auto x_main_module_2174 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2161);
    auto x_main_module_2175 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2173, x_main_module_2174);
    auto x_main_module_2176 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_2175);
    auto x_main_module_2177 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[1,0,1,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_2158,
        x_main_module_246);
    auto x_main_module_2178 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_245);
    auto x_main_module_2179 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_244);
    auto x_main_module_2180 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_243);
    auto x_main_module_2181 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_242);
    auto x_main_module_2182 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2180);
    auto x_main_module_2183 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_2177, x_main_module_2182);
    auto x_main_module_2184 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[384,1,1]}"), x_main_module_21);
    auto x_main_module_2185 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2181, x_main_module_2184);
    auto x_main_module_2186 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[384,1,1]}"), x_main_module_22);
    auto x_main_module_2187 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_2185, x_main_module_2186);
    auto x_main_module_2188 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2187);
    auto x_main_module_2189 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_2183, x_main_module_2188);
    auto x_main_module_2190 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2178);
    auto x_main_module_2191 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_2189, x_main_module_2190);
    auto x_main_module_2192 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2179);
    auto x_main_module_2193 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2191, x_main_module_2192);
    auto x_main_module_2194 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_2193);
    auto x_main_module_2195 = mmain->add_instruction(
        migraphx::make_json_op("concat", "{axis:1}"), x_main_module_2176, x_main_module_2194);
    auto x_main_module_2196 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:0,padding:[1,1,1,1],stride:[1,1]}"),
        x_main_module_2049);
    auto x_main_module_2197 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_2196,
        x_main_module_241);
    auto x_main_module_2198 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_240);
    auto x_main_module_2199 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_239);
    auto x_main_module_2200 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_238);
    auto x_main_module_2201 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_237);
    auto x_main_module_2202 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,8,8]}"), x_main_module_2200);
    auto x_main_module_2203 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_2197, x_main_module_2202);
    auto x_main_module_2204 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_19);
    auto x_main_module_2205 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2201, x_main_module_2204);
    auto x_main_module_2206 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_20);
    auto x_main_module_2207 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_2205, x_main_module_2206);
    auto x_main_module_2208 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,8,8]}"), x_main_module_2207);
    auto x_main_module_2209 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_2203, x_main_module_2208);
    auto x_main_module_2210 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,8,8]}"), x_main_module_2198);
    auto x_main_module_2211 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_2209, x_main_module_2210);
    auto x_main_module_2212 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,8,8]}"), x_main_module_2199);
    auto x_main_module_2213 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2211, x_main_module_2212);
    auto x_main_module_2214 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_2213);
    auto x_main_module_2215 = mmain->add_instruction(migraphx::make_json_op("concat", "{axis:1}"),
                                                     x_main_module_2067,
                                                     x_main_module_2122,
                                                     x_main_module_2195,
                                                     x_main_module_2214);
    auto x_main_module_2216 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_2215,
        x_main_module_236);
    auto x_main_module_2217 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_235);
    auto x_main_module_2218 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_234);
    auto x_main_module_2219 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_233);
    auto x_main_module_2220 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_232);
    auto x_main_module_2221 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,320,8,8]}"), x_main_module_2219);
    auto x_main_module_2222 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_2216, x_main_module_2221);
    auto x_main_module_2223 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[320,1,1]}"), x_main_module_17);
    auto x_main_module_2224 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2220, x_main_module_2223);
    auto x_main_module_2225 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[320,1,1]}"), x_main_module_18);
    auto x_main_module_2226 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_2224, x_main_module_2225);
    auto x_main_module_2227 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,320,8,8]}"), x_main_module_2226);
    auto x_main_module_2228 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_2222, x_main_module_2227);
    auto x_main_module_2229 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,320,8,8]}"), x_main_module_2217);
    auto x_main_module_2230 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_2228, x_main_module_2229);
    auto x_main_module_2231 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,320,8,8]}"), x_main_module_2218);
    auto x_main_module_2232 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2230, x_main_module_2231);
    auto x_main_module_2233 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_2232);
    auto x_main_module_2234 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_2215,
        x_main_module_231);
    auto x_main_module_2235 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_230);
    auto x_main_module_2236 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_229);
    auto x_main_module_2237 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_228);
    auto x_main_module_2238 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_227);
    auto x_main_module_2239 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2237);
    auto x_main_module_2240 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_2234, x_main_module_2239);
    auto x_main_module_2241 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[384,1,1]}"), x_main_module_15);
    auto x_main_module_2242 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2238, x_main_module_2241);
    auto x_main_module_2243 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[384,1,1]}"), x_main_module_16);
    auto x_main_module_2244 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_2242, x_main_module_2243);
    auto x_main_module_2245 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2244);
    auto x_main_module_2246 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_2240, x_main_module_2245);
    auto x_main_module_2247 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2235);
    auto x_main_module_2248 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_2246, x_main_module_2247);
    auto x_main_module_2249 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2236);
    auto x_main_module_2250 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2248, x_main_module_2249);
    auto x_main_module_2251 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_2250);
    auto x_main_module_2252 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,1,0,1],padding_mode:0,stride:[1,1]}"),
        x_main_module_2251,
        x_main_module_226);
    auto x_main_module_2253 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_225);
    auto x_main_module_2254 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_224);
    auto x_main_module_2255 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_223);
    auto x_main_module_2256 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_222);
    auto x_main_module_2257 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2255);
    auto x_main_module_2258 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_2252, x_main_module_2257);
    auto x_main_module_2259 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[384,1,1]}"), x_main_module_13);
    auto x_main_module_2260 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2256, x_main_module_2259);
    auto x_main_module_2261 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[384,1,1]}"), x_main_module_14);
    auto x_main_module_2262 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_2260, x_main_module_2261);
    auto x_main_module_2263 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2262);
    auto x_main_module_2264 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_2258, x_main_module_2263);
    auto x_main_module_2265 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2253);
    auto x_main_module_2266 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_2264, x_main_module_2265);
    auto x_main_module_2267 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2254);
    auto x_main_module_2268 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2266, x_main_module_2267);
    auto x_main_module_2269 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_2268);
    auto x_main_module_2270 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[1,0,1,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_2251,
        x_main_module_221);
    auto x_main_module_2271 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_220);
    auto x_main_module_2272 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_219);
    auto x_main_module_2273 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_218);
    auto x_main_module_2274 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_217);
    auto x_main_module_2275 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2273);
    auto x_main_module_2276 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_2270, x_main_module_2275);
    auto x_main_module_2277 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[384,1,1]}"), x_main_module_11);
    auto x_main_module_2278 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2274, x_main_module_2277);
    auto x_main_module_2279 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[384,1,1]}"), x_main_module_12);
    auto x_main_module_2280 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_2278, x_main_module_2279);
    auto x_main_module_2281 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2280);
    auto x_main_module_2282 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_2276, x_main_module_2281);
    auto x_main_module_2283 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2271);
    auto x_main_module_2284 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_2282, x_main_module_2283);
    auto x_main_module_2285 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2272);
    auto x_main_module_2286 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2284, x_main_module_2285);
    auto x_main_module_2287 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_2286);
    auto x_main_module_2288 = mmain->add_instruction(
        migraphx::make_json_op("concat", "{axis:1}"), x_main_module_2269, x_main_module_2287);
    auto x_main_module_2289 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_2215,
        x_main_module_216);
    auto x_main_module_2290 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_215);
    auto x_main_module_2291 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_214);
    auto x_main_module_2292 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_213);
    auto x_main_module_2293 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_212);
    auto x_main_module_2294 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,448,8,8]}"), x_main_module_2292);
    auto x_main_module_2295 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_2289, x_main_module_2294);
    auto x_main_module_2296 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[448,1,1]}"), x_main_module_9);
    auto x_main_module_2297 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2293, x_main_module_2296);
    auto x_main_module_2298 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[448,1,1]}"), x_main_module_10);
    auto x_main_module_2299 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_2297, x_main_module_2298);
    auto x_main_module_2300 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,448,8,8]}"), x_main_module_2299);
    auto x_main_module_2301 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_2295, x_main_module_2300);
    auto x_main_module_2302 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,448,8,8]}"), x_main_module_2290);
    auto x_main_module_2303 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_2301, x_main_module_2302);
    auto x_main_module_2304 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,448,8,8]}"), x_main_module_2291);
    auto x_main_module_2305 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2303, x_main_module_2304);
    auto x_main_module_2306 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_2305);
    auto x_main_module_2307 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,1]}"),
        x_main_module_2306,
        x_main_module_211);
    auto x_main_module_2308 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_210);
    auto x_main_module_2309 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_209);
    auto x_main_module_2310 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_208);
    auto x_main_module_2311 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_207);
    auto x_main_module_2312 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2310);
    auto x_main_module_2313 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_2307, x_main_module_2312);
    auto x_main_module_2314 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[384,1,1]}"), x_main_module_7);
    auto x_main_module_2315 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2311, x_main_module_2314);
    auto x_main_module_2316 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[384,1,1]}"), x_main_module_8);
    auto x_main_module_2317 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_2315, x_main_module_2316);
    auto x_main_module_2318 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2317);
    auto x_main_module_2319 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_2313, x_main_module_2318);
    auto x_main_module_2320 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2308);
    auto x_main_module_2321 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_2319, x_main_module_2320);
    auto x_main_module_2322 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2309);
    auto x_main_module_2323 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2321, x_main_module_2322);
    auto x_main_module_2324 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_2323);
    auto x_main_module_2325 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,1,0,1],padding_mode:0,stride:[1,1]}"),
        x_main_module_2324,
        x_main_module_206);
    auto x_main_module_2326 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_205);
    auto x_main_module_2327 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_204);
    auto x_main_module_2328 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_203);
    auto x_main_module_2329 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_202);
    auto x_main_module_2330 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2328);
    auto x_main_module_2331 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_2325, x_main_module_2330);
    auto x_main_module_2332 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[384,1,1]}"), x_main_module_5);
    auto x_main_module_2333 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2329, x_main_module_2332);
    auto x_main_module_2334 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[384,1,1]}"), x_main_module_6);
    auto x_main_module_2335 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_2333, x_main_module_2334);
    auto x_main_module_2336 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2335);
    auto x_main_module_2337 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_2331, x_main_module_2336);
    auto x_main_module_2338 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2326);
    auto x_main_module_2339 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_2337, x_main_module_2338);
    auto x_main_module_2340 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2327);
    auto x_main_module_2341 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2339, x_main_module_2340);
    auto x_main_module_2342 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_2341);
    auto x_main_module_2343 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[1,0,1,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_2324,
        x_main_module_201);
    auto x_main_module_2344 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_200);
    auto x_main_module_2345 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_199);
    auto x_main_module_2346 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_198);
    auto x_main_module_2347 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_197);
    auto x_main_module_2348 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2346);
    auto x_main_module_2349 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_2343, x_main_module_2348);
    auto x_main_module_2350 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[384,1,1]}"), x_main_module_3);
    auto x_main_module_2351 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2347, x_main_module_2350);
    auto x_main_module_2352 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[384,1,1]}"), x_main_module_4);
    auto x_main_module_2353 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_2351, x_main_module_2352);
    auto x_main_module_2354 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2353);
    auto x_main_module_2355 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_2349, x_main_module_2354);
    auto x_main_module_2356 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2344);
    auto x_main_module_2357 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_2355, x_main_module_2356);
    auto x_main_module_2358 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,384,8,8]}"), x_main_module_2345);
    auto x_main_module_2359 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2357, x_main_module_2358);
    auto x_main_module_2360 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_2359);
    auto x_main_module_2361 = mmain->add_instruction(
        migraphx::make_json_op("concat", "{axis:1}"), x_main_module_2342, x_main_module_2360);
    auto x_main_module_2362 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:0,padding:[1,1,1,1],stride:[1,1]}"),
        x_main_module_2215);
    auto x_main_module_2363 = mmain->add_instruction(
        migraphx::make_json_op(
            "convolution",
            "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,1]}"),
        x_main_module_2362,
        x_main_module_196);
    auto x_main_module_2364 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_195);
    auto x_main_module_2365 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_194);
    auto x_main_module_2366 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_193);
    auto x_main_module_2367 = mmain->add_instruction(
        migraphx::make_json_op("unsqueeze", "{axes:[1,2],steps:[]}"), x_main_module_192);
    auto x_main_module_2368 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,8,8]}"), x_main_module_2366);
    auto x_main_module_2369 =
        mmain->add_instruction(migraphx::make_op("sub"), x_main_module_2363, x_main_module_2368);
    auto x_main_module_2370 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_1);
    auto x_main_module_2371 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2367, x_main_module_2370);
    auto x_main_module_2372 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[192,1,1]}"), x_main_module_2);
    auto x_main_module_2373 =
        mmain->add_instruction(migraphx::make_op("pow"), x_main_module_2371, x_main_module_2372);
    auto x_main_module_2374 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,8,8]}"), x_main_module_2373);
    auto x_main_module_2375 =
        mmain->add_instruction(migraphx::make_op("div"), x_main_module_2369, x_main_module_2374);
    auto x_main_module_2376 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,8,8]}"), x_main_module_2364);
    auto x_main_module_2377 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_2375, x_main_module_2376);
    auto x_main_module_2378 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,192,8,8]}"), x_main_module_2365);
    auto x_main_module_2379 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2377, x_main_module_2378);
    auto x_main_module_2380 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_2379);
    auto x_main_module_2381 = mmain->add_instruction(migraphx::make_json_op("concat", "{axis:1}"),
                                                     x_main_module_2233,
                                                     x_main_module_2288,
                                                     x_main_module_2361,
                                                     x_main_module_2380);
    auto x_main_module_2382 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[8,8],lp_order:2,mode:0,padding:[0,0,0,0],stride:[8,8]}"),
        x_main_module_2381);
    auto x_main_module_2383 =
        mmain->add_instruction(migraphx::make_op("identity"), x_main_module_2382);
    auto x_main_module_2384 =
        mmain->add_instruction(migraphx::make_json_op("flatten", "{axis:1}"), x_main_module_2383);
    auto x_main_module_2385 = mmain->add_instruction(
        migraphx::make_json_op("transpose", "{permutation:[1,0]}"), x_main_module_191);
    auto x_main_module_2386 =
        mmain->add_instruction(migraphx::make_op("dot"), x_main_module_2384, x_main_module_2385);
    auto x_main_module_2387 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1000]}"), x_main_module_190);
    auto x_main_module_2388 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1000]}"), x_main_module_0);
    auto x_main_module_2389 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_2387, x_main_module_2388);
    auto x_main_module_2390 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_2386, x_main_module_2389);
    mmain->add_return({x_main_module_2390});

    return p;
}
} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx
