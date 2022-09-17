
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
    auto x_0                   = mmain->add_parameter(
        "0", migraphx::shape{migraphx::shape::float_type, {batch, 3, 224, 224}});
    auto x_main_module_2 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1000}}, 1));
    auto x_main_module_3 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1000, 2048}}, 2));
    auto x_main_module_4 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 3)));
    auto x_main_module_5 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 4));
    auto x_main_module_6 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 5));
    auto x_main_module_7  = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 6)));
    auto x_main_module_8  = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {2048, 512, 1, 1}}, 7));
    auto x_main_module_9  = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 8)));
    auto x_main_module_10 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 9));
    auto x_main_module_11 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 10));
    auto x_main_module_12 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 11)));
    auto x_main_module_13 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 512, 3, 3}}, 12));
    auto x_main_module_14 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 13)));
    auto x_main_module_15 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 14));
    auto x_main_module_16 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 15));
    auto x_main_module_17 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 16)));
    auto x_main_module_18 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 2048, 1, 1}}, 17));
    auto x_main_module_19 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 18)));
    auto x_main_module_20 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 19));
    auto x_main_module_21 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 20));
    auto x_main_module_22 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 21)));
    auto x_main_module_23 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {2048, 512, 1, 1}}, 22));
    auto x_main_module_24 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 23)));
    auto x_main_module_25 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 24));
    auto x_main_module_26 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 25));
    auto x_main_module_27 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 26)));
    auto x_main_module_28 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 512, 3, 3}}, 27));
    auto x_main_module_29 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 28)));
    auto x_main_module_30 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 29));
    auto x_main_module_31 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 30));
    auto x_main_module_32 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 31)));
    auto x_main_module_33 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 2048, 1, 1}}, 32));
    auto x_main_module_34 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 33)));
    auto x_main_module_35 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 34));
    auto x_main_module_36 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 35));
    auto x_main_module_37 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 36)));
    auto x_main_module_38 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {2048, 1024, 1, 1}}, 37));
    auto x_main_module_39 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 38)));
    auto x_main_module_40 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 39));
    auto x_main_module_41 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 40));
    auto x_main_module_42 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 41)));
    auto x_main_module_43 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {2048, 512, 1, 1}}, 42));
    auto x_main_module_44 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 43)));
    auto x_main_module_45 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 44));
    auto x_main_module_46 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 45));
    auto x_main_module_47 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 46)));
    auto x_main_module_48 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 512, 3, 3}}, 47));
    auto x_main_module_49 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 48)));
    auto x_main_module_50 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 49));
    auto x_main_module_51 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 50));
    auto x_main_module_52 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 51)));
    auto x_main_module_53 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 1024, 1, 1}}, 52));
    auto x_main_module_54 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 53)));
    auto x_main_module_55 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 54));
    auto x_main_module_56 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 55));
    auto x_main_module_57 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 56));
    auto x_main_module_58 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 256, 1, 1}}, 57));
    auto x_main_module_59 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 58)));
    auto x_main_module_60 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 59));
    auto x_main_module_61 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 60));
    auto x_main_module_62 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 61)));
    auto x_main_module_63 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 256, 3, 3}}, 62));
    auto x_main_module_64 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 63)));
    auto x_main_module_65 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 64));
    auto x_main_module_66 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 65));
    auto x_main_module_67 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 66)));
    auto x_main_module_68 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 1024, 1, 1}}, 67));
    auto x_main_module_69 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 68)));
    auto x_main_module_70 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 69));
    auto x_main_module_71 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 70));
    auto x_main_module_72 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 71));
    auto x_main_module_73 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 256, 1, 1}}, 72));
    auto x_main_module_74 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 73)));
    auto x_main_module_75 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 74));
    auto x_main_module_76 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 75));
    auto x_main_module_77 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 76)));
    auto x_main_module_78 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 256, 3, 3}}, 77));
    auto x_main_module_79 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 78)));
    auto x_main_module_80 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 79));
    auto x_main_module_81 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 80));
    auto x_main_module_82 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 81)));
    auto x_main_module_83 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 1024, 1, 1}}, 82));
    auto x_main_module_84 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 83)));
    auto x_main_module_85 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 84));
    auto x_main_module_86 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 85));
    auto x_main_module_87 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 86));
    auto x_main_module_88 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 256, 1, 1}}, 87));
    auto x_main_module_89 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 88)));
    auto x_main_module_90 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 89));
    auto x_main_module_91 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 90));
    auto x_main_module_92 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 91)));
    auto x_main_module_93 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 256, 3, 3}}, 92));
    auto x_main_module_94 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 93)));
    auto x_main_module_95 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 94));
    auto x_main_module_96 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 95));
    auto x_main_module_97  = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 96)));
    auto x_main_module_98  = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 1024, 1, 1}}, 97));
    auto x_main_module_99  = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 98)));
    auto x_main_module_100 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 99));
    auto x_main_module_101 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 100));
    auto x_main_module_102 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 101));
    auto x_main_module_103 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 256, 1, 1}}, 102));
    auto x_main_module_104 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 103)));
    auto x_main_module_105 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 104));
    auto x_main_module_106 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 105));
    auto x_main_module_107 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 106)));
    auto x_main_module_108 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 256, 3, 3}}, 107));
    auto x_main_module_109 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 108)));
    auto x_main_module_110 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 109));
    auto x_main_module_111 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 110));
    auto x_main_module_112 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 111)));
    auto x_main_module_113 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 1024, 1, 1}}, 112));
    auto x_main_module_114 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 113)));
    auto x_main_module_115 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 114));
    auto x_main_module_116 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 115));
    auto x_main_module_117 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 116));
    auto x_main_module_118 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 256, 1, 1}}, 117));
    auto x_main_module_119 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 118)));
    auto x_main_module_120 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 119));
    auto x_main_module_121 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 120));
    auto x_main_module_122 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 121)));
    auto x_main_module_123 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 256, 3, 3}}, 122));
    auto x_main_module_124 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 123)));
    auto x_main_module_125 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 124));
    auto x_main_module_126 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 125));
    auto x_main_module_127 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 126)));
    auto x_main_module_128 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 1024, 1, 1}}, 127));
    auto x_main_module_129 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 128)));
    auto x_main_module_130 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 129));
    auto x_main_module_131 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 130));
    auto x_main_module_132 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 131));
    auto x_main_module_133 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 512, 1, 1}}, 132));
    auto x_main_module_134 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 133)));
    auto x_main_module_135 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 134));
    auto x_main_module_136 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 135));
    auto x_main_module_137 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 136));
    auto x_main_module_138 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 256, 1, 1}}, 137));
    auto x_main_module_139 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 138)));
    auto x_main_module_140 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 139));
    auto x_main_module_141 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 140));
    auto x_main_module_142 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 141)));
    auto x_main_module_143 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 256, 3, 3}}, 142));
    auto x_main_module_144 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 143)));
    auto x_main_module_145 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 144));
    auto x_main_module_146 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 145));
    auto x_main_module_147 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 146)));
    auto x_main_module_148 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 512, 1, 1}}, 147));
    auto x_main_module_149 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 148)));
    auto x_main_module_150 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 149));
    auto x_main_module_151 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 150));
    auto x_main_module_152 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 151));
    auto x_main_module_153 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 128, 1, 1}}, 152));
    auto x_main_module_154 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 153)));
    auto x_main_module_155 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 154));
    auto x_main_module_156 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 155));
    auto x_main_module_157 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 156)));
    auto x_main_module_158 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 3, 3}}, 157));
    auto x_main_module_159 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 158)));
    auto x_main_module_160 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 159));
    auto x_main_module_161 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 160));
    auto x_main_module_162 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 161)));
    auto x_main_module_163 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 512, 1, 1}}, 162));
    auto x_main_module_164 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 163)));
    auto x_main_module_165 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 164));
    auto x_main_module_166 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 165));
    auto x_main_module_167 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 166));
    auto x_main_module_168 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 128, 1, 1}}, 167));
    auto x_main_module_169 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 168)));
    auto x_main_module_170 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 169));
    auto x_main_module_171 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 170));
    auto x_main_module_172 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 171)));
    auto x_main_module_173 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 3, 3}}, 172));
    auto x_main_module_174 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 173)));
    auto x_main_module_175 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 174));
    auto x_main_module_176 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 175));
    auto x_main_module_177 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 176)));
    auto x_main_module_178 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 512, 1, 1}}, 177));
    auto x_main_module_179 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 178)));
    auto x_main_module_180 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 179));
    auto x_main_module_181 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 180));
    auto x_main_module_182 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 181));
    auto x_main_module_183 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 128, 1, 1}}, 182));
    auto x_main_module_184 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 183)));
    auto x_main_module_185 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 184));
    auto x_main_module_186 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 185));
    auto x_main_module_187 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 186)));
    auto x_main_module_188 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 3, 3}}, 187));
    auto x_main_module_189 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 188)));
    auto x_main_module_190 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 189));
    auto x_main_module_191 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 190));
    auto x_main_module_192 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 191)));
    auto x_main_module_193 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 512, 1, 1}}, 192));
    auto x_main_module_194 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 193)));
    auto x_main_module_195 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 194));
    auto x_main_module_196 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 195));
    auto x_main_module_197 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 196));
    auto x_main_module_198 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 256, 1, 1}}, 197));
    auto x_main_module_199 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 198)));
    auto x_main_module_200 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 199));
    auto x_main_module_201 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 200));
    auto x_main_module_202 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 201));
    auto x_main_module_203 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 128, 1, 1}}, 202));
    auto x_main_module_204 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 203)));
    auto x_main_module_205 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 204));
    auto x_main_module_206 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 205));
    auto x_main_module_207 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 206)));
    auto x_main_module_208 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 3, 3}}, 207));
    auto x_main_module_209 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 208)));
    auto x_main_module_210 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 209));
    auto x_main_module_211 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 210));
    auto x_main_module_212 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 211)));
    auto x_main_module_213 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 256, 1, 1}}, 212));
    auto x_main_module_214 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 213)));
    auto x_main_module_215 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 214));
    auto x_main_module_216 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 215));
    auto x_main_module_217 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 216));
    auto x_main_module_218 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 64, 1, 1}}, 217));
    auto x_main_module_219 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 218)));
    auto x_main_module_220 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 219));
    auto x_main_module_221 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 220));
    auto x_main_module_222 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 221)));
    auto x_main_module_223 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 64, 3, 3}}, 222));
    auto x_main_module_224 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 223)));
    auto x_main_module_225 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 224));
    auto x_main_module_226 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 225));
    auto x_main_module_227 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 226)));
    auto x_main_module_228 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 256, 1, 1}}, 227));
    auto x_main_module_229 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 228)));
    auto x_main_module_230 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 229));
    auto x_main_module_231 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 230));
    auto x_main_module_232 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 231));
    auto x_main_module_233 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 64, 1, 1}}, 232));
    auto x_main_module_234 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 233)));
    auto x_main_module_235 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 234));
    auto x_main_module_236 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 235));
    auto x_main_module_237 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 236)));
    auto x_main_module_238 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 64, 3, 3}}, 237));
    auto x_main_module_239 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 238)));
    auto x_main_module_240 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 239));
    auto x_main_module_241 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 240));
    auto x_main_module_242 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 241)));
    auto x_main_module_243 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 256, 1, 1}}, 242));
    auto x_main_module_244 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 243)));
    auto x_main_module_245 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 244));
    auto x_main_module_246 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 245));
    auto x_main_module_247 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 246));
    auto x_main_module_248 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 64, 1, 1}}, 247));
    auto x_main_module_249 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 248)));
    auto x_main_module_250 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 249));
    auto x_main_module_251 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 250));
    auto x_main_module_252 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 251));
    auto x_main_module_253 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 64, 1, 1}}, 252));
    auto x_main_module_254 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 253)));
    auto x_main_module_255 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 254));
    auto x_main_module_256 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 255));
    auto x_main_module_257 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 256)));
    auto x_main_module_258 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 64, 3, 3}}, 257));
    auto x_main_module_259 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 258)));
    auto x_main_module_260 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 259));
    auto x_main_module_261 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 260));
    auto x_main_module_262 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 261)));
    auto x_main_module_263 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 64, 1, 1}}, 262));
    auto x_main_module_264 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 263)));
    auto x_main_module_265 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 264));
    auto x_main_module_266 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 265));
    auto x_main_module_267 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 266)));
    auto x_main_module_268 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 3, 7, 7}}, 267));
    auto x_main_module_269 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[3,3,3,3],padding_mode:0,stride:[2,"
                               "2],use_dynamic_same_auto_pad:0}"),
        x_0,
        x_main_module_268);
    auto x_main_module_270 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_269,
        x_main_module_267,
        x_main_module_266,
        x_main_module_265,
        x_main_module_264);
    auto x_main_module_271 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_270);
    auto x_main_module_272 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[3,3],lp_order:2,mode:1,padding:[1,1,1,1],stride:[2,2]}"),
        x_main_module_271);
    auto x_main_module_273 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_272,
        x_main_module_263);
    auto x_main_module_274 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_273,
        x_main_module_262,
        x_main_module_261,
        x_main_module_260,
        x_main_module_259);
    auto x_main_module_275 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_274);
    auto x_main_module_276 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_275,
        x_main_module_258);
    auto x_main_module_277 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_276,
        x_main_module_257,
        x_main_module_256,
        x_main_module_255,
        x_main_module_254);
    auto x_main_module_278 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_277);
    auto x_main_module_279 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_278,
        x_main_module_253);
    auto x_main_module_280 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_279,
        x_main_module_252,
        x_main_module_251,
        x_main_module_250,
        x_main_module_249);
    auto x_main_module_281 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_272,
        x_main_module_248);
    auto x_main_module_282 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_281,
        x_main_module_247,
        x_main_module_246,
        x_main_module_245,
        x_main_module_244);
    auto x_main_module_283 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_280, x_main_module_282);
    auto x_main_module_284 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_283);
    auto x_main_module_285 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_284,
        x_main_module_243);
    auto x_main_module_286 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_285,
        x_main_module_242,
        x_main_module_241,
        x_main_module_240,
        x_main_module_239);
    auto x_main_module_287 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_286);
    auto x_main_module_288 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_287,
        x_main_module_238);
    auto x_main_module_289 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_288,
        x_main_module_237,
        x_main_module_236,
        x_main_module_235,
        x_main_module_234);
    auto x_main_module_290 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_289);
    auto x_main_module_291 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_290,
        x_main_module_233);
    auto x_main_module_292 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_291,
        x_main_module_232,
        x_main_module_231,
        x_main_module_230,
        x_main_module_229);
    auto x_main_module_293 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_292, x_main_module_284);
    auto x_main_module_294 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_293);
    auto x_main_module_295 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_294,
        x_main_module_228);
    auto x_main_module_296 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_295,
        x_main_module_227,
        x_main_module_226,
        x_main_module_225,
        x_main_module_224);
    auto x_main_module_297 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_296);
    auto x_main_module_298 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_297,
        x_main_module_223);
    auto x_main_module_299 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_298,
        x_main_module_222,
        x_main_module_221,
        x_main_module_220,
        x_main_module_219);
    auto x_main_module_300 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_299);
    auto x_main_module_301 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_300,
        x_main_module_218);
    auto x_main_module_302 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_301,
        x_main_module_217,
        x_main_module_216,
        x_main_module_215,
        x_main_module_214);
    auto x_main_module_303 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_302, x_main_module_294);
    auto x_main_module_304 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_303);
    auto x_main_module_305 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_304,
        x_main_module_213);
    auto x_main_module_306 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_305,
        x_main_module_212,
        x_main_module_211,
        x_main_module_210,
        x_main_module_209);
    auto x_main_module_307 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_306);
    auto x_main_module_308 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[2,"
                               "2],use_dynamic_same_auto_pad:0}"),
        x_main_module_307,
        x_main_module_208);
    auto x_main_module_309 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_308,
        x_main_module_207,
        x_main_module_206,
        x_main_module_205,
        x_main_module_204);
    auto x_main_module_310 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_309);
    auto x_main_module_311 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_310,
        x_main_module_203);
    auto x_main_module_312 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_311,
        x_main_module_202,
        x_main_module_201,
        x_main_module_200,
        x_main_module_199);
    auto x_main_module_313 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[2,"
                               "2],use_dynamic_same_auto_pad:0}"),
        x_main_module_304,
        x_main_module_198);
    auto x_main_module_314 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_313,
        x_main_module_197,
        x_main_module_196,
        x_main_module_195,
        x_main_module_194);
    auto x_main_module_315 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_312, x_main_module_314);
    auto x_main_module_316 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_315);
    auto x_main_module_317 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_316,
        x_main_module_193);
    auto x_main_module_318 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_317,
        x_main_module_192,
        x_main_module_191,
        x_main_module_190,
        x_main_module_189);
    auto x_main_module_319 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_318);
    auto x_main_module_320 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_319,
        x_main_module_188);
    auto x_main_module_321 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_320,
        x_main_module_187,
        x_main_module_186,
        x_main_module_185,
        x_main_module_184);
    auto x_main_module_322 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_321);
    auto x_main_module_323 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_322,
        x_main_module_183);
    auto x_main_module_324 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_323,
        x_main_module_182,
        x_main_module_181,
        x_main_module_180,
        x_main_module_179);
    auto x_main_module_325 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_324, x_main_module_316);
    auto x_main_module_326 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_325);
    auto x_main_module_327 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_326,
        x_main_module_178);
    auto x_main_module_328 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_327,
        x_main_module_177,
        x_main_module_176,
        x_main_module_175,
        x_main_module_174);
    auto x_main_module_329 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_328);
    auto x_main_module_330 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_329,
        x_main_module_173);
    auto x_main_module_331 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_330,
        x_main_module_172,
        x_main_module_171,
        x_main_module_170,
        x_main_module_169);
    auto x_main_module_332 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_331);
    auto x_main_module_333 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_332,
        x_main_module_168);
    auto x_main_module_334 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_333,
        x_main_module_167,
        x_main_module_166,
        x_main_module_165,
        x_main_module_164);
    auto x_main_module_335 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_334, x_main_module_326);
    auto x_main_module_336 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_335);
    auto x_main_module_337 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_336,
        x_main_module_163);
    auto x_main_module_338 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_337,
        x_main_module_162,
        x_main_module_161,
        x_main_module_160,
        x_main_module_159);
    auto x_main_module_339 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_338);
    auto x_main_module_340 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_339,
        x_main_module_158);
    auto x_main_module_341 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_340,
        x_main_module_157,
        x_main_module_156,
        x_main_module_155,
        x_main_module_154);
    auto x_main_module_342 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_341);
    auto x_main_module_343 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_342,
        x_main_module_153);
    auto x_main_module_344 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_343,
        x_main_module_152,
        x_main_module_151,
        x_main_module_150,
        x_main_module_149);
    auto x_main_module_345 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_344, x_main_module_336);
    auto x_main_module_346 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_345);
    auto x_main_module_347 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_346,
        x_main_module_148);
    auto x_main_module_348 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_347,
        x_main_module_147,
        x_main_module_146,
        x_main_module_145,
        x_main_module_144);
    auto x_main_module_349 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_348);
    auto x_main_module_350 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[2,"
                               "2],use_dynamic_same_auto_pad:0}"),
        x_main_module_349,
        x_main_module_143);
    auto x_main_module_351 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_350,
        x_main_module_142,
        x_main_module_141,
        x_main_module_140,
        x_main_module_139);
    auto x_main_module_352 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_351);
    auto x_main_module_353 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_352,
        x_main_module_138);
    auto x_main_module_354 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_353,
        x_main_module_137,
        x_main_module_136,
        x_main_module_135,
        x_main_module_134);
    auto x_main_module_355 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[2,"
                               "2],use_dynamic_same_auto_pad:0}"),
        x_main_module_346,
        x_main_module_133);
    auto x_main_module_356 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_355,
        x_main_module_132,
        x_main_module_131,
        x_main_module_130,
        x_main_module_129);
    auto x_main_module_357 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_354, x_main_module_356);
    auto x_main_module_358 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_357);
    auto x_main_module_359 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_358,
        x_main_module_128);
    auto x_main_module_360 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_359,
        x_main_module_127,
        x_main_module_126,
        x_main_module_125,
        x_main_module_124);
    auto x_main_module_361 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_360);
    auto x_main_module_362 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_361,
        x_main_module_123);
    auto x_main_module_363 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_362,
        x_main_module_122,
        x_main_module_121,
        x_main_module_120,
        x_main_module_119);
    auto x_main_module_364 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_363);
    auto x_main_module_365 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_364,
        x_main_module_118);
    auto x_main_module_366 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_365,
        x_main_module_117,
        x_main_module_116,
        x_main_module_115,
        x_main_module_114);
    auto x_main_module_367 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_366, x_main_module_358);
    auto x_main_module_368 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_367);
    auto x_main_module_369 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_368,
        x_main_module_113);
    auto x_main_module_370 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_369,
        x_main_module_112,
        x_main_module_111,
        x_main_module_110,
        x_main_module_109);
    auto x_main_module_371 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_370);
    auto x_main_module_372 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_371,
        x_main_module_108);
    auto x_main_module_373 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_372,
        x_main_module_107,
        x_main_module_106,
        x_main_module_105,
        x_main_module_104);
    auto x_main_module_374 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_373);
    auto x_main_module_375 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_374,
        x_main_module_103);
    auto x_main_module_376 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_375,
        x_main_module_102,
        x_main_module_101,
        x_main_module_100,
        x_main_module_99);
    auto x_main_module_377 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_376, x_main_module_368);
    auto x_main_module_378 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_377);
    auto x_main_module_379 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_378,
        x_main_module_98);
    auto x_main_module_380 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_379,
        x_main_module_97,
        x_main_module_96,
        x_main_module_95,
        x_main_module_94);
    auto x_main_module_381 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_380);
    auto x_main_module_382 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_381,
        x_main_module_93);
    auto x_main_module_383 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_382,
        x_main_module_92,
        x_main_module_91,
        x_main_module_90,
        x_main_module_89);
    auto x_main_module_384 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_383);
    auto x_main_module_385 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_384,
        x_main_module_88);
    auto x_main_module_386 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_385,
        x_main_module_87,
        x_main_module_86,
        x_main_module_85,
        x_main_module_84);
    auto x_main_module_387 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_386, x_main_module_378);
    auto x_main_module_388 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_387);
    auto x_main_module_389 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_388,
        x_main_module_83);
    auto x_main_module_390 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_389,
        x_main_module_82,
        x_main_module_81,
        x_main_module_80,
        x_main_module_79);
    auto x_main_module_391 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_390);
    auto x_main_module_392 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_391,
        x_main_module_78);
    auto x_main_module_393 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_392,
        x_main_module_77,
        x_main_module_76,
        x_main_module_75,
        x_main_module_74);
    auto x_main_module_394 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_393);
    auto x_main_module_395 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_394,
        x_main_module_73);
    auto x_main_module_396 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_395,
        x_main_module_72,
        x_main_module_71,
        x_main_module_70,
        x_main_module_69);
    auto x_main_module_397 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_396, x_main_module_388);
    auto x_main_module_398 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_397);
    auto x_main_module_399 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_398,
        x_main_module_68);
    auto x_main_module_400 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_399,
        x_main_module_67,
        x_main_module_66,
        x_main_module_65,
        x_main_module_64);
    auto x_main_module_401 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_400);
    auto x_main_module_402 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_401,
        x_main_module_63);
    auto x_main_module_403 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_402,
        x_main_module_62,
        x_main_module_61,
        x_main_module_60,
        x_main_module_59);
    auto x_main_module_404 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_403);
    auto x_main_module_405 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_404,
        x_main_module_58);
    auto x_main_module_406 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_405,
        x_main_module_57,
        x_main_module_56,
        x_main_module_55,
        x_main_module_54);
    auto x_main_module_407 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_406, x_main_module_398);
    auto x_main_module_408 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_407);
    auto x_main_module_409 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_408,
        x_main_module_53);
    auto x_main_module_410 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_409,
        x_main_module_52,
        x_main_module_51,
        x_main_module_50,
        x_main_module_49);
    auto x_main_module_411 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_410);
    auto x_main_module_412 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[2,"
                               "2],use_dynamic_same_auto_pad:0}"),
        x_main_module_411,
        x_main_module_48);
    auto x_main_module_413 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_412,
        x_main_module_47,
        x_main_module_46,
        x_main_module_45,
        x_main_module_44);
    auto x_main_module_414 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_413);
    auto x_main_module_415 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_414,
        x_main_module_43);
    auto x_main_module_416 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_415,
        x_main_module_42,
        x_main_module_41,
        x_main_module_40,
        x_main_module_39);
    auto x_main_module_417 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[2,"
                               "2],use_dynamic_same_auto_pad:0}"),
        x_main_module_408,
        x_main_module_38);
    auto x_main_module_418 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_417,
        x_main_module_37,
        x_main_module_36,
        x_main_module_35,
        x_main_module_34);
    auto x_main_module_419 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_416, x_main_module_418);
    auto x_main_module_420 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_419);
    auto x_main_module_421 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_420,
        x_main_module_33);
    auto x_main_module_422 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_421,
        x_main_module_32,
        x_main_module_31,
        x_main_module_30,
        x_main_module_29);
    auto x_main_module_423 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_422);
    auto x_main_module_424 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_423,
        x_main_module_28);
    auto x_main_module_425 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_424,
        x_main_module_27,
        x_main_module_26,
        x_main_module_25,
        x_main_module_24);
    auto x_main_module_426 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_425);
    auto x_main_module_427 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_426,
        x_main_module_23);
    auto x_main_module_428 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_427,
        x_main_module_22,
        x_main_module_21,
        x_main_module_20,
        x_main_module_19);
    auto x_main_module_429 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_428, x_main_module_420);
    auto x_main_module_430 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_429);
    auto x_main_module_431 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_430,
        x_main_module_18);
    auto x_main_module_432 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_431,
        x_main_module_17,
        x_main_module_16,
        x_main_module_15,
        x_main_module_14);
    auto x_main_module_433 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_432);
    auto x_main_module_434 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[1,1,1,1],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_433,
        x_main_module_13);
    auto x_main_module_435 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_434,
        x_main_module_12,
        x_main_module_11,
        x_main_module_10,
        x_main_module_9);
    auto x_main_module_436 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_435);
    auto x_main_module_437 = mmain->add_instruction(
        migraphx::make_json_op("convolution",
                               "{dilation:[1,1],group:1,padding:[0,0,0,0],padding_mode:0,stride:[1,"
                               "1],use_dynamic_same_auto_pad:0}"),
        x_main_module_436,
        x_main_module_8);
    auto x_main_module_438 = mmain->add_instruction(
        migraphx::make_json_op(
            "batch_norm_inference",
            "{bn_mode:1,epsilon:9.999999747378752e-06,momentum:0.8999999761581421}"),
        x_main_module_437,
        x_main_module_7,
        x_main_module_6,
        x_main_module_5,
        x_main_module_4);
    auto x_main_module_439 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_438, x_main_module_430);
    auto x_main_module_440 = mmain->add_instruction(migraphx::make_op("relu"), x_main_module_439);
    auto x_main_module_441 = mmain->add_instruction(
        migraphx::make_json_op(
            "pooling",
            "{ceil_mode:0,lengths:[7,7],lp_order:2,mode:0,padding:[0,0,0,0],stride:[1,1]}"),
        x_main_module_440);
    auto x_main_module_442 =
        mmain->add_instruction(migraphx::make_json_op("flatten", "{axis:1}"), x_main_module_441);
    auto x_main_module_443 = mmain->add_instruction(
        migraphx::make_json_op("transpose", "{permutation:[1,0]}"), x_main_module_3);
    auto x_main_module_444 =
        mmain->add_instruction(migraphx::make_op("dot"), x_main_module_442, x_main_module_443);
    auto x_main_module_445 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1000]}"), x_main_module_2);
    auto x_main_module_446 = mmain->add_instruction(
        migraphx::make_json_op("multibroadcast", "{out_lens:[1,1000]}"), x_main_module_0);
    auto x_main_module_447 =
        mmain->add_instruction(migraphx::make_op("mul"), x_main_module_445, x_main_module_446);
    auto x_main_module_448 =
        mmain->add_instruction(migraphx::make_op("add"), x_main_module_444, x_main_module_447);
    mmain->add_return({x_main_module_448});

    return p;
}
} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx
