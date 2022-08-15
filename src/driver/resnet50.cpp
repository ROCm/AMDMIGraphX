
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
migraphx::program resnet50(unsigned batch) // NOLINT(readability-function-size)
{
    migraphx::program p;
    migraphx::module_ref mmain = p.get_main_module();
    auto _x_main_module_0      = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1}}, 0)));
    auto _x_0                  = mmain->add_parameter(
        "0", migraphx::shape{migraphx::shape::float_type, {batch, 3, 224, 224}});
    auto _x_main_module_2 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1000}}, 1));
    auto _x_main_module_3 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1000, 2048}}, 2));
    auto _x_main_module_4 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 3)));
    auto _x_main_module_5 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 4));
    auto _x_main_module_6 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 5));
    auto _x_main_module_7  = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 6)));
    auto _x_main_module_8  = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {2048, 512, 1, 1}}, 7));
    auto _x_main_module_9  = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 8)));
    auto _x_main_module_10 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 9));
    auto _x_main_module_11 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 10));
    auto _x_main_module_12 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 11)));
    auto _x_main_module_13 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 512, 3, 3}}, 12));
    auto _x_main_module_14 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 13)));
    auto _x_main_module_15 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 14));
    auto _x_main_module_16 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 15));
    auto _x_main_module_17 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 16)));
    auto _x_main_module_18 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 2048, 1, 1}}, 17));
    auto _x_main_module_19 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 18)));
    auto _x_main_module_20 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 19));
    auto _x_main_module_21 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 20));
    auto _x_main_module_22 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 21)));
    auto _x_main_module_23 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {2048, 512, 1, 1}}, 22));
    auto _x_main_module_24 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 23)));
    auto _x_main_module_25 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 24));
    auto _x_main_module_26 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 25));
    auto _x_main_module_27 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 26)));
    auto _x_main_module_28 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 512, 3, 3}}, 27));
    auto _x_main_module_29 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 28)));
    auto _x_main_module_30 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 29));
    auto _x_main_module_31 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 30));
    auto _x_main_module_32 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 31)));
    auto _x_main_module_33 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 2048, 1, 1}}, 32));
    auto _x_main_module_34 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 33)));
    auto _x_main_module_35 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 34));
    auto _x_main_module_36 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 35));
    auto _x_main_module_37 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 36)));
    auto _x_main_module_38 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {2048, 1024, 1, 1}}, 37));
    auto _x_main_module_39 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 38)));
    auto _x_main_module_40 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 39));
    auto _x_main_module_41 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 40));
    auto _x_main_module_42 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 41)));
    auto _x_main_module_43 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {2048, 512, 1, 1}}, 42));
    auto _x_main_module_44 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 43)));
    auto _x_main_module_45 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 44));
    auto _x_main_module_46 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 45));
    auto _x_main_module_47 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 46)));
    auto _x_main_module_48 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 512, 3, 3}}, 47));
    auto _x_main_module_49 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 48)));
    auto _x_main_module_50 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 49));
    auto _x_main_module_51 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 50));
    auto _x_main_module_52 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 51)));
    auto _x_main_module_53 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 1024, 1, 1}}, 52));
    auto _x_main_module_54 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 53)));
    auto _x_main_module_55 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 54));
    auto _x_main_module_56 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 55));
    auto _x_main_module_57 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 56));
    auto _x_main_module_58 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 256, 1, 1}}, 57));
    auto _x_main_module_59 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 58)));
    auto _x_main_module_60 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 59));
    auto _x_main_module_61 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 60));
    auto _x_main_module_62 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 61)));
    auto _x_main_module_63 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 256, 3, 3}}, 62));
    auto _x_main_module_64 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 63)));
    auto _x_main_module_65 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 64));
    auto _x_main_module_66 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 65));
    auto _x_main_module_67 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 66)));
    auto _x_main_module_68 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 1024, 1, 1}}, 67));
    auto _x_main_module_69 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 68)));
    auto _x_main_module_70 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 69));
    auto _x_main_module_71 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 70));
    auto _x_main_module_72 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 71));
    auto _x_main_module_73 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 256, 1, 1}}, 72));
    auto _x_main_module_74 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 73)));
    auto _x_main_module_75 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 74));
    auto _x_main_module_76 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 75));
    auto _x_main_module_77 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 76)));
    auto _x_main_module_78 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 256, 3, 3}}, 77));
    auto _x_main_module_79 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 78)));
    auto _x_main_module_80 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 79));
    auto _x_main_module_81 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 80));
    auto _x_main_module_82 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 81)));
    auto _x_main_module_83 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 1024, 1, 1}}, 82));
    auto _x_main_module_84 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 83)));
    auto _x_main_module_85 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 84));
    auto _x_main_module_86 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 85));
    auto _x_main_module_87 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 86));
    auto _x_main_module_88 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 256, 1, 1}}, 87));
    auto _x_main_module_89 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 88)));
    auto _x_main_module_90 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 89));
    auto _x_main_module_91 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 90));
    auto _x_main_module_92 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 91)));
    auto _x_main_module_93 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 256, 3, 3}}, 92));
    auto _x_main_module_94 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 93)));
    auto _x_main_module_95 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 94));
    auto _x_main_module_96 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 95));
    auto _x_main_module_97  = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 96)));
    auto _x_main_module_98  = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 1024, 1, 1}}, 97));
    auto _x_main_module_99  = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 98)));
    auto _x_main_module_100 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 99));
    auto _x_main_module_101 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 100));
    auto _x_main_module_102 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 101));
    auto _x_main_module_103 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 256, 1, 1}}, 102));
    auto _x_main_module_104 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 103)));
    auto _x_main_module_105 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 104));
    auto _x_main_module_106 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 105));
    auto _x_main_module_107 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 106)));
    auto _x_main_module_108 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 256, 3, 3}}, 107));
    auto _x_main_module_109 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 108)));
    auto _x_main_module_110 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 109));
    auto _x_main_module_111 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 110));
    auto _x_main_module_112 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 111)));
    auto _x_main_module_113 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 1024, 1, 1}}, 112));
    auto _x_main_module_114 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 113)));
    auto _x_main_module_115 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 114));
    auto _x_main_module_116 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 115));
    auto _x_main_module_117 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 116));
    auto _x_main_module_118 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 256, 1, 1}}, 117));
    auto _x_main_module_119 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 118)));
    auto _x_main_module_120 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 119));
    auto _x_main_module_121 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 120));
    auto _x_main_module_122 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 121)));
    auto _x_main_module_123 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 256, 3, 3}}, 122));
    auto _x_main_module_124 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 123)));
    auto _x_main_module_125 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 124));
    auto _x_main_module_126 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 125));
    auto _x_main_module_127 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 126)));
    auto _x_main_module_128 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 1024, 1, 1}}, 127));
    auto _x_main_module_129 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 128)));
    auto _x_main_module_130 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 129));
    auto _x_main_module_131 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 130));
    auto _x_main_module_132 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 131));
    auto _x_main_module_133 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 512, 1, 1}}, 132));
    auto _x_main_module_134 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 133)));
    auto _x_main_module_135 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 134));
    auto _x_main_module_136 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 135));
    auto _x_main_module_137 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 136));
    auto _x_main_module_138 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 256, 1, 1}}, 137));
    auto _x_main_module_139 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 138)));
    auto _x_main_module_140 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 139));
    auto _x_main_module_141 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 140));
    auto _x_main_module_142 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 141)));
    auto _x_main_module_143 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 256, 3, 3}}, 142));
    auto _x_main_module_144 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 143)));
    auto _x_main_module_145 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 144));
    auto _x_main_module_146 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 145));
    auto _x_main_module_147 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 146)));
    auto _x_main_module_148 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 512, 1, 1}}, 147));
    auto _x_main_module_149 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 148)));
    auto _x_main_module_150 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 149));
    auto _x_main_module_151 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 150));
    auto _x_main_module_152 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 151));
    auto _x_main_module_153 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 128, 1, 1}}, 152));
    auto _x_main_module_154 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 153)));
    auto _x_main_module_155 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 154));
    auto _x_main_module_156 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 155));
    auto _x_main_module_157 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 156)));
    auto _x_main_module_158 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 3, 3}}, 157));
    auto _x_main_module_159 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 158)));
    auto _x_main_module_160 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 159));
    auto _x_main_module_161 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 160));
    auto _x_main_module_162 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 161)));
    auto _x_main_module_163 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 512, 1, 1}}, 162));
    auto _x_main_module_164 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 163)));
    auto _x_main_module_165 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 164));
    auto _x_main_module_166 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 165));
    auto _x_main_module_167 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 166));
    auto _x_main_module_168 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 128, 1, 1}}, 167));
    auto _x_main_module_169 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 168)));
    auto _x_main_module_170 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 169));
    auto _x_main_module_171 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 170));
    auto _x_main_module_172 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 171)));
    auto _x_main_module_173 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 3, 3}}, 172));
    auto _x_main_module_174 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 173)));
    auto _x_main_module_175 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 174));
    auto _x_main_module_176 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 175));
    auto _x_main_module_177 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 176)));
    auto _x_main_module_178 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 512, 1, 1}}, 177));
    auto _x_main_module_179 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 178)));
    auto _x_main_module_180 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 179));
    auto _x_main_module_181 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 180));
    auto _x_main_module_182 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 181));
    auto _x_main_module_183 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 128, 1, 1}}, 182));
    auto _x_main_module_184 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 183)));
    auto _x_main_module_185 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 184));
    auto _x_main_module_186 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 185));
    auto _x_main_module_187 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 186)));
    auto _x_main_module_188 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 3, 3}}, 187));
    auto _x_main_module_189 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 188)));
    auto _x_main_module_190 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 189));
    auto _x_main_module_191 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 190));
    auto _x_main_module_192 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 191)));
    auto _x_main_module_193 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 512, 1, 1}}, 192));
    auto _x_main_module_194 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 193)));
    auto _x_main_module_195 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 194));
    auto _x_main_module_196 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 195));
    auto _x_main_module_197 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 196));
    auto _x_main_module_198 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 256, 1, 1}}, 197));
    auto _x_main_module_199 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 198)));
    auto _x_main_module_200 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 199));
    auto _x_main_module_201 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 200));
    auto _x_main_module_202 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 201));
    auto _x_main_module_203 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 128, 1, 1}}, 202));
    auto _x_main_module_204 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 203)));
    auto _x_main_module_205 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 204));
    auto _x_main_module_206 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 205));
    auto _x_main_module_207 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 206)));
    auto _x_main_module_208 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 3, 3}}, 207));
    auto _x_main_module_209 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 208)));
    auto _x_main_module_210 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 209));
    auto _x_main_module_211 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 210));
    auto _x_main_module_212 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 211)));
    auto _x_main_module_213 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 256, 1, 1}}, 212));
    auto _x_main_module_214 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 213)));
    auto _x_main_module_215 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 214));
    auto _x_main_module_216 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 215));
    auto _x_main_module_217 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 216));
    auto _x_main_module_218 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 64, 1, 1}}, 217));
    auto _x_main_module_219 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 218)));
    auto _x_main_module_220 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 219));
    auto _x_main_module_221 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 220));
    auto _x_main_module_222 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 221)));
    auto _x_main_module_223 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 64, 3, 3}}, 222));
    auto _x_main_module_224 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 223)));
    auto _x_main_module_225 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 224));
    auto _x_main_module_226 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 225));
    auto _x_main_module_227 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 226)));
    auto _x_main_module_228 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 256, 1, 1}}, 227));
    auto _x_main_module_229 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 228)));
    auto _x_main_module_230 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 229));
    auto _x_main_module_231 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 230));
    auto _x_main_module_232 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 231));
    auto _x_main_module_233 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 64, 1, 1}}, 232));
    auto _x_main_module_234 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 233)));
    auto _x_main_module_235 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 234));
    auto _x_main_module_236 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 235));
    auto _x_main_module_237 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 236)));
    auto _x_main_module_238 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 64, 3, 3}}, 237));
    auto _x_main_module_239 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 238)));
    auto _x_main_module_240 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 239));
    auto _x_main_module_241 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 240));
    auto _x_main_module_242 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 241)));
    auto _x_main_module_243 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 256, 1, 1}}, 242));
    auto _x_main_module_244 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 243)));
    auto _x_main_module_245 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 244));
    auto _x_main_module_246 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 245));
    auto _x_main_module_247 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 246));
    auto _x_main_module_248 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 64, 1, 1}}, 247));
    auto _x_main_module_249 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 248)));
    auto _x_main_module_250 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 249));
    auto _x_main_module_251 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 250));
    auto _x_main_module_252 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 251));
    auto _x_main_module_253 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 64, 1, 1}}, 252));
    auto _x_main_module_254 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 253)));
    auto _x_main_module_255 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 254));
    auto _x_main_module_256 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 255));
    auto _x_main_module_257 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 256)));
    auto _x_main_module_258 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 64, 3, 3}}, 257));
    auto _x_main_module_259 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 258)));
    auto _x_main_module_260 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 259));
    auto _x_main_module_261 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 260));
    auto _x_main_module_262 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 261)));
    auto _x_main_module_263 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 64, 1, 1}}, 262));
    auto _x_main_module_264 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 263)));
    auto _x_main_module_265 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 264));
    auto _x_main_module_266 = mmain->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 265));
    auto _x_main_module_267 = mmain->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 266)));
    auto _x_main_module_268 = mmain->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 3, 7, 7}}, 267));
    auto _x_main_module_269 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[3,3,3,3],\"padding_"
                              "mode\":0,\"stride\":[2,2],\"use_dynamic_same_auto_pad\":0}")),
        _x_0,
        _x_main_module_268);
    auto _x_main_module_270 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_269,
        _x_main_module_267,
        _x_main_module_266,
        _x_main_module_265,
        _x_main_module_264);
    auto _x_main_module_271 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_270);
    auto _x_main_module_272 = mmain->add_instruction(
        migraphx::make_op(
            "pooling",
            migraphx::from_json_string("{\"ceil_mode\":0,\"lengths\":[3,3],\"lp_order\":2,\"mode\":"
                                       "1,\"padding\":[1,1,1,1],\"stride\":[2,2]}")),
        _x_main_module_271);
    auto _x_main_module_273 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_272,
        _x_main_module_263);
    auto _x_main_module_274 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_273,
        _x_main_module_262,
        _x_main_module_261,
        _x_main_module_260,
        _x_main_module_259);
    auto _x_main_module_275 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_274);
    auto _x_main_module_276 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[1,1,1,1],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_275,
        _x_main_module_258);
    auto _x_main_module_277 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_276,
        _x_main_module_257,
        _x_main_module_256,
        _x_main_module_255,
        _x_main_module_254);
    auto _x_main_module_278 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_277);
    auto _x_main_module_279 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_278,
        _x_main_module_253);
    auto _x_main_module_280 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_279,
        _x_main_module_252,
        _x_main_module_251,
        _x_main_module_250,
        _x_main_module_249);
    auto _x_main_module_281 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_272,
        _x_main_module_248);
    auto _x_main_module_282 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_281,
        _x_main_module_247,
        _x_main_module_246,
        _x_main_module_245,
        _x_main_module_244);
    auto _x_main_module_283 =
        mmain->add_instruction(migraphx::make_op("add"), _x_main_module_280, _x_main_module_282);
    auto _x_main_module_284 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_283);
    auto _x_main_module_285 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_284,
        _x_main_module_243);
    auto _x_main_module_286 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_285,
        _x_main_module_242,
        _x_main_module_241,
        _x_main_module_240,
        _x_main_module_239);
    auto _x_main_module_287 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_286);
    auto _x_main_module_288 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[1,1,1,1],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_287,
        _x_main_module_238);
    auto _x_main_module_289 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_288,
        _x_main_module_237,
        _x_main_module_236,
        _x_main_module_235,
        _x_main_module_234);
    auto _x_main_module_290 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_289);
    auto _x_main_module_291 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_290,
        _x_main_module_233);
    auto _x_main_module_292 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_291,
        _x_main_module_232,
        _x_main_module_231,
        _x_main_module_230,
        _x_main_module_229);
    auto _x_main_module_293 =
        mmain->add_instruction(migraphx::make_op("add"), _x_main_module_292, _x_main_module_284);
    auto _x_main_module_294 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_293);
    auto _x_main_module_295 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_294,
        _x_main_module_228);
    auto _x_main_module_296 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_295,
        _x_main_module_227,
        _x_main_module_226,
        _x_main_module_225,
        _x_main_module_224);
    auto _x_main_module_297 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_296);
    auto _x_main_module_298 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[1,1,1,1],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_297,
        _x_main_module_223);
    auto _x_main_module_299 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_298,
        _x_main_module_222,
        _x_main_module_221,
        _x_main_module_220,
        _x_main_module_219);
    auto _x_main_module_300 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_299);
    auto _x_main_module_301 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_300,
        _x_main_module_218);
    auto _x_main_module_302 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_301,
        _x_main_module_217,
        _x_main_module_216,
        _x_main_module_215,
        _x_main_module_214);
    auto _x_main_module_303 =
        mmain->add_instruction(migraphx::make_op("add"), _x_main_module_302, _x_main_module_294);
    auto _x_main_module_304 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_303);
    auto _x_main_module_305 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_304,
        _x_main_module_213);
    auto _x_main_module_306 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_305,
        _x_main_module_212,
        _x_main_module_211,
        _x_main_module_210,
        _x_main_module_209);
    auto _x_main_module_307 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_306);
    auto _x_main_module_308 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[1,1,1,1],\"padding_"
                              "mode\":0,\"stride\":[2,2],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_307,
        _x_main_module_208);
    auto _x_main_module_309 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_308,
        _x_main_module_207,
        _x_main_module_206,
        _x_main_module_205,
        _x_main_module_204);
    auto _x_main_module_310 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_309);
    auto _x_main_module_311 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_310,
        _x_main_module_203);
    auto _x_main_module_312 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_311,
        _x_main_module_202,
        _x_main_module_201,
        _x_main_module_200,
        _x_main_module_199);
    auto _x_main_module_313 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[2,2],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_304,
        _x_main_module_198);
    auto _x_main_module_314 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_313,
        _x_main_module_197,
        _x_main_module_196,
        _x_main_module_195,
        _x_main_module_194);
    auto _x_main_module_315 =
        mmain->add_instruction(migraphx::make_op("add"), _x_main_module_312, _x_main_module_314);
    auto _x_main_module_316 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_315);
    auto _x_main_module_317 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_316,
        _x_main_module_193);
    auto _x_main_module_318 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_317,
        _x_main_module_192,
        _x_main_module_191,
        _x_main_module_190,
        _x_main_module_189);
    auto _x_main_module_319 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_318);
    auto _x_main_module_320 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[1,1,1,1],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_319,
        _x_main_module_188);
    auto _x_main_module_321 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_320,
        _x_main_module_187,
        _x_main_module_186,
        _x_main_module_185,
        _x_main_module_184);
    auto _x_main_module_322 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_321);
    auto _x_main_module_323 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_322,
        _x_main_module_183);
    auto _x_main_module_324 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_323,
        _x_main_module_182,
        _x_main_module_181,
        _x_main_module_180,
        _x_main_module_179);
    auto _x_main_module_325 =
        mmain->add_instruction(migraphx::make_op("add"), _x_main_module_324, _x_main_module_316);
    auto _x_main_module_326 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_325);
    auto _x_main_module_327 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_326,
        _x_main_module_178);
    auto _x_main_module_328 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_327,
        _x_main_module_177,
        _x_main_module_176,
        _x_main_module_175,
        _x_main_module_174);
    auto _x_main_module_329 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_328);
    auto _x_main_module_330 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[1,1,1,1],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_329,
        _x_main_module_173);
    auto _x_main_module_331 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_330,
        _x_main_module_172,
        _x_main_module_171,
        _x_main_module_170,
        _x_main_module_169);
    auto _x_main_module_332 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_331);
    auto _x_main_module_333 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_332,
        _x_main_module_168);
    auto _x_main_module_334 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_333,
        _x_main_module_167,
        _x_main_module_166,
        _x_main_module_165,
        _x_main_module_164);
    auto _x_main_module_335 =
        mmain->add_instruction(migraphx::make_op("add"), _x_main_module_334, _x_main_module_326);
    auto _x_main_module_336 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_335);
    auto _x_main_module_337 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_336,
        _x_main_module_163);
    auto _x_main_module_338 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_337,
        _x_main_module_162,
        _x_main_module_161,
        _x_main_module_160,
        _x_main_module_159);
    auto _x_main_module_339 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_338);
    auto _x_main_module_340 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[1,1,1,1],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_339,
        _x_main_module_158);
    auto _x_main_module_341 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_340,
        _x_main_module_157,
        _x_main_module_156,
        _x_main_module_155,
        _x_main_module_154);
    auto _x_main_module_342 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_341);
    auto _x_main_module_343 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_342,
        _x_main_module_153);
    auto _x_main_module_344 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_343,
        _x_main_module_152,
        _x_main_module_151,
        _x_main_module_150,
        _x_main_module_149);
    auto _x_main_module_345 =
        mmain->add_instruction(migraphx::make_op("add"), _x_main_module_344, _x_main_module_336);
    auto _x_main_module_346 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_345);
    auto _x_main_module_347 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_346,
        _x_main_module_148);
    auto _x_main_module_348 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_347,
        _x_main_module_147,
        _x_main_module_146,
        _x_main_module_145,
        _x_main_module_144);
    auto _x_main_module_349 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_348);
    auto _x_main_module_350 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[1,1,1,1],\"padding_"
                              "mode\":0,\"stride\":[2,2],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_349,
        _x_main_module_143);
    auto _x_main_module_351 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_350,
        _x_main_module_142,
        _x_main_module_141,
        _x_main_module_140,
        _x_main_module_139);
    auto _x_main_module_352 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_351);
    auto _x_main_module_353 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_352,
        _x_main_module_138);
    auto _x_main_module_354 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_353,
        _x_main_module_137,
        _x_main_module_136,
        _x_main_module_135,
        _x_main_module_134);
    auto _x_main_module_355 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[2,2],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_346,
        _x_main_module_133);
    auto _x_main_module_356 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_355,
        _x_main_module_132,
        _x_main_module_131,
        _x_main_module_130,
        _x_main_module_129);
    auto _x_main_module_357 =
        mmain->add_instruction(migraphx::make_op("add"), _x_main_module_354, _x_main_module_356);
    auto _x_main_module_358 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_357);
    auto _x_main_module_359 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_358,
        _x_main_module_128);
    auto _x_main_module_360 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_359,
        _x_main_module_127,
        _x_main_module_126,
        _x_main_module_125,
        _x_main_module_124);
    auto _x_main_module_361 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_360);
    auto _x_main_module_362 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[1,1,1,1],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_361,
        _x_main_module_123);
    auto _x_main_module_363 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_362,
        _x_main_module_122,
        _x_main_module_121,
        _x_main_module_120,
        _x_main_module_119);
    auto _x_main_module_364 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_363);
    auto _x_main_module_365 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_364,
        _x_main_module_118);
    auto _x_main_module_366 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_365,
        _x_main_module_117,
        _x_main_module_116,
        _x_main_module_115,
        _x_main_module_114);
    auto _x_main_module_367 =
        mmain->add_instruction(migraphx::make_op("add"), _x_main_module_366, _x_main_module_358);
    auto _x_main_module_368 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_367);
    auto _x_main_module_369 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_368,
        _x_main_module_113);
    auto _x_main_module_370 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_369,
        _x_main_module_112,
        _x_main_module_111,
        _x_main_module_110,
        _x_main_module_109);
    auto _x_main_module_371 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_370);
    auto _x_main_module_372 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[1,1,1,1],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_371,
        _x_main_module_108);
    auto _x_main_module_373 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_372,
        _x_main_module_107,
        _x_main_module_106,
        _x_main_module_105,
        _x_main_module_104);
    auto _x_main_module_374 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_373);
    auto _x_main_module_375 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_374,
        _x_main_module_103);
    auto _x_main_module_376 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_375,
        _x_main_module_102,
        _x_main_module_101,
        _x_main_module_100,
        _x_main_module_99);
    auto _x_main_module_377 =
        mmain->add_instruction(migraphx::make_op("add"), _x_main_module_376, _x_main_module_368);
    auto _x_main_module_378 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_377);
    auto _x_main_module_379 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_378,
        _x_main_module_98);
    auto _x_main_module_380 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_379,
        _x_main_module_97,
        _x_main_module_96,
        _x_main_module_95,
        _x_main_module_94);
    auto _x_main_module_381 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_380);
    auto _x_main_module_382 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[1,1,1,1],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_381,
        _x_main_module_93);
    auto _x_main_module_383 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_382,
        _x_main_module_92,
        _x_main_module_91,
        _x_main_module_90,
        _x_main_module_89);
    auto _x_main_module_384 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_383);
    auto _x_main_module_385 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_384,
        _x_main_module_88);
    auto _x_main_module_386 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_385,
        _x_main_module_87,
        _x_main_module_86,
        _x_main_module_85,
        _x_main_module_84);
    auto _x_main_module_387 =
        mmain->add_instruction(migraphx::make_op("add"), _x_main_module_386, _x_main_module_378);
    auto _x_main_module_388 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_387);
    auto _x_main_module_389 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_388,
        _x_main_module_83);
    auto _x_main_module_390 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_389,
        _x_main_module_82,
        _x_main_module_81,
        _x_main_module_80,
        _x_main_module_79);
    auto _x_main_module_391 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_390);
    auto _x_main_module_392 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[1,1,1,1],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_391,
        _x_main_module_78);
    auto _x_main_module_393 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_392,
        _x_main_module_77,
        _x_main_module_76,
        _x_main_module_75,
        _x_main_module_74);
    auto _x_main_module_394 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_393);
    auto _x_main_module_395 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_394,
        _x_main_module_73);
    auto _x_main_module_396 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_395,
        _x_main_module_72,
        _x_main_module_71,
        _x_main_module_70,
        _x_main_module_69);
    auto _x_main_module_397 =
        mmain->add_instruction(migraphx::make_op("add"), _x_main_module_396, _x_main_module_388);
    auto _x_main_module_398 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_397);
    auto _x_main_module_399 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_398,
        _x_main_module_68);
    auto _x_main_module_400 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_399,
        _x_main_module_67,
        _x_main_module_66,
        _x_main_module_65,
        _x_main_module_64);
    auto _x_main_module_401 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_400);
    auto _x_main_module_402 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[1,1,1,1],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_401,
        _x_main_module_63);
    auto _x_main_module_403 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_402,
        _x_main_module_62,
        _x_main_module_61,
        _x_main_module_60,
        _x_main_module_59);
    auto _x_main_module_404 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_403);
    auto _x_main_module_405 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_404,
        _x_main_module_58);
    auto _x_main_module_406 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_405,
        _x_main_module_57,
        _x_main_module_56,
        _x_main_module_55,
        _x_main_module_54);
    auto _x_main_module_407 =
        mmain->add_instruction(migraphx::make_op("add"), _x_main_module_406, _x_main_module_398);
    auto _x_main_module_408 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_407);
    auto _x_main_module_409 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_408,
        _x_main_module_53);
    auto _x_main_module_410 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_409,
        _x_main_module_52,
        _x_main_module_51,
        _x_main_module_50,
        _x_main_module_49);
    auto _x_main_module_411 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_410);
    auto _x_main_module_412 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[1,1,1,1],\"padding_"
                              "mode\":0,\"stride\":[2,2],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_411,
        _x_main_module_48);
    auto _x_main_module_413 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_412,
        _x_main_module_47,
        _x_main_module_46,
        _x_main_module_45,
        _x_main_module_44);
    auto _x_main_module_414 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_413);
    auto _x_main_module_415 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_414,
        _x_main_module_43);
    auto _x_main_module_416 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_415,
        _x_main_module_42,
        _x_main_module_41,
        _x_main_module_40,
        _x_main_module_39);
    auto _x_main_module_417 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[2,2],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_408,
        _x_main_module_38);
    auto _x_main_module_418 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_417,
        _x_main_module_37,
        _x_main_module_36,
        _x_main_module_35,
        _x_main_module_34);
    auto _x_main_module_419 =
        mmain->add_instruction(migraphx::make_op("add"), _x_main_module_416, _x_main_module_418);
    auto _x_main_module_420 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_419);
    auto _x_main_module_421 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_420,
        _x_main_module_33);
    auto _x_main_module_422 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_421,
        _x_main_module_32,
        _x_main_module_31,
        _x_main_module_30,
        _x_main_module_29);
    auto _x_main_module_423 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_422);
    auto _x_main_module_424 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[1,1,1,1],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_423,
        _x_main_module_28);
    auto _x_main_module_425 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_424,
        _x_main_module_27,
        _x_main_module_26,
        _x_main_module_25,
        _x_main_module_24);
    auto _x_main_module_426 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_425);
    auto _x_main_module_427 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_426,
        _x_main_module_23);
    auto _x_main_module_428 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_427,
        _x_main_module_22,
        _x_main_module_21,
        _x_main_module_20,
        _x_main_module_19);
    auto _x_main_module_429 =
        mmain->add_instruction(migraphx::make_op("add"), _x_main_module_428, _x_main_module_420);
    auto _x_main_module_430 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_429);
    auto _x_main_module_431 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_430,
        _x_main_module_18);
    auto _x_main_module_432 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_431,
        _x_main_module_17,
        _x_main_module_16,
        _x_main_module_15,
        _x_main_module_14);
    auto _x_main_module_433 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_432);
    auto _x_main_module_434 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[1,1,1,1],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_433,
        _x_main_module_13);
    auto _x_main_module_435 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_434,
        _x_main_module_12,
        _x_main_module_11,
        _x_main_module_10,
        _x_main_module_9);
    auto _x_main_module_436 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_435);
    auto _x_main_module_437 = mmain->add_instruction(
        migraphx::make_op("convolution",
                          migraphx::from_json_string(
                              "{\"dilation\":[1,1],\"group\":1,\"padding\":[0,0,0,0],\"padding_"
                              "mode\":0,\"stride\":[1,1],\"use_dynamic_same_auto_pad\":0}")),
        _x_main_module_436,
        _x_main_module_8);
    auto _x_main_module_438 = mmain->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            migraphx::from_json_string("{\"bn_mode\":1,\"epsilon\":9.999999747378752e-06,"
                                       "\"momentum\":0.8999999761581421}")),
        _x_main_module_437,
        _x_main_module_7,
        _x_main_module_6,
        _x_main_module_5,
        _x_main_module_4);
    auto _x_main_module_439 =
        mmain->add_instruction(migraphx::make_op("add"), _x_main_module_438, _x_main_module_430);
    auto _x_main_module_440 = mmain->add_instruction(migraphx::make_op("relu"), _x_main_module_439);
    auto _x_main_module_441 = mmain->add_instruction(
        migraphx::make_op(
            "pooling",
            migraphx::from_json_string("{\"ceil_mode\":0,\"lengths\":[7,7],\"lp_order\":2,\"mode\":"
                                       "0,\"padding\":[0,0,0,0],\"stride\":[1,1]}")),
        _x_main_module_440);
    auto _x_main_module_442 = mmain->add_instruction(
        migraphx::make_op("flatten", migraphx::from_json_string("{\"axis\":1}")),
        _x_main_module_441);
    auto _x_main_module_443 = mmain->add_instruction(
        migraphx::make_op("transpose", migraphx::from_json_string("{\"permutation\":[1,0]}")),
        _x_main_module_3);
    auto _x_main_module_444 =
        mmain->add_instruction(migraphx::make_op("dot"), _x_main_module_442, _x_main_module_443);
    auto _x_main_module_445 = mmain->add_instruction(
        migraphx::make_op("multibroadcast", migraphx::from_json_string("{\"out_lens\":[1,1000]}")),
        _x_main_module_2);
    auto _x_main_module_446 = mmain->add_instruction(
        migraphx::make_op("multibroadcast", migraphx::from_json_string("{\"out_lens\":[1,1000]}")),
        _x_main_module_0);
    auto _x_main_module_447 =
        mmain->add_instruction(migraphx::make_op("mul"), _x_main_module_445, _x_main_module_446);
    auto _x_main_module_448 =
        mmain->add_instruction(migraphx::make_op("add"), _x_main_module_444, _x_main_module_447);
    mmain->add_return({_x_main_module_448});

    return p;
}
} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx
