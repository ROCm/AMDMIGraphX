#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/apply_alpha_beta.hpp>
#include "models.hpp"

namespace migraphx {
namespace driver {
inline namespace MIGRAPHX_INLINE_NS {

migraphx::program inceptionv3(unsigned batch) // NOLINT(readability-function-size)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto m0 =
        mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {batch, 3, 299, 299}});
    auto mx0 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1000}}, 0));
    auto mx1 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1000, 2048}}, 1));
    auto mx2 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 2)));
    auto mx3 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 3));
    auto mx4 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 4));
    auto mx5 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 5)));
    auto mx6 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 2048, 1, 1}}, 6));
    auto mx7 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 7)));
    auto mx8 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 8));
    auto mx9 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 9));
    auto mx10 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 10)));
    auto mx11 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 3, 1}}, 11));
    auto mx12 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 12)));
    auto mx13 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 13));
    auto mx14 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 14));
    auto mx15 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 15)));
    auto mx16 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 1, 3}}, 16));
    auto mx17 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 17)));
    auto mx18 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 18));
    auto mx19 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 19));
    auto mx20 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 20)));
    auto mx21 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 448, 3, 3}}, 21));
    auto mx22 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 22)));
    auto mx23 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 23));
    auto mx24 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 24));
    auto mx25 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 25)));
    auto mx26 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {448, 2048, 1, 1}}, 26));
    auto mx27 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 27)));
    auto mx28 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 28)));
    auto mx29 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 29));
    auto mx30 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 30)));
    auto mx31 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 3, 1}}, 31));
    auto mx32 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 32)));
    auto mx33 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 33)));
    auto mx34 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 34));
    auto mx35 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 35)));
    auto mx36 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 1, 3}}, 36));
    auto mx37 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 37)));
    auto mx38 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 38));
    auto mx39 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 39));
    auto mx40 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 40)));
    auto mx41 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 2048, 1, 1}}, 41));
    auto mx42 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 42)));
    auto mx43 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 43));
    auto mx44 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 44));
    auto mx45 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 45)));
    auto mx46 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {320, 2048, 1, 1}}, 46));
    auto mx47 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 47)));
    auto mx48 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 48));
    auto mx49 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 49));
    auto mx50 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 50)));
    auto mx51 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 1280, 1, 1}}, 51));
    auto mx52 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 52)));
    auto mx53 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 53));
    auto mx54 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 54));
    auto mx55 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 55)));
    auto mx56 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 3, 1}}, 56));
    auto mx57 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 57)));
    auto mx58 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 58));
    auto mx59 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 59));
    auto mx60 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 60)));
    auto mx61 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 1, 3}}, 61));
    auto mx62 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 62)));
    auto mx63 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 63));
    auto mx64 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 64));
    auto mx65 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 65)));
    auto mx66 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 448, 3, 3}}, 66));
    auto mx67 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 67)));
    auto mx68 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 68));
    auto mx69 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 69));
    auto mx70 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 70)));
    auto mx71 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {448, 1280, 1, 1}}, 71));
    auto mx72 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 72)));
    auto mx73 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 73));
    auto mx74 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 74));
    auto mx75 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 75)));
    auto mx76 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 3, 1}}, 76));
    auto mx77 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 77)));
    auto mx78 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 78));
    auto mx79 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 79));
    auto mx80 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 80)));
    auto mx81 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 1, 3}}, 81));
    auto mx82 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 82)));
    auto mx83 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 83));
    auto mx84 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 84));
    auto mx85 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 85)));
    auto mx86 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 1280, 1, 1}}, 86));
    auto mx87 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 87)));
    auto mx88 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 88));
    auto mx89 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 89));
    auto mx90 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 90)));
    auto mx91 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {320, 1280, 1, 1}}, 91));
    auto mx92 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 92)));
    auto mx93 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 93));
    auto mx94 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 94));
    auto mx95 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 95)));
    auto mx96 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 3, 3}}, 96));
    auto mx97 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 97)));
    auto mx98 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 98));
    auto mx99 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 99));
    auto mx100 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 100)));
    auto mx101 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 7, 1}}, 101));
    auto mx102 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 102)));
    auto mx103 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 103));
    auto mx104 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 104));
    auto mx105 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 105)));
    auto mx106 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 1, 7}}, 106));
    auto mx107 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 107)));
    auto mx108 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 108));
    auto mx109 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 109));
    auto mx110 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 110)));
    auto mx111 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 111));
    auto mx112 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 112)));
    auto mx113 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 113));
    auto mx114 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 114));
    auto mx115 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 115)));
    auto mx116 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {320, 192, 3, 3}}, 116));
    auto mx117 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 117)));
    auto mx118 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 118));
    auto mx119 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 119));
    auto mx120 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 120)));
    auto mx121 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 121));
    auto mx134 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 134)));
    auto mx135 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 135));
    auto mx136 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 136));
    auto mx137 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 137)));
    auto mx138 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 138));
    auto mx139 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 139)));
    auto mx140 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 140));
    auto mx141 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 141));
    auto mx142 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 142)));
    auto mx143 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 1, 7}}, 143));
    auto mx144 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 144)));
    auto mx145 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 145));
    auto mx146 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 146));
    auto mx147 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 147)));
    auto mx148 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 7, 1}}, 148));
    auto mx149 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 149)));
    auto mx150 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 150));
    auto mx151 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 151));
    auto mx152 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 152)));
    auto mx153 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 1, 7}}, 153));
    auto mx154 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 154)));
    auto mx155 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 155));
    auto mx156 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 156));
    auto mx157 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 157)));
    auto mx158 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 7, 1}}, 158));
    auto mx159 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 159)));
    auto mx160 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 160));
    auto mx161 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 161));
    auto mx162 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 162)));
    auto mx163 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 163));
    auto mx164 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 164)));
    auto mx165 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 165));
    auto mx166 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 166));
    auto mx167 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 167)));
    auto mx168 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 7, 1}}, 168));
    auto mx169 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 169)));
    auto mx170 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 170));
    auto mx171 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 171));
    auto mx172 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 172)));
    auto mx173 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 1, 7}}, 173));
    auto mx174 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 174)));
    auto mx175 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 175));
    auto mx176 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 176));
    auto mx177 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 177)));
    auto mx178 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 178));
    auto mx179 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 179)));
    auto mx180 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 180));
    auto mx181 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 181));
    auto mx182 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 182)));
    auto mx183 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 183));
    auto mx184 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 184)));
    auto mx185 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 185));
    auto mx186 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 186));
    auto mx187 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 187)));
    auto mx188 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 188));
    auto mx189 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 189)));
    auto mx190 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 190));
    auto mx191 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 191));
    auto mx192 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 192)));
    auto mx193 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 160, 1, 7}}, 193));
    auto mx194 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 194)));
    auto mx195 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 195));
    auto mx196 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 196));
    auto mx197 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 197)));
    auto mx198 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 7, 1}}, 198));
    auto mx199 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 199)));
    auto mx200 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 200));
    auto mx201 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 201));
    auto mx202 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 202)));
    auto mx203 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 1, 7}}, 203));
    auto mx204 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 204)));
    auto mx205 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 205));
    auto mx206 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 206));
    auto mx207 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 207)));
    auto mx208 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 7, 1}}, 208));
    auto mx209 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 209)));
    auto mx210 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 210));
    auto mx211 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 211));
    auto mx212 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 212)));
    auto mx213 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 768, 1, 1}}, 213));
    auto mx214 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 214)));
    auto mx215 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 215));
    auto mx216 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 216));
    auto mx217 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 217)));
    auto mx218 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 160, 7, 1}}, 218));
    auto mx219 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 219)));
    auto mx220 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 220));
    auto mx221 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 221));
    auto mx222 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 222)));
    auto mx223 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 1, 7}}, 223));
    auto mx224 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 224)));
    auto mx225 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 225));
    auto mx226 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 226));
    auto mx227 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 227)));
    auto mx228 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 768, 1, 1}}, 228));
    auto mx229 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 229)));
    auto mx230 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 230));
    auto mx231 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 231));
    auto mx232 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 232)));
    auto mx233 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 233));
    auto mx234 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 234)));
    auto mx235 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 235));
    auto mx236 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 236));
    auto mx237 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 237)));
    auto mx238 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 238));
    auto mx239 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 239)));
    auto mx240 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 240));
    auto mx241 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 241));
    auto mx242 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 242)));
    auto mx243 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 160, 1, 7}}, 243));
    auto mx244 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 244)));
    auto mx245 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 245));
    auto mx246 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 246));
    auto mx247 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 247)));
    auto mx248 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 7, 1}}, 248));
    auto mx249 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 249)));
    auto mx250 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 250));
    auto mx251 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 251));
    auto mx252 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 252)));
    auto mx253 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 1, 7}}, 253));
    auto mx254 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 254)));
    auto mx255 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 255));
    auto mx256 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 256));
    auto mx257 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 257)));
    auto mx258 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 7, 1}}, 258));
    auto mx259 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 259)));
    auto mx260 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 260));
    auto mx261 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 261));
    auto mx262 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 262)));
    auto mx263 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 768, 1, 1}}, 263));
    auto mx264 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 264)));
    auto mx265 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 265));
    auto mx266 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 266));
    auto mx267 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 267)));
    auto mx268 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 160, 7, 1}}, 268));
    auto mx269 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 269)));
    auto mx270 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 270));
    auto mx271 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 271));
    auto mx272 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 272)));
    auto mx273 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 1, 7}}, 273));
    auto mx274 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 274)));
    auto mx275 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 275));
    auto mx276 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 276));
    auto mx277 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 277)));
    auto mx278 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 768, 1, 1}}, 278));
    auto mx279 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 279)));
    auto mx280 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 280));
    auto mx281 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 281));
    auto mx282 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 282)));
    auto mx283 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 283));
    auto mx284 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 284)));
    auto mx285 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 285));
    auto mx286 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 286));
    auto mx287 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 287)));
    auto mx288 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 288));
    auto mx289 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 289)));
    auto mx290 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 290));
    auto mx291 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 291));
    auto mx292 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 292)));
    auto mx293 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 128, 1, 7}}, 293));
    auto mx294 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 294)));
    auto mx295 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 295));
    auto mx296 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 296));
    auto mx297 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 297)));
    auto mx298 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 7, 1}}, 298));
    auto mx299 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 299)));
    auto mx300 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 300));
    auto mx301 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 301));
    auto mx302 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 302)));
    auto mx303 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 1, 7}}, 303));
    auto mx304 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 304)));
    auto mx305 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 305));
    auto mx306 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 306));
    auto mx307 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 307)));
    auto mx308 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 7, 1}}, 308));
    auto mx309 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 309)));
    auto mx310 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 310));
    auto mx311 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 311));
    auto mx312 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 312)));
    auto mx313 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 768, 1, 1}}, 313));
    auto mx314 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 314)));
    auto mx315 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 315));
    auto mx316 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 316));
    auto mx317 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 317)));
    auto mx318 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 128, 7, 1}}, 318));
    auto mx319 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 319)));
    auto mx320 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 320));
    auto mx321 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 321));
    auto mx322 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 322)));
    auto mx323 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 1, 7}}, 323));
    auto mx324 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 324)));
    auto mx325 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 325));
    auto mx326 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 326));
    auto mx327 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 327)));
    auto mx328 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 768, 1, 1}}, 328));
    auto mx329 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 329)));
    auto mx330 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 330));
    auto mx331 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 331));
    auto mx332 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 332)));
    auto mx333 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 333));
    auto mx334 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 334)));
    auto mx335 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 335));
    auto mx336 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 336));
    auto mx337 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 337)));
    auto mx338 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 96, 3, 3}}, 338));
    auto mx339 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 339)));
    auto mx340 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 340));
    auto mx341 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 341));
    auto mx342 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 342)));
    auto mx343 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 64, 3, 3}}, 343));
    auto mx344 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 344)));
    auto mx345 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 345));
    auto mx346 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 346));
    auto mx347 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 347)));
    auto mx348 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 288, 1, 1}}, 348));
    auto mx349 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 349)));
    auto mx350 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 350));
    auto mx351 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 351));
    auto mx352 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 352)));
    auto mx353 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 288, 3, 3}}, 353));
    auto mx354 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 354)));
    auto mx355 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 355));
    auto mx356 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 356));
    auto mx357 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 357)));
    auto mx358 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 288, 1, 1}}, 358));
    auto mx359 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 359)));
    auto mx360 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 360));
    auto mx361 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 361));
    auto mx362 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 362)));
    auto mx363 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 96, 3, 3}}, 363));
    auto mx364 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 364)));
    auto mx365 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 365));
    auto mx366 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 366));
    auto mx367 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 367)));
    auto mx368 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 64, 3, 3}}, 368));
    auto mx369 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 369)));
    auto mx370 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 370));
    auto mx371 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 371));
    auto mx372 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 372)));
    auto mx373 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 288, 1, 1}}, 373));
    auto mx374 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 374)));
    auto mx375 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 375));
    auto mx376 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 376));
    auto mx377 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 377)));
    auto mx378 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 48, 5, 5}}, 378));
    auto mx379 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 379)));
    auto mx380 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 380));
    auto mx381 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 381));
    auto mx382 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 382)));
    auto mx383 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {48, 288, 1, 1}}, 383));
    auto mx384 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 384)));
    auto mx385 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 385));
    auto mx386 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 386));
    auto mx387 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 387)));
    auto mx388 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 288, 1, 1}}, 388));
    auto mx389 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 389)));
    auto mx390 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 390));
    auto mx391 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 391));
    auto mx392 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 392)));
    auto mx393 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 256, 1, 1}}, 393));
    auto mx394 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 394)));
    auto mx395 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 395));
    auto mx396 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 396));
    auto mx397 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 397)));
    auto mx398 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 96, 3, 3}}, 398));
    auto mx399 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 399)));
    auto mx400 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 400));
    auto mx401 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 401));
    auto mx402 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 402)));
    auto mx403 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 64, 3, 3}}, 403));
    auto mx404 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 404)));
    auto mx405 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 405));
    auto mx406 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 406));
    auto mx407 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 407)));
    auto mx408 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 256, 1, 1}}, 408));
    auto mx409 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 409)));
    auto mx410 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 410));
    auto mx411 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 411));
    auto mx412 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 412)));
    auto mx413 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 48, 5, 5}}, 413));
    auto mx414 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 414)));
    auto mx415 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 415));
    auto mx416 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 416));
    auto mx417 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 417)));
    auto mx418 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {48, 256, 1, 1}}, 418));
    auto mx419 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 419)));
    auto mx420 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 420));
    auto mx421 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 421));
    auto mx422 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 422)));
    auto mx423 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 256, 1, 1}}, 423));
    auto mx424 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 424)));
    auto mx425 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 425));
    auto mx426 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 426));
    auto mx427 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 427)));
    auto mx428 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {32, 192, 1, 1}}, 428));
    auto mx429 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 429)));
    auto mx430 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 430));
    auto mx431 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 431));
    auto mx432 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 432)));
    auto mx433 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 96, 3, 3}}, 433));
    auto mx434 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 434)));
    auto mx435 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 435));
    auto mx436 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 436));
    auto mx437 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 437)));
    auto mx438 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 64, 3, 3}}, 438));
    auto mx439 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 439)));
    auto mx440 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 440));
    auto mx441 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 441));
    auto mx442 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 442)));
    auto mx443 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 192, 1, 1}}, 443));
    auto mx444 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 444)));
    auto mx445 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 445));
    auto mx446 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 446));
    auto mx447 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 447)));
    auto mx448 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 48, 5, 5}}, 448));
    auto mx449 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 449)));
    auto mx450 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 450));
    auto mx451 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 451));
    auto mx452 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 452)));
    auto mx453 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {48, 192, 1, 1}}, 453));
    auto mx454 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 454)));
    auto mx455 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 455));
    auto mx456 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 456));
    auto mx457 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 457)));
    auto mx458 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 192, 1, 1}}, 458));
    auto mx459 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 459)));
    auto mx460 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 460));
    auto mx461 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 461));
    auto mx462 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 462)));
    auto mx463 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 80, 3, 3}}, 463));
    auto mx464 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {80}}, 464)));
    auto mx465 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {80}}, 465));
    auto mx466 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {80}}, 466));
    auto mx467 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {80}}, 467)));
    auto mx468 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {80, 64, 1, 1}}, 468));
    auto mx469 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 469)));
    auto mx470 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 470));
    auto mx471 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 471));
    auto mx472 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 472)));
    auto mx473 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 32, 3, 3}}, 473));
    auto mx474 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 474)));
    auto mx475 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 475));
    auto mx476 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 476));
    auto mx477 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 477)));
    auto mx478 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {32, 32, 3, 3}}, 478));
    auto mx479 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 479)));
    auto mx480 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 480));
    auto mx481 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 481));
    auto mx482 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 482)));
    auto mx483 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {32, 3, 3, 3}}, 483));
    migraphx::op::convolution convolution484;
    convolution484.padding  = {0, 0};
    convolution484.stride   = {2, 2};
    convolution484.dilation = {1, 1};
    convolution484.group    = 1;
    auto mx484              = mm->add_instruction(convolution484, m0, mx483);
    migraphx::op::batch_norm_inference batch_norm_inference485;
    batch_norm_inference485.epsilon  = 0.001;
    batch_norm_inference485.momentum = 0.9;
    auto mx485 = mm->add_instruction(batch_norm_inference485, mx484, mx482, mx481, mx480, mx479);
    migraphx::op::relu relu486;
    auto mx486 = mm->add_instruction(relu486, mx485);
    migraphx::op::convolution convolution487;
    convolution487.padding  = {0, 0};
    convolution487.stride   = {1, 1};
    convolution487.dilation = {1, 1};
    convolution487.group    = 1;
    auto mx487              = mm->add_instruction(convolution487, mx486, mx478);
    migraphx::op::batch_norm_inference batch_norm_inference488;
    batch_norm_inference488.epsilon  = 0.001;
    batch_norm_inference488.momentum = 0.9;
    auto mx488 = mm->add_instruction(batch_norm_inference488, mx487, mx477, mx476, mx475, mx474);
    migraphx::op::relu relu489;
    auto mx489 = mm->add_instruction(relu489, mx488);
    migraphx::op::convolution convolution490;
    convolution490.padding  = {1, 1};
    convolution490.stride   = {1, 1};
    convolution490.dilation = {1, 1};
    convolution490.group    = 1;
    auto mx490              = mm->add_instruction(convolution490, mx489, mx473);
    migraphx::op::batch_norm_inference batch_norm_inference491;
    batch_norm_inference491.epsilon  = 0.001;
    batch_norm_inference491.momentum = 0.9;
    auto mx491 = mm->add_instruction(batch_norm_inference491, mx490, mx472, mx471, mx470, mx469);
    migraphx::op::relu relu492;
    auto mx492 = mm->add_instruction(relu492, mx491);
    migraphx::op::pooling pooling493;
    pooling493.mode    = migraphx::op::pooling_mode::max;
    pooling493.padding = {0, 0};
    pooling493.stride  = {2, 2};
    pooling493.lengths = {3, 3};
    auto mx493         = mm->add_instruction(pooling493, mx492);
    migraphx::op::convolution convolution494;
    convolution494.padding  = {0, 0};
    convolution494.stride   = {1, 1};
    convolution494.dilation = {1, 1};
    convolution494.group    = 1;
    auto mx494              = mm->add_instruction(convolution494, mx493, mx468);
    migraphx::op::batch_norm_inference batch_norm_inference495;
    batch_norm_inference495.epsilon  = 0.001;
    batch_norm_inference495.momentum = 0.9;
    auto mx495 = mm->add_instruction(batch_norm_inference495, mx494, mx467, mx466, mx465, mx464);
    migraphx::op::relu relu496;
    auto mx496 = mm->add_instruction(relu496, mx495);
    migraphx::op::convolution convolution497;
    convolution497.padding  = {0, 0};
    convolution497.stride   = {1, 1};
    convolution497.dilation = {1, 1};
    convolution497.group    = 1;
    auto mx497              = mm->add_instruction(convolution497, mx496, mx463);
    migraphx::op::batch_norm_inference batch_norm_inference498;
    batch_norm_inference498.epsilon  = 0.001;
    batch_norm_inference498.momentum = 0.9;
    auto mx498 = mm->add_instruction(batch_norm_inference498, mx497, mx462, mx461, mx460, mx459);
    migraphx::op::relu relu499;
    auto mx499 = mm->add_instruction(relu499, mx498);
    migraphx::op::pooling pooling500;
    pooling500.mode    = migraphx::op::pooling_mode::max;
    pooling500.padding = {0, 0};
    pooling500.stride  = {2, 2};
    pooling500.lengths = {3, 3};
    auto mx500         = mm->add_instruction(pooling500, mx499);
    migraphx::op::convolution convolution501;
    convolution501.padding  = {0, 0};
    convolution501.stride   = {1, 1};
    convolution501.dilation = {1, 1};
    convolution501.group    = 1;
    auto mx501              = mm->add_instruction(convolution501, mx500, mx458);
    migraphx::op::batch_norm_inference batch_norm_inference502;
    batch_norm_inference502.epsilon  = 0.001;
    batch_norm_inference502.momentum = 0.9;
    auto mx502 = mm->add_instruction(batch_norm_inference502, mx501, mx457, mx456, mx455, mx454);
    migraphx::op::relu relu503;
    auto mx503 = mm->add_instruction(relu503, mx502);
    migraphx::op::convolution convolution504;
    convolution504.padding  = {0, 0};
    convolution504.stride   = {1, 1};
    convolution504.dilation = {1, 1};
    convolution504.group    = 1;
    auto mx504              = mm->add_instruction(convolution504, mx500, mx453);
    migraphx::op::batch_norm_inference batch_norm_inference505;
    batch_norm_inference505.epsilon  = 0.001;
    batch_norm_inference505.momentum = 0.9;
    auto mx505 = mm->add_instruction(batch_norm_inference505, mx504, mx452, mx451, mx450, mx449);
    migraphx::op::relu relu506;
    auto mx506 = mm->add_instruction(relu506, mx505);
    migraphx::op::convolution convolution507;
    convolution507.padding  = {2, 2};
    convolution507.stride   = {1, 1};
    convolution507.dilation = {1, 1};
    convolution507.group    = 1;
    auto mx507              = mm->add_instruction(convolution507, mx506, mx448);
    migraphx::op::batch_norm_inference batch_norm_inference508;
    batch_norm_inference508.epsilon  = 0.001;
    batch_norm_inference508.momentum = 0.9;
    auto mx508 = mm->add_instruction(batch_norm_inference508, mx507, mx447, mx446, mx445, mx444);
    migraphx::op::relu relu509;
    auto mx509 = mm->add_instruction(relu509, mx508);
    migraphx::op::convolution convolution510;
    convolution510.padding  = {0, 0};
    convolution510.stride   = {1, 1};
    convolution510.dilation = {1, 1};
    convolution510.group    = 1;
    auto mx510              = mm->add_instruction(convolution510, mx500, mx443);
    migraphx::op::batch_norm_inference batch_norm_inference511;
    batch_norm_inference511.epsilon  = 0.001;
    batch_norm_inference511.momentum = 0.9;
    auto mx511 = mm->add_instruction(batch_norm_inference511, mx510, mx442, mx441, mx440, mx439);
    migraphx::op::relu relu512;
    auto mx512 = mm->add_instruction(relu512, mx511);
    migraphx::op::convolution convolution513;
    convolution513.padding  = {1, 1};
    convolution513.stride   = {1, 1};
    convolution513.dilation = {1, 1};
    convolution513.group    = 1;
    auto mx513              = mm->add_instruction(convolution513, mx512, mx438);
    migraphx::op::batch_norm_inference batch_norm_inference514;
    batch_norm_inference514.epsilon  = 0.001;
    batch_norm_inference514.momentum = 0.9;
    auto mx514 = mm->add_instruction(batch_norm_inference514, mx513, mx437, mx436, mx435, mx434);
    migraphx::op::relu relu515;
    auto mx515 = mm->add_instruction(relu515, mx514);
    migraphx::op::convolution convolution516;
    convolution516.padding  = {1, 1};
    convolution516.stride   = {1, 1};
    convolution516.dilation = {1, 1};
    convolution516.group    = 1;
    auto mx516              = mm->add_instruction(convolution516, mx515, mx433);
    migraphx::op::batch_norm_inference batch_norm_inference517;
    batch_norm_inference517.epsilon  = 0.001;
    batch_norm_inference517.momentum = 0.9;
    auto mx517 = mm->add_instruction(batch_norm_inference517, mx516, mx432, mx431, mx430, mx429);
    migraphx::op::relu relu518;
    auto mx518 = mm->add_instruction(relu518, mx517);
    migraphx::op::pooling pooling519;
    pooling519.mode    = migraphx::op::pooling_mode::average;
    pooling519.padding = {1, 1};
    pooling519.stride  = {1, 1};
    pooling519.lengths = {3, 3};
    auto mx519         = mm->add_instruction(pooling519, mx500);
    migraphx::op::convolution convolution520;
    convolution520.padding  = {0, 0};
    convolution520.stride   = {1, 1};
    convolution520.dilation = {1, 1};
    convolution520.group    = 1;
    auto mx520              = mm->add_instruction(convolution520, mx519, mx428);
    migraphx::op::batch_norm_inference batch_norm_inference521;
    batch_norm_inference521.epsilon  = 0.001;
    batch_norm_inference521.momentum = 0.9;
    auto mx521 = mm->add_instruction(batch_norm_inference521, mx520, mx427, mx426, mx425, mx424);
    migraphx::op::relu relu522;
    auto mx522 = mm->add_instruction(relu522, mx521);
    migraphx::op::concat concat523;
    concat523.axis = 1;
    auto mx523     = mm->add_instruction(concat523, mx503, mx509, mx518, mx522);
    migraphx::op::convolution convolution524;
    convolution524.padding  = {0, 0};
    convolution524.stride   = {1, 1};
    convolution524.dilation = {1, 1};
    convolution524.group    = 1;
    auto mx524              = mm->add_instruction(convolution524, mx523, mx423);
    migraphx::op::batch_norm_inference batch_norm_inference525;
    batch_norm_inference525.epsilon  = 0.001;
    batch_norm_inference525.momentum = 0.9;
    auto mx525 = mm->add_instruction(batch_norm_inference525, mx524, mx422, mx421, mx420, mx419);
    migraphx::op::relu relu526;
    auto mx526 = mm->add_instruction(relu526, mx525);
    migraphx::op::convolution convolution527;
    convolution527.padding  = {0, 0};
    convolution527.stride   = {1, 1};
    convolution527.dilation = {1, 1};
    convolution527.group    = 1;
    auto mx527              = mm->add_instruction(convolution527, mx523, mx418);
    migraphx::op::batch_norm_inference batch_norm_inference528;
    batch_norm_inference528.epsilon  = 0.001;
    batch_norm_inference528.momentum = 0.9;
    auto mx528 = mm->add_instruction(batch_norm_inference528, mx527, mx417, mx416, mx415, mx414);
    migraphx::op::relu relu529;
    auto mx529 = mm->add_instruction(relu529, mx528);
    migraphx::op::convolution convolution530;
    convolution530.padding  = {2, 2};
    convolution530.stride   = {1, 1};
    convolution530.dilation = {1, 1};
    convolution530.group    = 1;
    auto mx530              = mm->add_instruction(convolution530, mx529, mx413);
    migraphx::op::batch_norm_inference batch_norm_inference531;
    batch_norm_inference531.epsilon  = 0.001;
    batch_norm_inference531.momentum = 0.9;
    auto mx531 = mm->add_instruction(batch_norm_inference531, mx530, mx412, mx411, mx410, mx409);
    migraphx::op::relu relu532;
    auto mx532 = mm->add_instruction(relu532, mx531);
    migraphx::op::convolution convolution533;
    convolution533.padding  = {0, 0};
    convolution533.stride   = {1, 1};
    convolution533.dilation = {1, 1};
    convolution533.group    = 1;
    auto mx533              = mm->add_instruction(convolution533, mx523, mx408);
    migraphx::op::batch_norm_inference batch_norm_inference534;
    batch_norm_inference534.epsilon  = 0.001;
    batch_norm_inference534.momentum = 0.9;
    auto mx534 = mm->add_instruction(batch_norm_inference534, mx533, mx407, mx406, mx405, mx404);
    migraphx::op::relu relu535;
    auto mx535 = mm->add_instruction(relu535, mx534);
    migraphx::op::convolution convolution536;
    convolution536.padding  = {1, 1};
    convolution536.stride   = {1, 1};
    convolution536.dilation = {1, 1};
    convolution536.group    = 1;
    auto mx536              = mm->add_instruction(convolution536, mx535, mx403);
    migraphx::op::batch_norm_inference batch_norm_inference537;
    batch_norm_inference537.epsilon  = 0.001;
    batch_norm_inference537.momentum = 0.9;
    auto mx537 = mm->add_instruction(batch_norm_inference537, mx536, mx402, mx401, mx400, mx399);
    migraphx::op::relu relu538;
    auto mx538 = mm->add_instruction(relu538, mx537);
    migraphx::op::convolution convolution539;
    convolution539.padding  = {1, 1};
    convolution539.stride   = {1, 1};
    convolution539.dilation = {1, 1};
    convolution539.group    = 1;
    auto mx539              = mm->add_instruction(convolution539, mx538, mx398);
    migraphx::op::batch_norm_inference batch_norm_inference540;
    batch_norm_inference540.epsilon  = 0.001;
    batch_norm_inference540.momentum = 0.9;
    auto mx540 = mm->add_instruction(batch_norm_inference540, mx539, mx397, mx396, mx395, mx394);
    migraphx::op::relu relu541;
    auto mx541 = mm->add_instruction(relu541, mx540);
    migraphx::op::pooling pooling542;
    pooling542.mode    = migraphx::op::pooling_mode::average;
    pooling542.padding = {1, 1};
    pooling542.stride  = {1, 1};
    pooling542.lengths = {3, 3};
    auto mx542         = mm->add_instruction(pooling542, mx523);
    migraphx::op::convolution convolution543;
    convolution543.padding  = {0, 0};
    convolution543.stride   = {1, 1};
    convolution543.dilation = {1, 1};
    convolution543.group    = 1;
    auto mx543              = mm->add_instruction(convolution543, mx542, mx393);
    migraphx::op::batch_norm_inference batch_norm_inference544;
    batch_norm_inference544.epsilon  = 0.001;
    batch_norm_inference544.momentum = 0.9;
    auto mx544 = mm->add_instruction(batch_norm_inference544, mx543, mx392, mx391, mx390, mx389);
    migraphx::op::relu relu545;
    auto mx545 = mm->add_instruction(relu545, mx544);
    migraphx::op::concat concat546;
    concat546.axis = 1;
    auto mx546     = mm->add_instruction(concat546, mx526, mx532, mx541, mx545);
    migraphx::op::convolution convolution547;
    convolution547.padding  = {0, 0};
    convolution547.stride   = {1, 1};
    convolution547.dilation = {1, 1};
    convolution547.group    = 1;
    auto mx547              = mm->add_instruction(convolution547, mx546, mx388);
    migraphx::op::batch_norm_inference batch_norm_inference548;
    batch_norm_inference548.epsilon  = 0.001;
    batch_norm_inference548.momentum = 0.9;
    auto mx548 = mm->add_instruction(batch_norm_inference548, mx547, mx387, mx386, mx385, mx384);
    migraphx::op::relu relu549;
    auto mx549 = mm->add_instruction(relu549, mx548);
    migraphx::op::convolution convolution550;
    convolution550.padding  = {0, 0};
    convolution550.stride   = {1, 1};
    convolution550.dilation = {1, 1};
    convolution550.group    = 1;
    auto mx550              = mm->add_instruction(convolution550, mx546, mx383);
    migraphx::op::batch_norm_inference batch_norm_inference551;
    batch_norm_inference551.epsilon  = 0.001;
    batch_norm_inference551.momentum = 0.9;
    auto mx551 = mm->add_instruction(batch_norm_inference551, mx550, mx382, mx381, mx380, mx379);
    migraphx::op::relu relu552;
    auto mx552 = mm->add_instruction(relu552, mx551);
    migraphx::op::convolution convolution553;
    convolution553.padding  = {2, 2};
    convolution553.stride   = {1, 1};
    convolution553.dilation = {1, 1};
    convolution553.group    = 1;
    auto mx553              = mm->add_instruction(convolution553, mx552, mx378);
    migraphx::op::batch_norm_inference batch_norm_inference554;
    batch_norm_inference554.epsilon  = 0.001;
    batch_norm_inference554.momentum = 0.9;
    auto mx554 = mm->add_instruction(batch_norm_inference554, mx553, mx377, mx376, mx375, mx374);
    migraphx::op::relu relu555;
    auto mx555 = mm->add_instruction(relu555, mx554);
    migraphx::op::convolution convolution556;
    convolution556.padding  = {0, 0};
    convolution556.stride   = {1, 1};
    convolution556.dilation = {1, 1};
    convolution556.group    = 1;
    auto mx556              = mm->add_instruction(convolution556, mx546, mx373);
    migraphx::op::batch_norm_inference batch_norm_inference557;
    batch_norm_inference557.epsilon  = 0.001;
    batch_norm_inference557.momentum = 0.9;
    auto mx557 = mm->add_instruction(batch_norm_inference557, mx556, mx372, mx371, mx370, mx369);
    migraphx::op::relu relu558;
    auto mx558 = mm->add_instruction(relu558, mx557);
    migraphx::op::convolution convolution559;
    convolution559.padding  = {1, 1};
    convolution559.stride   = {1, 1};
    convolution559.dilation = {1, 1};
    convolution559.group    = 1;
    auto mx559              = mm->add_instruction(convolution559, mx558, mx368);
    migraphx::op::batch_norm_inference batch_norm_inference560;
    batch_norm_inference560.epsilon  = 0.001;
    batch_norm_inference560.momentum = 0.9;
    auto mx560 = mm->add_instruction(batch_norm_inference560, mx559, mx367, mx366, mx365, mx364);
    migraphx::op::relu relu561;
    auto mx561 = mm->add_instruction(relu561, mx560);
    migraphx::op::convolution convolution562;
    convolution562.padding  = {1, 1};
    convolution562.stride   = {1, 1};
    convolution562.dilation = {1, 1};
    convolution562.group    = 1;
    auto mx562              = mm->add_instruction(convolution562, mx561, mx363);
    migraphx::op::batch_norm_inference batch_norm_inference563;
    batch_norm_inference563.epsilon  = 0.001;
    batch_norm_inference563.momentum = 0.9;
    auto mx563 = mm->add_instruction(batch_norm_inference563, mx562, mx362, mx361, mx360, mx359);
    migraphx::op::relu relu564;
    auto mx564 = mm->add_instruction(relu564, mx563);
    migraphx::op::pooling pooling565;
    pooling565.mode    = migraphx::op::pooling_mode::average;
    pooling565.padding = {1, 1};
    pooling565.stride  = {1, 1};
    pooling565.lengths = {3, 3};
    auto mx565         = mm->add_instruction(pooling565, mx546);
    migraphx::op::convolution convolution566;
    convolution566.padding  = {0, 0};
    convolution566.stride   = {1, 1};
    convolution566.dilation = {1, 1};
    convolution566.group    = 1;
    auto mx566              = mm->add_instruction(convolution566, mx565, mx358);
    migraphx::op::batch_norm_inference batch_norm_inference567;
    batch_norm_inference567.epsilon  = 0.001;
    batch_norm_inference567.momentum = 0.9;
    auto mx567 = mm->add_instruction(batch_norm_inference567, mx566, mx357, mx356, mx355, mx354);
    migraphx::op::relu relu568;
    auto mx568 = mm->add_instruction(relu568, mx567);
    migraphx::op::concat concat569;
    concat569.axis = 1;
    auto mx569     = mm->add_instruction(concat569, mx549, mx555, mx564, mx568);
    migraphx::op::convolution convolution570;
    convolution570.padding  = {0, 0};
    convolution570.stride   = {2, 2};
    convolution570.dilation = {1, 1};
    convolution570.group    = 1;
    auto mx570              = mm->add_instruction(convolution570, mx569, mx353);
    migraphx::op::batch_norm_inference batch_norm_inference571;
    batch_norm_inference571.epsilon  = 0.001;
    batch_norm_inference571.momentum = 0.9;
    auto mx571 = mm->add_instruction(batch_norm_inference571, mx570, mx352, mx351, mx350, mx349);
    migraphx::op::relu relu572;
    auto mx572 = mm->add_instruction(relu572, mx571);
    migraphx::op::convolution convolution573;
    convolution573.padding  = {0, 0};
    convolution573.stride   = {1, 1};
    convolution573.dilation = {1, 1};
    convolution573.group    = 1;
    auto mx573              = mm->add_instruction(convolution573, mx569, mx348);
    migraphx::op::batch_norm_inference batch_norm_inference574;
    batch_norm_inference574.epsilon  = 0.001;
    batch_norm_inference574.momentum = 0.9;
    auto mx574 = mm->add_instruction(batch_norm_inference574, mx573, mx347, mx346, mx345, mx344);
    migraphx::op::relu relu575;
    auto mx575 = mm->add_instruction(relu575, mx574);
    migraphx::op::convolution convolution576;
    convolution576.padding  = {1, 1};
    convolution576.stride   = {1, 1};
    convolution576.dilation = {1, 1};
    convolution576.group    = 1;
    auto mx576              = mm->add_instruction(convolution576, mx575, mx343);
    migraphx::op::batch_norm_inference batch_norm_inference577;
    batch_norm_inference577.epsilon  = 0.001;
    batch_norm_inference577.momentum = 0.9;
    auto mx577 = mm->add_instruction(batch_norm_inference577, mx576, mx342, mx341, mx340, mx339);
    migraphx::op::relu relu578;
    auto mx578 = mm->add_instruction(relu578, mx577);
    migraphx::op::convolution convolution579;
    convolution579.padding  = {0, 0};
    convolution579.stride   = {2, 2};
    convolution579.dilation = {1, 1};
    convolution579.group    = 1;
    auto mx579              = mm->add_instruction(convolution579, mx578, mx338);
    migraphx::op::batch_norm_inference batch_norm_inference580;
    batch_norm_inference580.epsilon  = 0.001;
    batch_norm_inference580.momentum = 0.9;
    auto mx580 = mm->add_instruction(batch_norm_inference580, mx579, mx337, mx336, mx335, mx334);
    migraphx::op::relu relu581;
    auto mx581 = mm->add_instruction(relu581, mx580);
    migraphx::op::pooling pooling582;
    pooling582.mode    = migraphx::op::pooling_mode::max;
    pooling582.padding = {0, 0};
    pooling582.stride  = {2, 2};
    pooling582.lengths = {3, 3};
    auto mx582         = mm->add_instruction(pooling582, mx569);
    migraphx::op::concat concat583;
    concat583.axis = 1;
    auto mx583     = mm->add_instruction(concat583, mx572, mx581, mx582);
    migraphx::op::convolution convolution584;
    convolution584.padding  = {0, 0};
    convolution584.stride   = {1, 1};
    convolution584.dilation = {1, 1};
    convolution584.group    = 1;
    auto mx584              = mm->add_instruction(convolution584, mx583, mx333);
    migraphx::op::batch_norm_inference batch_norm_inference585;
    batch_norm_inference585.epsilon  = 0.001;
    batch_norm_inference585.momentum = 0.9;
    auto mx585 = mm->add_instruction(batch_norm_inference585, mx584, mx332, mx331, mx330, mx329);
    migraphx::op::relu relu586;
    auto mx586 = mm->add_instruction(relu586, mx585);
    migraphx::op::convolution convolution587;
    convolution587.padding  = {0, 0};
    convolution587.stride   = {1, 1};
    convolution587.dilation = {1, 1};
    convolution587.group    = 1;
    auto mx587              = mm->add_instruction(convolution587, mx583, mx328);
    migraphx::op::batch_norm_inference batch_norm_inference588;
    batch_norm_inference588.epsilon  = 0.001;
    batch_norm_inference588.momentum = 0.9;
    auto mx588 = mm->add_instruction(batch_norm_inference588, mx587, mx327, mx326, mx325, mx324);
    migraphx::op::relu relu589;
    auto mx589 = mm->add_instruction(relu589, mx588);
    migraphx::op::convolution convolution590;
    convolution590.padding  = {0, 3};
    convolution590.stride   = {1, 1};
    convolution590.dilation = {1, 1};
    convolution590.group    = 1;
    auto mx590              = mm->add_instruction(convolution590, mx589, mx323);
    migraphx::op::batch_norm_inference batch_norm_inference591;
    batch_norm_inference591.epsilon  = 0.001;
    batch_norm_inference591.momentum = 0.9;
    auto mx591 = mm->add_instruction(batch_norm_inference591, mx590, mx322, mx321, mx320, mx319);
    migraphx::op::relu relu592;
    auto mx592 = mm->add_instruction(relu592, mx591);
    migraphx::op::convolution convolution593;
    convolution593.padding  = {3, 0};
    convolution593.stride   = {1, 1};
    convolution593.dilation = {1, 1};
    convolution593.group    = 1;
    auto mx593              = mm->add_instruction(convolution593, mx592, mx318);
    migraphx::op::batch_norm_inference batch_norm_inference594;
    batch_norm_inference594.epsilon  = 0.001;
    batch_norm_inference594.momentum = 0.9;
    auto mx594 = mm->add_instruction(batch_norm_inference594, mx593, mx317, mx316, mx315, mx314);
    migraphx::op::relu relu595;
    auto mx595 = mm->add_instruction(relu595, mx594);
    migraphx::op::convolution convolution596;
    convolution596.padding  = {0, 0};
    convolution596.stride   = {1, 1};
    convolution596.dilation = {1, 1};
    convolution596.group    = 1;
    auto mx596              = mm->add_instruction(convolution596, mx583, mx313);
    migraphx::op::batch_norm_inference batch_norm_inference597;
    batch_norm_inference597.epsilon  = 0.001;
    batch_norm_inference597.momentum = 0.9;
    auto mx597 = mm->add_instruction(batch_norm_inference597, mx596, mx312, mx311, mx310, mx309);
    migraphx::op::relu relu598;
    auto mx598 = mm->add_instruction(relu598, mx597);
    migraphx::op::convolution convolution599;
    convolution599.padding  = {3, 0};
    convolution599.stride   = {1, 1};
    convolution599.dilation = {1, 1};
    convolution599.group    = 1;
    auto mx599              = mm->add_instruction(convolution599, mx598, mx308);
    migraphx::op::batch_norm_inference batch_norm_inference600;
    batch_norm_inference600.epsilon  = 0.001;
    batch_norm_inference600.momentum = 0.9;
    auto mx600 = mm->add_instruction(batch_norm_inference600, mx599, mx307, mx306, mx305, mx304);
    migraphx::op::relu relu601;
    auto mx601 = mm->add_instruction(relu601, mx600);
    migraphx::op::convolution convolution602;
    convolution602.padding  = {0, 3};
    convolution602.stride   = {1, 1};
    convolution602.dilation = {1, 1};
    convolution602.group    = 1;
    auto mx602              = mm->add_instruction(convolution602, mx601, mx303);
    migraphx::op::batch_norm_inference batch_norm_inference603;
    batch_norm_inference603.epsilon  = 0.001;
    batch_norm_inference603.momentum = 0.9;
    auto mx603 = mm->add_instruction(batch_norm_inference603, mx602, mx302, mx301, mx300, mx299);
    migraphx::op::relu relu604;
    auto mx604 = mm->add_instruction(relu604, mx603);
    migraphx::op::convolution convolution605;
    convolution605.padding  = {3, 0};
    convolution605.stride   = {1, 1};
    convolution605.dilation = {1, 1};
    convolution605.group    = 1;
    auto mx605              = mm->add_instruction(convolution605, mx604, mx298);
    migraphx::op::batch_norm_inference batch_norm_inference606;
    batch_norm_inference606.epsilon  = 0.001;
    batch_norm_inference606.momentum = 0.9;
    auto mx606 = mm->add_instruction(batch_norm_inference606, mx605, mx297, mx296, mx295, mx294);
    migraphx::op::relu relu607;
    auto mx607 = mm->add_instruction(relu607, mx606);
    migraphx::op::convolution convolution608;
    convolution608.padding  = {0, 3};
    convolution608.stride   = {1, 1};
    convolution608.dilation = {1, 1};
    convolution608.group    = 1;
    auto mx608              = mm->add_instruction(convolution608, mx607, mx293);
    migraphx::op::batch_norm_inference batch_norm_inference609;
    batch_norm_inference609.epsilon  = 0.001;
    batch_norm_inference609.momentum = 0.9;
    auto mx609 = mm->add_instruction(batch_norm_inference609, mx608, mx292, mx291, mx290, mx289);
    migraphx::op::relu relu610;
    auto mx610 = mm->add_instruction(relu610, mx609);
    migraphx::op::pooling pooling611;
    pooling611.mode    = migraphx::op::pooling_mode::average;
    pooling611.padding = {1, 1};
    pooling611.stride  = {1, 1};
    pooling611.lengths = {3, 3};
    auto mx611         = mm->add_instruction(pooling611, mx583);
    migraphx::op::convolution convolution612;
    convolution612.padding  = {0, 0};
    convolution612.stride   = {1, 1};
    convolution612.dilation = {1, 1};
    convolution612.group    = 1;
    auto mx612              = mm->add_instruction(convolution612, mx611, mx288);
    migraphx::op::batch_norm_inference batch_norm_inference613;
    batch_norm_inference613.epsilon  = 0.001;
    batch_norm_inference613.momentum = 0.9;
    auto mx613 = mm->add_instruction(batch_norm_inference613, mx612, mx287, mx286, mx285, mx284);
    migraphx::op::relu relu614;
    auto mx614 = mm->add_instruction(relu614, mx613);
    migraphx::op::concat concat615;
    concat615.axis = 1;
    auto mx615     = mm->add_instruction(concat615, mx586, mx595, mx610, mx614);
    migraphx::op::convolution convolution616;
    convolution616.padding  = {0, 0};
    convolution616.stride   = {1, 1};
    convolution616.dilation = {1, 1};
    convolution616.group    = 1;
    auto mx616              = mm->add_instruction(convolution616, mx615, mx283);
    migraphx::op::batch_norm_inference batch_norm_inference617;
    batch_norm_inference617.epsilon  = 0.001;
    batch_norm_inference617.momentum = 0.9;
    auto mx617 = mm->add_instruction(batch_norm_inference617, mx616, mx282, mx281, mx280, mx279);
    migraphx::op::relu relu618;
    auto mx618 = mm->add_instruction(relu618, mx617);
    migraphx::op::convolution convolution619;
    convolution619.padding  = {0, 0};
    convolution619.stride   = {1, 1};
    convolution619.dilation = {1, 1};
    convolution619.group    = 1;
    auto mx619              = mm->add_instruction(convolution619, mx615, mx278);
    migraphx::op::batch_norm_inference batch_norm_inference620;
    batch_norm_inference620.epsilon  = 0.001;
    batch_norm_inference620.momentum = 0.9;
    auto mx620 = mm->add_instruction(batch_norm_inference620, mx619, mx277, mx276, mx275, mx274);
    migraphx::op::relu relu621;
    auto mx621 = mm->add_instruction(relu621, mx620);
    migraphx::op::convolution convolution622;
    convolution622.padding  = {0, 3};
    convolution622.stride   = {1, 1};
    convolution622.dilation = {1, 1};
    convolution622.group    = 1;
    auto mx622              = mm->add_instruction(convolution622, mx621, mx273);
    migraphx::op::batch_norm_inference batch_norm_inference623;
    batch_norm_inference623.epsilon  = 0.001;
    batch_norm_inference623.momentum = 0.9;
    auto mx623 = mm->add_instruction(batch_norm_inference623, mx622, mx272, mx271, mx270, mx269);
    migraphx::op::relu relu624;
    auto mx624 = mm->add_instruction(relu624, mx623);
    migraphx::op::convolution convolution625;
    convolution625.padding  = {3, 0};
    convolution625.stride   = {1, 1};
    convolution625.dilation = {1, 1};
    convolution625.group    = 1;
    auto mx625              = mm->add_instruction(convolution625, mx624, mx268);
    migraphx::op::batch_norm_inference batch_norm_inference626;
    batch_norm_inference626.epsilon  = 0.001;
    batch_norm_inference626.momentum = 0.9;
    auto mx626 = mm->add_instruction(batch_norm_inference626, mx625, mx267, mx266, mx265, mx264);
    migraphx::op::relu relu627;
    auto mx627 = mm->add_instruction(relu627, mx626);
    migraphx::op::convolution convolution628;
    convolution628.padding  = {0, 0};
    convolution628.stride   = {1, 1};
    convolution628.dilation = {1, 1};
    convolution628.group    = 1;
    auto mx628              = mm->add_instruction(convolution628, mx615, mx263);
    migraphx::op::batch_norm_inference batch_norm_inference629;
    batch_norm_inference629.epsilon  = 0.001;
    batch_norm_inference629.momentum = 0.9;
    auto mx629 = mm->add_instruction(batch_norm_inference629, mx628, mx262, mx261, mx260, mx259);
    migraphx::op::relu relu630;
    auto mx630 = mm->add_instruction(relu630, mx629);
    migraphx::op::convolution convolution631;
    convolution631.padding  = {3, 0};
    convolution631.stride   = {1, 1};
    convolution631.dilation = {1, 1};
    convolution631.group    = 1;
    auto mx631              = mm->add_instruction(convolution631, mx630, mx258);
    migraphx::op::batch_norm_inference batch_norm_inference632;
    batch_norm_inference632.epsilon  = 0.001;
    batch_norm_inference632.momentum = 0.9;
    auto mx632 = mm->add_instruction(batch_norm_inference632, mx631, mx257, mx256, mx255, mx254);
    migraphx::op::relu relu633;
    auto mx633 = mm->add_instruction(relu633, mx632);
    migraphx::op::convolution convolution634;
    convolution634.padding  = {0, 3};
    convolution634.stride   = {1, 1};
    convolution634.dilation = {1, 1};
    convolution634.group    = 1;
    auto mx634              = mm->add_instruction(convolution634, mx633, mx253);
    migraphx::op::batch_norm_inference batch_norm_inference635;
    batch_norm_inference635.epsilon  = 0.001;
    batch_norm_inference635.momentum = 0.9;
    auto mx635 = mm->add_instruction(batch_norm_inference635, mx634, mx252, mx251, mx250, mx249);
    migraphx::op::relu relu636;
    auto mx636 = mm->add_instruction(relu636, mx635);
    migraphx::op::convolution convolution637;
    convolution637.padding  = {3, 0};
    convolution637.stride   = {1, 1};
    convolution637.dilation = {1, 1};
    convolution637.group    = 1;
    auto mx637              = mm->add_instruction(convolution637, mx636, mx248);
    migraphx::op::batch_norm_inference batch_norm_inference638;
    batch_norm_inference638.epsilon  = 0.001;
    batch_norm_inference638.momentum = 0.9;
    auto mx638 = mm->add_instruction(batch_norm_inference638, mx637, mx247, mx246, mx245, mx244);
    migraphx::op::relu relu639;
    auto mx639 = mm->add_instruction(relu639, mx638);
    migraphx::op::convolution convolution640;
    convolution640.padding  = {0, 3};
    convolution640.stride   = {1, 1};
    convolution640.dilation = {1, 1};
    convolution640.group    = 1;
    auto mx640              = mm->add_instruction(convolution640, mx639, mx243);
    migraphx::op::batch_norm_inference batch_norm_inference641;
    batch_norm_inference641.epsilon  = 0.001;
    batch_norm_inference641.momentum = 0.9;
    auto mx641 = mm->add_instruction(batch_norm_inference641, mx640, mx242, mx241, mx240, mx239);
    migraphx::op::relu relu642;
    auto mx642 = mm->add_instruction(relu642, mx641);
    migraphx::op::pooling pooling643;
    pooling643.mode    = migraphx::op::pooling_mode::average;
    pooling643.padding = {1, 1};
    pooling643.stride  = {1, 1};
    pooling643.lengths = {3, 3};
    auto mx643         = mm->add_instruction(pooling643, mx615);
    migraphx::op::convolution convolution644;
    convolution644.padding  = {0, 0};
    convolution644.stride   = {1, 1};
    convolution644.dilation = {1, 1};
    convolution644.group    = 1;
    auto mx644              = mm->add_instruction(convolution644, mx643, mx238);
    migraphx::op::batch_norm_inference batch_norm_inference645;
    batch_norm_inference645.epsilon  = 0.001;
    batch_norm_inference645.momentum = 0.9;
    auto mx645 = mm->add_instruction(batch_norm_inference645, mx644, mx237, mx236, mx235, mx234);
    migraphx::op::relu relu646;
    auto mx646 = mm->add_instruction(relu646, mx645);
    migraphx::op::concat concat647;
    concat647.axis = 1;
    auto mx647     = mm->add_instruction(concat647, mx618, mx627, mx642, mx646);
    migraphx::op::convolution convolution648;
    convolution648.padding  = {0, 0};
    convolution648.stride   = {1, 1};
    convolution648.dilation = {1, 1};
    convolution648.group    = 1;
    auto mx648              = mm->add_instruction(convolution648, mx647, mx233);
    migraphx::op::batch_norm_inference batch_norm_inference649;
    batch_norm_inference649.epsilon  = 0.001;
    batch_norm_inference649.momentum = 0.9;
    auto mx649 = mm->add_instruction(batch_norm_inference649, mx648, mx232, mx231, mx230, mx229);
    migraphx::op::relu relu650;
    auto mx650 = mm->add_instruction(relu650, mx649);
    migraphx::op::convolution convolution651;
    convolution651.padding  = {0, 0};
    convolution651.stride   = {1, 1};
    convolution651.dilation = {1, 1};
    convolution651.group    = 1;
    auto mx651              = mm->add_instruction(convolution651, mx647, mx228);
    migraphx::op::batch_norm_inference batch_norm_inference652;
    batch_norm_inference652.epsilon  = 0.001;
    batch_norm_inference652.momentum = 0.9;
    auto mx652 = mm->add_instruction(batch_norm_inference652, mx651, mx227, mx226, mx225, mx224);
    migraphx::op::relu relu653;
    auto mx653 = mm->add_instruction(relu653, mx652);
    migraphx::op::convolution convolution654;
    convolution654.padding  = {0, 3};
    convolution654.stride   = {1, 1};
    convolution654.dilation = {1, 1};
    convolution654.group    = 1;
    auto mx654              = mm->add_instruction(convolution654, mx653, mx223);
    migraphx::op::batch_norm_inference batch_norm_inference655;
    batch_norm_inference655.epsilon  = 0.001;
    batch_norm_inference655.momentum = 0.9;
    auto mx655 = mm->add_instruction(batch_norm_inference655, mx654, mx222, mx221, mx220, mx219);
    migraphx::op::relu relu656;
    auto mx656 = mm->add_instruction(relu656, mx655);
    migraphx::op::convolution convolution657;
    convolution657.padding  = {3, 0};
    convolution657.stride   = {1, 1};
    convolution657.dilation = {1, 1};
    convolution657.group    = 1;
    auto mx657              = mm->add_instruction(convolution657, mx656, mx218);
    migraphx::op::batch_norm_inference batch_norm_inference658;
    batch_norm_inference658.epsilon  = 0.001;
    batch_norm_inference658.momentum = 0.9;
    auto mx658 = mm->add_instruction(batch_norm_inference658, mx657, mx217, mx216, mx215, mx214);
    migraphx::op::relu relu659;
    auto mx659 = mm->add_instruction(relu659, mx658);
    migraphx::op::convolution convolution660;
    convolution660.padding  = {0, 0};
    convolution660.stride   = {1, 1};
    convolution660.dilation = {1, 1};
    convolution660.group    = 1;
    auto mx660              = mm->add_instruction(convolution660, mx647, mx213);
    migraphx::op::batch_norm_inference batch_norm_inference661;
    batch_norm_inference661.epsilon  = 0.001;
    batch_norm_inference661.momentum = 0.9;
    auto mx661 = mm->add_instruction(batch_norm_inference661, mx660, mx212, mx211, mx210, mx209);
    migraphx::op::relu relu662;
    auto mx662 = mm->add_instruction(relu662, mx661);
    migraphx::op::convolution convolution663;
    convolution663.padding  = {3, 0};
    convolution663.stride   = {1, 1};
    convolution663.dilation = {1, 1};
    convolution663.group    = 1;
    auto mx663              = mm->add_instruction(convolution663, mx662, mx208);
    migraphx::op::batch_norm_inference batch_norm_inference664;
    batch_norm_inference664.epsilon  = 0.001;
    batch_norm_inference664.momentum = 0.9;
    auto mx664 = mm->add_instruction(batch_norm_inference664, mx663, mx207, mx206, mx205, mx204);
    migraphx::op::relu relu665;
    auto mx665 = mm->add_instruction(relu665, mx664);
    migraphx::op::convolution convolution666;
    convolution666.padding  = {0, 3};
    convolution666.stride   = {1, 1};
    convolution666.dilation = {1, 1};
    convolution666.group    = 1;
    auto mx666              = mm->add_instruction(convolution666, mx665, mx203);
    migraphx::op::batch_norm_inference batch_norm_inference667;
    batch_norm_inference667.epsilon  = 0.001;
    batch_norm_inference667.momentum = 0.9;
    auto mx667 = mm->add_instruction(batch_norm_inference667, mx666, mx202, mx201, mx200, mx199);
    migraphx::op::relu relu668;
    auto mx668 = mm->add_instruction(relu668, mx667);
    migraphx::op::convolution convolution669;
    convolution669.padding  = {3, 0};
    convolution669.stride   = {1, 1};
    convolution669.dilation = {1, 1};
    convolution669.group    = 1;
    auto mx669              = mm->add_instruction(convolution669, mx668, mx198);
    migraphx::op::batch_norm_inference batch_norm_inference670;
    batch_norm_inference670.epsilon  = 0.001;
    batch_norm_inference670.momentum = 0.9;
    auto mx670 = mm->add_instruction(batch_norm_inference670, mx669, mx197, mx196, mx195, mx194);
    migraphx::op::relu relu671;
    auto mx671 = mm->add_instruction(relu671, mx670);
    migraphx::op::convolution convolution672;
    convolution672.padding  = {0, 3};
    convolution672.stride   = {1, 1};
    convolution672.dilation = {1, 1};
    convolution672.group    = 1;
    auto mx672              = mm->add_instruction(convolution672, mx671, mx193);
    migraphx::op::batch_norm_inference batch_norm_inference673;
    batch_norm_inference673.epsilon  = 0.001;
    batch_norm_inference673.momentum = 0.9;
    auto mx673 = mm->add_instruction(batch_norm_inference673, mx672, mx192, mx191, mx190, mx189);
    migraphx::op::relu relu674;
    auto mx674 = mm->add_instruction(relu674, mx673);
    migraphx::op::pooling pooling675;
    pooling675.mode    = migraphx::op::pooling_mode::average;
    pooling675.padding = {1, 1};
    pooling675.stride  = {1, 1};
    pooling675.lengths = {3, 3};
    auto mx675         = mm->add_instruction(pooling675, mx647);
    migraphx::op::convolution convolution676;
    convolution676.padding  = {0, 0};
    convolution676.stride   = {1, 1};
    convolution676.dilation = {1, 1};
    convolution676.group    = 1;
    auto mx676              = mm->add_instruction(convolution676, mx675, mx188);
    migraphx::op::batch_norm_inference batch_norm_inference677;
    batch_norm_inference677.epsilon  = 0.001;
    batch_norm_inference677.momentum = 0.9;
    auto mx677 = mm->add_instruction(batch_norm_inference677, mx676, mx187, mx186, mx185, mx184);
    migraphx::op::relu relu678;
    auto mx678 = mm->add_instruction(relu678, mx677);
    migraphx::op::concat concat679;
    concat679.axis = 1;
    auto mx679     = mm->add_instruction(concat679, mx650, mx659, mx674, mx678);
    migraphx::op::convolution convolution680;
    convolution680.padding  = {0, 0};
    convolution680.stride   = {1, 1};
    convolution680.dilation = {1, 1};
    convolution680.group    = 1;
    auto mx680              = mm->add_instruction(convolution680, mx679, mx183);
    migraphx::op::batch_norm_inference batch_norm_inference681;
    batch_norm_inference681.epsilon  = 0.001;
    batch_norm_inference681.momentum = 0.9;
    auto mx681 = mm->add_instruction(batch_norm_inference681, mx680, mx182, mx181, mx180, mx179);
    migraphx::op::relu relu682;
    auto mx682 = mm->add_instruction(relu682, mx681);
    migraphx::op::convolution convolution683;
    convolution683.padding  = {0, 0};
    convolution683.stride   = {1, 1};
    convolution683.dilation = {1, 1};
    convolution683.group    = 1;
    auto mx683              = mm->add_instruction(convolution683, mx679, mx178);
    migraphx::op::batch_norm_inference batch_norm_inference684;
    batch_norm_inference684.epsilon  = 0.001;
    batch_norm_inference684.momentum = 0.9;
    auto mx684 = mm->add_instruction(batch_norm_inference684, mx683, mx177, mx176, mx175, mx174);
    migraphx::op::relu relu685;
    auto mx685 = mm->add_instruction(relu685, mx684);
    migraphx::op::convolution convolution686;
    convolution686.padding  = {0, 3};
    convolution686.stride   = {1, 1};
    convolution686.dilation = {1, 1};
    convolution686.group    = 1;
    auto mx686              = mm->add_instruction(convolution686, mx685, mx173);
    migraphx::op::batch_norm_inference batch_norm_inference687;
    batch_norm_inference687.epsilon  = 0.001;
    batch_norm_inference687.momentum = 0.9;
    auto mx687 = mm->add_instruction(batch_norm_inference687, mx686, mx172, mx171, mx170, mx169);
    migraphx::op::relu relu688;
    auto mx688 = mm->add_instruction(relu688, mx687);
    migraphx::op::convolution convolution689;
    convolution689.padding  = {3, 0};
    convolution689.stride   = {1, 1};
    convolution689.dilation = {1, 1};
    convolution689.group    = 1;
    auto mx689              = mm->add_instruction(convolution689, mx688, mx168);
    migraphx::op::batch_norm_inference batch_norm_inference690;
    batch_norm_inference690.epsilon  = 0.001;
    batch_norm_inference690.momentum = 0.9;
    auto mx690 = mm->add_instruction(batch_norm_inference690, mx689, mx167, mx166, mx165, mx164);
    migraphx::op::relu relu691;
    auto mx691 = mm->add_instruction(relu691, mx690);
    migraphx::op::convolution convolution692;
    convolution692.padding  = {0, 0};
    convolution692.stride   = {1, 1};
    convolution692.dilation = {1, 1};
    convolution692.group    = 1;
    auto mx692              = mm->add_instruction(convolution692, mx679, mx163);
    migraphx::op::batch_norm_inference batch_norm_inference693;
    batch_norm_inference693.epsilon  = 0.001;
    batch_norm_inference693.momentum = 0.9;
    auto mx693 = mm->add_instruction(batch_norm_inference693, mx692, mx162, mx161, mx160, mx159);
    migraphx::op::relu relu694;
    auto mx694 = mm->add_instruction(relu694, mx693);
    migraphx::op::convolution convolution695;
    convolution695.padding  = {3, 0};
    convolution695.stride   = {1, 1};
    convolution695.dilation = {1, 1};
    convolution695.group    = 1;
    auto mx695              = mm->add_instruction(convolution695, mx694, mx158);
    migraphx::op::batch_norm_inference batch_norm_inference696;
    batch_norm_inference696.epsilon  = 0.001;
    batch_norm_inference696.momentum = 0.9;
    auto mx696 = mm->add_instruction(batch_norm_inference696, mx695, mx157, mx156, mx155, mx154);
    migraphx::op::relu relu697;
    auto mx697 = mm->add_instruction(relu697, mx696);
    migraphx::op::convolution convolution698;
    convolution698.padding  = {0, 3};
    convolution698.stride   = {1, 1};
    convolution698.dilation = {1, 1};
    convolution698.group    = 1;
    auto mx698              = mm->add_instruction(convolution698, mx697, mx153);
    migraphx::op::batch_norm_inference batch_norm_inference699;
    batch_norm_inference699.epsilon  = 0.001;
    batch_norm_inference699.momentum = 0.9;
    auto mx699 = mm->add_instruction(batch_norm_inference699, mx698, mx152, mx151, mx150, mx149);
    migraphx::op::relu relu700;
    auto mx700 = mm->add_instruction(relu700, mx699);
    migraphx::op::convolution convolution701;
    convolution701.padding  = {3, 0};
    convolution701.stride   = {1, 1};
    convolution701.dilation = {1, 1};
    convolution701.group    = 1;
    auto mx701              = mm->add_instruction(convolution701, mx700, mx148);
    migraphx::op::batch_norm_inference batch_norm_inference702;
    batch_norm_inference702.epsilon  = 0.001;
    batch_norm_inference702.momentum = 0.9;
    auto mx702 = mm->add_instruction(batch_norm_inference702, mx701, mx147, mx146, mx145, mx144);
    migraphx::op::relu relu703;
    auto mx703 = mm->add_instruction(relu703, mx702);
    migraphx::op::convolution convolution704;
    convolution704.padding  = {0, 3};
    convolution704.stride   = {1, 1};
    convolution704.dilation = {1, 1};
    convolution704.group    = 1;
    auto mx704              = mm->add_instruction(convolution704, mx703, mx143);
    migraphx::op::batch_norm_inference batch_norm_inference705;
    batch_norm_inference705.epsilon  = 0.001;
    batch_norm_inference705.momentum = 0.9;
    auto mx705 = mm->add_instruction(batch_norm_inference705, mx704, mx142, mx141, mx140, mx139);
    migraphx::op::relu relu706;
    auto mx706 = mm->add_instruction(relu706, mx705);
    migraphx::op::pooling pooling707;
    pooling707.mode    = migraphx::op::pooling_mode::average;
    pooling707.padding = {1, 1};
    pooling707.stride  = {1, 1};
    pooling707.lengths = {3, 3};
    auto mx707         = mm->add_instruction(pooling707, mx679);
    migraphx::op::convolution convolution708;
    convolution708.padding  = {0, 0};
    convolution708.stride   = {1, 1};
    convolution708.dilation = {1, 1};
    convolution708.group    = 1;
    auto mx708              = mm->add_instruction(convolution708, mx707, mx138);
    migraphx::op::batch_norm_inference batch_norm_inference709;
    batch_norm_inference709.epsilon  = 0.001;
    batch_norm_inference709.momentum = 0.9;
    auto mx709 = mm->add_instruction(batch_norm_inference709, mx708, mx137, mx136, mx135, mx134);
    migraphx::op::relu relu710;
    auto mx710 = mm->add_instruction(relu710, mx709);
    migraphx::op::concat concat711;
    concat711.axis = 1;
    auto mx711     = mm->add_instruction(concat711, mx682, mx691, mx706, mx710);
    migraphx::op::convolution convolution712;
    convolution712.padding  = {0, 0};
    convolution712.stride   = {1, 1};
    convolution712.dilation = {1, 1};
    convolution712.group    = 1;
    auto mx712              = mm->add_instruction(convolution712, mx711, mx121);
    migraphx::op::batch_norm_inference batch_norm_inference713;
    batch_norm_inference713.epsilon  = 0.001;
    batch_norm_inference713.momentum = 0.9;
    auto mx713 = mm->add_instruction(batch_norm_inference713, mx712, mx120, mx119, mx118, mx117);
    migraphx::op::relu relu714;
    auto mx714 = mm->add_instruction(relu714, mx713);
    migraphx::op::convolution convolution715;
    convolution715.padding  = {0, 0};
    convolution715.stride   = {2, 2};
    convolution715.dilation = {1, 1};
    convolution715.group    = 1;
    auto mx715              = mm->add_instruction(convolution715, mx714, mx116);
    migraphx::op::batch_norm_inference batch_norm_inference716;
    batch_norm_inference716.epsilon  = 0.001;
    batch_norm_inference716.momentum = 0.9;
    auto mx716 = mm->add_instruction(batch_norm_inference716, mx715, mx115, mx114, mx113, mx112);
    migraphx::op::relu relu717;
    auto mx717 = mm->add_instruction(relu717, mx716);
    migraphx::op::convolution convolution718;
    convolution718.padding  = {0, 0};
    convolution718.stride   = {1, 1};
    convolution718.dilation = {1, 1};
    convolution718.group    = 1;
    auto mx718              = mm->add_instruction(convolution718, mx711, mx111);
    migraphx::op::batch_norm_inference batch_norm_inference719;
    batch_norm_inference719.epsilon  = 0.001;
    batch_norm_inference719.momentum = 0.9;
    auto mx719 = mm->add_instruction(batch_norm_inference719, mx718, mx110, mx109, mx108, mx107);
    migraphx::op::relu relu720;
    auto mx720 = mm->add_instruction(relu720, mx719);
    migraphx::op::convolution convolution721;
    convolution721.padding  = {0, 3};
    convolution721.stride   = {1, 1};
    convolution721.dilation = {1, 1};
    convolution721.group    = 1;
    auto mx721              = mm->add_instruction(convolution721, mx720, mx106);
    migraphx::op::batch_norm_inference batch_norm_inference722;
    batch_norm_inference722.epsilon  = 0.001;
    batch_norm_inference722.momentum = 0.9;
    auto mx722 = mm->add_instruction(batch_norm_inference722, mx721, mx105, mx104, mx103, mx102);
    migraphx::op::relu relu723;
    auto mx723 = mm->add_instruction(relu723, mx722);
    migraphx::op::convolution convolution724;
    convolution724.padding  = {3, 0};
    convolution724.stride   = {1, 1};
    convolution724.dilation = {1, 1};
    convolution724.group    = 1;
    auto mx724              = mm->add_instruction(convolution724, mx723, mx101);
    migraphx::op::batch_norm_inference batch_norm_inference725;
    batch_norm_inference725.epsilon  = 0.001;
    batch_norm_inference725.momentum = 0.9;
    auto mx725 = mm->add_instruction(batch_norm_inference725, mx724, mx100, mx99, mx98, mx97);
    migraphx::op::relu relu726;
    auto mx726 = mm->add_instruction(relu726, mx725);
    migraphx::op::convolution convolution727;
    convolution727.padding  = {0, 0};
    convolution727.stride   = {2, 2};
    convolution727.dilation = {1, 1};
    convolution727.group    = 1;
    auto mx727              = mm->add_instruction(convolution727, mx726, mx96);
    migraphx::op::batch_norm_inference batch_norm_inference728;
    batch_norm_inference728.epsilon  = 0.001;
    batch_norm_inference728.momentum = 0.9;
    auto mx728 = mm->add_instruction(batch_norm_inference728, mx727, mx95, mx94, mx93, mx92);
    migraphx::op::relu relu729;
    auto mx729 = mm->add_instruction(relu729, mx728);
    migraphx::op::pooling pooling730;
    pooling730.mode    = migraphx::op::pooling_mode::max;
    pooling730.padding = {0, 0};
    pooling730.stride  = {2, 2};
    pooling730.lengths = {3, 3};
    auto mx730         = mm->add_instruction(pooling730, mx711);
    migraphx::op::concat concat731;
    concat731.axis = 1;
    auto mx731     = mm->add_instruction(concat731, mx717, mx729, mx730);
    migraphx::op::convolution convolution732;
    convolution732.padding  = {0, 0};
    convolution732.stride   = {1, 1};
    convolution732.dilation = {1, 1};
    convolution732.group    = 1;
    auto mx732              = mm->add_instruction(convolution732, mx731, mx91);
    migraphx::op::batch_norm_inference batch_norm_inference733;
    batch_norm_inference733.epsilon  = 0.001;
    batch_norm_inference733.momentum = 0.9;
    auto mx733 = mm->add_instruction(batch_norm_inference733, mx732, mx90, mx89, mx88, mx87);
    migraphx::op::relu relu734;
    auto mx734 = mm->add_instruction(relu734, mx733);
    migraphx::op::convolution convolution735;
    convolution735.padding  = {0, 0};
    convolution735.stride   = {1, 1};
    convolution735.dilation = {1, 1};
    convolution735.group    = 1;
    auto mx735              = mm->add_instruction(convolution735, mx731, mx86);
    migraphx::op::batch_norm_inference batch_norm_inference736;
    batch_norm_inference736.epsilon  = 0.001;
    batch_norm_inference736.momentum = 0.9;
    auto mx736 = mm->add_instruction(batch_norm_inference736, mx735, mx85, mx84, mx83, mx82);
    migraphx::op::relu relu737;
    auto mx737 = mm->add_instruction(relu737, mx736);
    migraphx::op::convolution convolution738;
    convolution738.padding  = {0, 1};
    convolution738.stride   = {1, 1};
    convolution738.dilation = {1, 1};
    convolution738.group    = 1;
    auto mx738              = mm->add_instruction(convolution738, mx737, mx81);
    migraphx::op::batch_norm_inference batch_norm_inference739;
    batch_norm_inference739.epsilon  = 0.001;
    batch_norm_inference739.momentum = 0.9;
    auto mx739 = mm->add_instruction(batch_norm_inference739, mx738, mx80, mx79, mx78, mx77);
    migraphx::op::relu relu740;
    auto mx740 = mm->add_instruction(relu740, mx739);
    migraphx::op::convolution convolution741;
    convolution741.padding  = {1, 0};
    convolution741.stride   = {1, 1};
    convolution741.dilation = {1, 1};
    convolution741.group    = 1;
    auto mx741              = mm->add_instruction(convolution741, mx737, mx76);
    migraphx::op::batch_norm_inference batch_norm_inference742;
    batch_norm_inference742.epsilon  = 0.001;
    batch_norm_inference742.momentum = 0.9;
    auto mx742 = mm->add_instruction(batch_norm_inference742, mx741, mx75, mx74, mx73, mx72);
    migraphx::op::relu relu743;
    auto mx743 = mm->add_instruction(relu743, mx742);
    migraphx::op::concat concat744;
    concat744.axis = 1;
    auto mx744     = mm->add_instruction(concat744, mx740, mx743);
    migraphx::op::convolution convolution745;
    convolution745.padding  = {0, 0};
    convolution745.stride   = {1, 1};
    convolution745.dilation = {1, 1};
    convolution745.group    = 1;
    auto mx745              = mm->add_instruction(convolution745, mx731, mx71);
    migraphx::op::batch_norm_inference batch_norm_inference746;
    batch_norm_inference746.epsilon  = 0.001;
    batch_norm_inference746.momentum = 0.9;
    auto mx746 = mm->add_instruction(batch_norm_inference746, mx745, mx70, mx69, mx68, mx67);
    migraphx::op::relu relu747;
    auto mx747 = mm->add_instruction(relu747, mx746);
    migraphx::op::convolution convolution748;
    convolution748.padding  = {1, 1};
    convolution748.stride   = {1, 1};
    convolution748.dilation = {1, 1};
    convolution748.group    = 1;
    auto mx748              = mm->add_instruction(convolution748, mx747, mx66);
    migraphx::op::batch_norm_inference batch_norm_inference749;
    batch_norm_inference749.epsilon  = 0.001;
    batch_norm_inference749.momentum = 0.9;
    auto mx749 = mm->add_instruction(batch_norm_inference749, mx748, mx65, mx64, mx63, mx62);
    migraphx::op::relu relu750;
    auto mx750 = mm->add_instruction(relu750, mx749);
    migraphx::op::convolution convolution751;
    convolution751.padding  = {0, 1};
    convolution751.stride   = {1, 1};
    convolution751.dilation = {1, 1};
    convolution751.group    = 1;
    auto mx751              = mm->add_instruction(convolution751, mx750, mx61);
    migraphx::op::batch_norm_inference batch_norm_inference752;
    batch_norm_inference752.epsilon  = 0.001;
    batch_norm_inference752.momentum = 0.9;
    auto mx752 = mm->add_instruction(batch_norm_inference752, mx751, mx60, mx59, mx58, mx57);
    migraphx::op::relu relu753;
    auto mx753 = mm->add_instruction(relu753, mx752);
    migraphx::op::convolution convolution754;
    convolution754.padding  = {1, 0};
    convolution754.stride   = {1, 1};
    convolution754.dilation = {1, 1};
    convolution754.group    = 1;
    auto mx754              = mm->add_instruction(convolution754, mx750, mx56);
    migraphx::op::batch_norm_inference batch_norm_inference755;
    batch_norm_inference755.epsilon  = 0.001;
    batch_norm_inference755.momentum = 0.9;
    auto mx755 = mm->add_instruction(batch_norm_inference755, mx754, mx55, mx54, mx53, mx52);
    migraphx::op::relu relu756;
    auto mx756 = mm->add_instruction(relu756, mx755);
    migraphx::op::concat concat757;
    concat757.axis = 1;
    auto mx757     = mm->add_instruction(concat757, mx753, mx756);
    migraphx::op::pooling pooling758;
    pooling758.mode    = migraphx::op::pooling_mode::average;
    pooling758.padding = {1, 1};
    pooling758.stride  = {1, 1};
    pooling758.lengths = {3, 3};
    auto mx758         = mm->add_instruction(pooling758, mx731);
    migraphx::op::convolution convolution759;
    convolution759.padding  = {0, 0};
    convolution759.stride   = {1, 1};
    convolution759.dilation = {1, 1};
    convolution759.group    = 1;
    auto mx759              = mm->add_instruction(convolution759, mx758, mx51);
    migraphx::op::batch_norm_inference batch_norm_inference760;
    batch_norm_inference760.epsilon  = 0.001;
    batch_norm_inference760.momentum = 0.9;
    auto mx760 = mm->add_instruction(batch_norm_inference760, mx759, mx50, mx49, mx48, mx47);
    migraphx::op::relu relu761;
    auto mx761 = mm->add_instruction(relu761, mx760);
    migraphx::op::concat concat762;
    concat762.axis = 1;
    auto mx762     = mm->add_instruction(concat762, mx734, mx744, mx757, mx761);
    migraphx::op::convolution convolution763;
    convolution763.padding  = {0, 0};
    convolution763.stride   = {1, 1};
    convolution763.dilation = {1, 1};
    convolution763.group    = 1;
    auto mx763              = mm->add_instruction(convolution763, mx762, mx46);
    migraphx::op::batch_norm_inference batch_norm_inference764;
    batch_norm_inference764.epsilon  = 0.001;
    batch_norm_inference764.momentum = 0.9;
    auto mx764 = mm->add_instruction(batch_norm_inference764, mx763, mx45, mx44, mx43, mx42);
    migraphx::op::relu relu765;
    auto mx765 = mm->add_instruction(relu765, mx764);
    migraphx::op::convolution convolution766;
    convolution766.padding  = {0, 0};
    convolution766.stride   = {1, 1};
    convolution766.dilation = {1, 1};
    convolution766.group    = 1;
    auto mx766              = mm->add_instruction(convolution766, mx762, mx41);
    migraphx::op::batch_norm_inference batch_norm_inference767;
    batch_norm_inference767.epsilon  = 0.001;
    batch_norm_inference767.momentum = 0.9;
    auto mx767 = mm->add_instruction(batch_norm_inference767, mx766, mx40, mx39, mx38, mx37);
    migraphx::op::relu relu768;
    auto mx768 = mm->add_instruction(relu768, mx767);
    migraphx::op::convolution convolution769;
    convolution769.padding  = {0, 1};
    convolution769.stride   = {1, 1};
    convolution769.dilation = {1, 1};
    convolution769.group    = 1;
    auto mx769              = mm->add_instruction(convolution769, mx768, mx36);
    migraphx::op::batch_norm_inference batch_norm_inference770;
    batch_norm_inference770.epsilon  = 0.001;
    batch_norm_inference770.momentum = 0.9;
    auto mx770 = mm->add_instruction(batch_norm_inference770, mx769, mx35, mx34, mx33, mx32);
    migraphx::op::relu relu771;
    auto mx771 = mm->add_instruction(relu771, mx770);
    migraphx::op::convolution convolution772;
    convolution772.padding  = {1, 0};
    convolution772.stride   = {1, 1};
    convolution772.dilation = {1, 1};
    convolution772.group    = 1;
    auto mx772              = mm->add_instruction(convolution772, mx768, mx31);
    migraphx::op::batch_norm_inference batch_norm_inference773;
    batch_norm_inference773.epsilon  = 0.001;
    batch_norm_inference773.momentum = 0.9;
    auto mx773 = mm->add_instruction(batch_norm_inference773, mx772, mx30, mx29, mx28, mx27);
    migraphx::op::relu relu774;
    auto mx774 = mm->add_instruction(relu774, mx773);
    migraphx::op::concat concat775;
    concat775.axis = 1;
    auto mx775     = mm->add_instruction(concat775, mx771, mx774);
    migraphx::op::convolution convolution776;
    convolution776.padding  = {0, 0};
    convolution776.stride   = {1, 1};
    convolution776.dilation = {1, 1};
    convolution776.group    = 1;
    auto mx776              = mm->add_instruction(convolution776, mx762, mx26);
    migraphx::op::batch_norm_inference batch_norm_inference777;
    batch_norm_inference777.epsilon  = 0.001;
    batch_norm_inference777.momentum = 0.9;
    auto mx777 = mm->add_instruction(batch_norm_inference777, mx776, mx25, mx24, mx23, mx22);
    migraphx::op::relu relu778;
    auto mx778 = mm->add_instruction(relu778, mx777);
    migraphx::op::convolution convolution779;
    convolution779.padding  = {1, 1};
    convolution779.stride   = {1, 1};
    convolution779.dilation = {1, 1};
    convolution779.group    = 1;
    auto mx779              = mm->add_instruction(convolution779, mx778, mx21);
    migraphx::op::batch_norm_inference batch_norm_inference780;
    batch_norm_inference780.epsilon  = 0.001;
    batch_norm_inference780.momentum = 0.9;
    auto mx780 = mm->add_instruction(batch_norm_inference780, mx779, mx20, mx19, mx18, mx17);
    migraphx::op::relu relu781;
    auto mx781 = mm->add_instruction(relu781, mx780);
    migraphx::op::convolution convolution782;
    convolution782.padding  = {0, 1};
    convolution782.stride   = {1, 1};
    convolution782.dilation = {1, 1};
    convolution782.group    = 1;
    auto mx782              = mm->add_instruction(convolution782, mx781, mx16);
    migraphx::op::batch_norm_inference batch_norm_inference783;
    batch_norm_inference783.epsilon  = 0.001;
    batch_norm_inference783.momentum = 0.9;
    auto mx783 = mm->add_instruction(batch_norm_inference783, mx782, mx15, mx14, mx13, mx12);
    migraphx::op::relu relu784;
    auto mx784 = mm->add_instruction(relu784, mx783);
    migraphx::op::convolution convolution785;
    convolution785.padding  = {1, 0};
    convolution785.stride   = {1, 1};
    convolution785.dilation = {1, 1};
    convolution785.group    = 1;
    auto mx785              = mm->add_instruction(convolution785, mx781, mx11);
    migraphx::op::batch_norm_inference batch_norm_inference786;
    batch_norm_inference786.epsilon  = 0.001;
    batch_norm_inference786.momentum = 0.9;
    auto mx786 = mm->add_instruction(batch_norm_inference786, mx785, mx10, mx9, mx8, mx7);
    migraphx::op::relu relu787;
    auto mx787 = mm->add_instruction(relu787, mx786);
    migraphx::op::concat concat788;
    concat788.axis = 1;
    auto mx788     = mm->add_instruction(concat788, mx784, mx787);
    migraphx::op::pooling pooling789;
    pooling789.mode    = migraphx::op::pooling_mode::average;
    pooling789.padding = {1, 1};
    pooling789.stride  = {1, 1};
    pooling789.lengths = {3, 3};
    auto mx789         = mm->add_instruction(pooling789, mx762);
    migraphx::op::convolution convolution790;
    convolution790.padding  = {0, 0};
    convolution790.stride   = {1, 1};
    convolution790.dilation = {1, 1};
    convolution790.group    = 1;
    auto mx790              = mm->add_instruction(convolution790, mx789, mx6);
    migraphx::op::batch_norm_inference batch_norm_inference791;
    batch_norm_inference791.epsilon  = 0.001;
    batch_norm_inference791.momentum = 0.9;
    auto mx791 = mm->add_instruction(batch_norm_inference791, mx790, mx5, mx4, mx3, mx2);
    migraphx::op::relu relu792;
    auto mx792 = mm->add_instruction(relu792, mx791);
    migraphx::op::concat concat793;
    concat793.axis = 1;
    auto mx793     = mm->add_instruction(concat793, mx765, mx775, mx788, mx792);
    migraphx::op::pooling pooling794;
    pooling794.mode    = migraphx::op::pooling_mode::average;
    pooling794.padding = {0, 0};
    pooling794.stride  = {8, 8};
    pooling794.lengths = {8, 8};
    auto mx794         = mm->add_instruction(pooling794, mx793);
    migraphx::op::identity identity795;
    auto mx795 = mm->add_instruction(identity795, mx794);
    migraphx::op::flatten flatten796;
    flatten796.axis = 1;
    auto mx796      = mm->add_instruction(flatten796, mx795);
    migraphx::op::transpose transpose797;
    transpose797.dims = {1, 0};
    auto mx797        = mm->add_instruction(transpose797, mx1);
    migraphx::op::multibroadcast multibroadcast798;
    multibroadcast798.output_lens = {batch, 1000};
    auto mx798                    = mm->add_instruction(multibroadcast798, mx0);
    float dot799_alpha            = 1;
    float dot799_beta             = 1;
    migraphx::add_apply_alpha_beta(
        *mm, {mx796, mx797, mx798}, migraphx::make_op("dot"), dot799_alpha, dot799_beta);

    return p;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx
