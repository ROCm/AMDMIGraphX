#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include "models.hpp"

namespace migraphx {
namespace driver {
inline namespace MIGRAPHX_INLINE_NS {

migraphx::program inceptionv3(unsigned batch)
{
    migraphx::program p;
    auto m0 =
        p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {batch, 3, 299, 299}});
    auto mx0 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1000}}, 0));
    auto mx1 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1000, 2048}}, 1));
    auto mx2 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 2)));
    auto mx3 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 3));
    auto mx4 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 4));
    auto mx5 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 5)));
    auto mx6 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 2048, 1, 1}}, 6));
    auto mx7 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 7)));
    auto mx8 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 8));
    auto mx9 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 9));
    auto mx10 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 10)));
    auto mx11 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 3, 1}}, 11));
    auto mx12 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 12)));
    auto mx13 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 13));
    auto mx14 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 14));
    auto mx15 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 15)));
    auto mx16 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 1, 3}}, 16));
    auto mx17 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 17)));
    auto mx18 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 18));
    auto mx19 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 19));
    auto mx20 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 20)));
    auto mx21 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 448, 3, 3}}, 21));
    auto mx22 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 22)));
    auto mx23 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 23));
    auto mx24 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 24));
    auto mx25 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 25)));
    auto mx26 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {448, 2048, 1, 1}}, 26));
    auto mx27 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 27)));
    auto mx28 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 28)));
    auto mx29 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 29));
    auto mx30 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 30)));
    auto mx31 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 3, 1}}, 31));
    auto mx32 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 32)));
    auto mx33 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 33)));
    auto mx34 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 34));
    auto mx35 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 35)));
    auto mx36 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 1, 3}}, 36));
    auto mx37 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 37)));
    auto mx38 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 38));
    auto mx39 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 39));
    auto mx40 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 40)));
    auto mx41 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 2048, 1, 1}}, 41));
    auto mx42 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 42)));
    auto mx43 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 43));
    auto mx44 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 44));
    auto mx45 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 45)));
    auto mx46 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {320, 2048, 1, 1}}, 46));
    auto mx47 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 47)));
    auto mx48 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 48));
    auto mx49 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 49));
    auto mx50 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 50)));
    auto mx51 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 1280, 1, 1}}, 51));
    auto mx52 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 52)));
    auto mx53 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 53));
    auto mx54 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 54));
    auto mx55 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 55)));
    auto mx56 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 3, 1}}, 56));
    auto mx57 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 57)));
    auto mx58 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 58));
    auto mx59 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 59));
    auto mx60 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 60)));
    auto mx61 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 1, 3}}, 61));
    auto mx62 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 62)));
    auto mx63 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 63));
    auto mx64 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 64));
    auto mx65 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 65)));
    auto mx66 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 448, 3, 3}}, 66));
    auto mx67 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 67)));
    auto mx68 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 68));
    auto mx69 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 69));
    auto mx70 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {448}}, 70)));
    auto mx71 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {448, 1280, 1, 1}}, 71));
    auto mx72 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 72)));
    auto mx73 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 73));
    auto mx74 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 74));
    auto mx75 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 75)));
    auto mx76 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 3, 1}}, 76));
    auto mx77 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 77)));
    auto mx78 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 78));
    auto mx79 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 79));
    auto mx80 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 80)));
    auto mx81 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 384, 1, 3}}, 81));
    auto mx82 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 82)));
    auto mx83 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 83));
    auto mx84 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 84));
    auto mx85 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 85)));
    auto mx86 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 1280, 1, 1}}, 86));
    auto mx87 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 87)));
    auto mx88 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 88));
    auto mx89 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 89));
    auto mx90 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 90)));
    auto mx91 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {320, 1280, 1, 1}}, 91));
    auto mx92 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 92)));
    auto mx93 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 93));
    auto mx94 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 94));
    auto mx95 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 95)));
    auto mx96 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 3, 3}}, 96));
    auto mx97 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 97)));
    auto mx98 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 98));
    auto mx99 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 99));
    auto mx100 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 100)));
    auto mx101 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 7, 1}}, 101));
    auto mx102 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 102)));
    auto mx103 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 103));
    auto mx104 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 104));
    auto mx105 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 105)));
    auto mx106 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 1, 7}}, 106));
    auto mx107 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 107)));
    auto mx108 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 108));
    auto mx109 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 109));
    auto mx110 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 110)));
    auto mx111 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 111));
    auto mx112 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 112)));
    auto mx113 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 113));
    auto mx114 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 114));
    auto mx115 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {320}}, 115)));
    auto mx116 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {320, 192, 3, 3}}, 116));
    auto mx117 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 117)));
    auto mx118 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 118));
    auto mx119 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 119));
    auto mx120 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 120)));
    auto mx121 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 121));
    auto mx134 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 134)));
    auto mx135 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 135));
    auto mx136 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 136));
    auto mx137 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 137)));
    auto mx138 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 138));
    auto mx139 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 139)));
    auto mx140 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 140));
    auto mx141 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 141));
    auto mx142 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 142)));
    auto mx143 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 1, 7}}, 143));
    auto mx144 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 144)));
    auto mx145 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 145));
    auto mx146 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 146));
    auto mx147 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 147)));
    auto mx148 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 7, 1}}, 148));
    auto mx149 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 149)));
    auto mx150 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 150));
    auto mx151 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 151));
    auto mx152 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 152)));
    auto mx153 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 1, 7}}, 153));
    auto mx154 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 154)));
    auto mx155 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 155));
    auto mx156 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 156));
    auto mx157 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 157)));
    auto mx158 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 7, 1}}, 158));
    auto mx159 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 159)));
    auto mx160 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 160));
    auto mx161 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 161));
    auto mx162 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 162)));
    auto mx163 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 163));
    auto mx164 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 164)));
    auto mx165 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 165));
    auto mx166 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 166));
    auto mx167 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 167)));
    auto mx168 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 7, 1}}, 168));
    auto mx169 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 169)));
    auto mx170 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 170));
    auto mx171 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 171));
    auto mx172 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 172)));
    auto mx173 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 192, 1, 7}}, 173));
    auto mx174 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 174)));
    auto mx175 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 175));
    auto mx176 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 176));
    auto mx177 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 177)));
    auto mx178 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 178));
    auto mx179 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 179)));
    auto mx180 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 180));
    auto mx181 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 181));
    auto mx182 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 182)));
    auto mx183 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 183));
    auto mx184 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 184)));
    auto mx185 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 185));
    auto mx186 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 186));
    auto mx187 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 187)));
    auto mx188 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 188));
    auto mx189 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 189)));
    auto mx190 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 190));
    auto mx191 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 191));
    auto mx192 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 192)));
    auto mx193 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 160, 1, 7}}, 193));
    auto mx194 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 194)));
    auto mx195 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 195));
    auto mx196 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 196));
    auto mx197 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 197)));
    auto mx198 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 7, 1}}, 198));
    auto mx199 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 199)));
    auto mx200 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 200));
    auto mx201 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 201));
    auto mx202 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 202)));
    auto mx203 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 1, 7}}, 203));
    auto mx204 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 204)));
    auto mx205 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 205));
    auto mx206 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 206));
    auto mx207 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 207)));
    auto mx208 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 7, 1}}, 208));
    auto mx209 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 209)));
    auto mx210 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 210));
    auto mx211 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 211));
    auto mx212 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 212)));
    auto mx213 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 768, 1, 1}}, 213));
    auto mx214 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 214)));
    auto mx215 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 215));
    auto mx216 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 216));
    auto mx217 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 217)));
    auto mx218 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 160, 7, 1}}, 218));
    auto mx219 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 219)));
    auto mx220 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 220));
    auto mx221 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 221));
    auto mx222 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 222)));
    auto mx223 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 1, 7}}, 223));
    auto mx224 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 224)));
    auto mx225 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 225));
    auto mx226 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 226));
    auto mx227 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 227)));
    auto mx228 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 768, 1, 1}}, 228));
    auto mx229 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 229)));
    auto mx230 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 230));
    auto mx231 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 231));
    auto mx232 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 232)));
    auto mx233 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 233));
    auto mx234 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 234)));
    auto mx235 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 235));
    auto mx236 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 236));
    auto mx237 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 237)));
    auto mx238 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 238));
    auto mx239 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 239)));
    auto mx240 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 240));
    auto mx241 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 241));
    auto mx242 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 242)));
    auto mx243 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 160, 1, 7}}, 243));
    auto mx244 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 244)));
    auto mx245 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 245));
    auto mx246 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 246));
    auto mx247 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 247)));
    auto mx248 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 7, 1}}, 248));
    auto mx249 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 249)));
    auto mx250 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 250));
    auto mx251 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 251));
    auto mx252 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 252)));
    auto mx253 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 1, 7}}, 253));
    auto mx254 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 254)));
    auto mx255 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 255));
    auto mx256 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 256));
    auto mx257 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 257)));
    auto mx258 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 7, 1}}, 258));
    auto mx259 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 259)));
    auto mx260 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 260));
    auto mx261 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 261));
    auto mx262 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 262)));
    auto mx263 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 768, 1, 1}}, 263));
    auto mx264 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 264)));
    auto mx265 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 265));
    auto mx266 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 266));
    auto mx267 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 267)));
    auto mx268 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 160, 7, 1}}, 268));
    auto mx269 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 269)));
    auto mx270 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 270));
    auto mx271 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 271));
    auto mx272 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 272)));
    auto mx273 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 160, 1, 7}}, 273));
    auto mx274 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 274)));
    auto mx275 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 275));
    auto mx276 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 276));
    auto mx277 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {160}}, 277)));
    auto mx278 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {160, 768, 1, 1}}, 278));
    auto mx279 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 279)));
    auto mx280 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 280));
    auto mx281 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 281));
    auto mx282 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 282)));
    auto mx283 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 283));
    auto mx284 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 284)));
    auto mx285 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 285));
    auto mx286 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 286));
    auto mx287 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 287)));
    auto mx288 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 288));
    auto mx289 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 289)));
    auto mx290 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 290));
    auto mx291 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 291));
    auto mx292 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 292)));
    auto mx293 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 128, 1, 7}}, 293));
    auto mx294 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 294)));
    auto mx295 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 295));
    auto mx296 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 296));
    auto mx297 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 297)));
    auto mx298 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 7, 1}}, 298));
    auto mx299 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 299)));
    auto mx300 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 300));
    auto mx301 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 301));
    auto mx302 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 302)));
    auto mx303 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 1, 7}}, 303));
    auto mx304 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 304)));
    auto mx305 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 305));
    auto mx306 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 306));
    auto mx307 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 307)));
    auto mx308 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 7, 1}}, 308));
    auto mx309 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 309)));
    auto mx310 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 310));
    auto mx311 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 311));
    auto mx312 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 312)));
    auto mx313 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 768, 1, 1}}, 313));
    auto mx314 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 314)));
    auto mx315 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 315));
    auto mx316 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 316));
    auto mx317 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 317)));
    auto mx318 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 128, 7, 1}}, 318));
    auto mx319 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 319)));
    auto mx320 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 320));
    auto mx321 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 321));
    auto mx322 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 322)));
    auto mx323 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 1, 7}}, 323));
    auto mx324 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 324)));
    auto mx325 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 325));
    auto mx326 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 326));
    auto mx327 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 327)));
    auto mx328 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 768, 1, 1}}, 328));
    auto mx329 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 329)));
    auto mx330 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 330));
    auto mx331 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 331));
    auto mx332 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 332)));
    auto mx333 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 768, 1, 1}}, 333));
    auto mx334 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 334)));
    auto mx335 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 335));
    auto mx336 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 336));
    auto mx337 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 337)));
    auto mx338 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 96, 3, 3}}, 338));
    auto mx339 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 339)));
    auto mx340 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 340));
    auto mx341 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 341));
    auto mx342 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 342)));
    auto mx343 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 64, 3, 3}}, 343));
    auto mx344 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 344)));
    auto mx345 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 345));
    auto mx346 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 346));
    auto mx347 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 347)));
    auto mx348 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 288, 1, 1}}, 348));
    auto mx349 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 349)));
    auto mx350 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 350));
    auto mx351 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 351));
    auto mx352 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 352)));
    auto mx353 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 288, 3, 3}}, 353));
    auto mx354 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 354)));
    auto mx355 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 355));
    auto mx356 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 356));
    auto mx357 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 357)));
    auto mx358 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 288, 1, 1}}, 358));
    auto mx359 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 359)));
    auto mx360 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 360));
    auto mx361 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 361));
    auto mx362 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 362)));
    auto mx363 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 96, 3, 3}}, 363));
    auto mx364 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 364)));
    auto mx365 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 365));
    auto mx366 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 366));
    auto mx367 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 367)));
    auto mx368 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 64, 3, 3}}, 368));
    auto mx369 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 369)));
    auto mx370 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 370));
    auto mx371 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 371));
    auto mx372 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 372)));
    auto mx373 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 288, 1, 1}}, 373));
    auto mx374 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 374)));
    auto mx375 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 375));
    auto mx376 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 376));
    auto mx377 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 377)));
    auto mx378 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 48, 5, 5}}, 378));
    auto mx379 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 379)));
    auto mx380 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 380));
    auto mx381 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 381));
    auto mx382 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 382)));
    auto mx383 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {48, 288, 1, 1}}, 383));
    auto mx384 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 384)));
    auto mx385 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 385));
    auto mx386 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 386));
    auto mx387 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 387)));
    auto mx388 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 288, 1, 1}}, 388));
    auto mx389 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 389)));
    auto mx390 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 390));
    auto mx391 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 391));
    auto mx392 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 392)));
    auto mx393 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 256, 1, 1}}, 393));
    auto mx394 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 394)));
    auto mx395 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 395));
    auto mx396 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 396));
    auto mx397 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 397)));
    auto mx398 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 96, 3, 3}}, 398));
    auto mx399 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 399)));
    auto mx400 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 400));
    auto mx401 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 401));
    auto mx402 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 402)));
    auto mx403 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 64, 3, 3}}, 403));
    auto mx404 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 404)));
    auto mx405 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 405));
    auto mx406 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 406));
    auto mx407 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 407)));
    auto mx408 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 256, 1, 1}}, 408));
    auto mx409 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 409)));
    auto mx410 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 410));
    auto mx411 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 411));
    auto mx412 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 412)));
    auto mx413 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 48, 5, 5}}, 413));
    auto mx414 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 414)));
    auto mx415 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 415));
    auto mx416 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 416));
    auto mx417 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 417)));
    auto mx418 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {48, 256, 1, 1}}, 418));
    auto mx419 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 419)));
    auto mx420 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 420));
    auto mx421 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 421));
    auto mx422 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 422)));
    auto mx423 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 256, 1, 1}}, 423));
    auto mx424 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 424)));
    auto mx425 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 425));
    auto mx426 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 426));
    auto mx427 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 427)));
    auto mx428 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {32, 192, 1, 1}}, 428));
    auto mx429 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 429)));
    auto mx430 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 430));
    auto mx431 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 431));
    auto mx432 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 432)));
    auto mx433 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 96, 3, 3}}, 433));
    auto mx434 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 434)));
    auto mx435 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 435));
    auto mx436 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 436));
    auto mx437 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {96}}, 437)));
    auto mx438 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {96, 64, 3, 3}}, 438));
    auto mx439 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 439)));
    auto mx440 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 440));
    auto mx441 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 441));
    auto mx442 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 442)));
    auto mx443 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 192, 1, 1}}, 443));
    auto mx444 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 444)));
    auto mx445 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 445));
    auto mx446 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 446));
    auto mx447 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 447)));
    auto mx448 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 48, 5, 5}}, 448));
    auto mx449 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 449)));
    auto mx450 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 450));
    auto mx451 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 451));
    auto mx452 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {48}}, 452)));
    auto mx453 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {48, 192, 1, 1}}, 453));
    auto mx454 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 454)));
    auto mx455 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 455));
    auto mx456 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 456));
    auto mx457 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 457)));
    auto mx458 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 192, 1, 1}}, 458));
    auto mx459 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 459)));
    auto mx460 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 460));
    auto mx461 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 461));
    auto mx462 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 462)));
    auto mx463 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 80, 3, 3}}, 463));
    auto mx464 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {80}}, 464)));
    auto mx465 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {80}}, 465));
    auto mx466 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {80}}, 466));
    auto mx467 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {80}}, 467)));
    auto mx468 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {80, 64, 1, 1}}, 468));
    auto mx469 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 469)));
    auto mx470 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 470));
    auto mx471 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 471));
    auto mx472 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 472)));
    auto mx473 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 32, 3, 3}}, 473));
    auto mx474 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 474)));
    auto mx475 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 475));
    auto mx476 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 476));
    auto mx477 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 477)));
    auto mx478 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {32, 32, 3, 3}}, 478));
    auto mx479 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 479)));
    auto mx480 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 480));
    auto mx481 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 481));
    auto mx482 = p.add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {32}}, 482)));
    auto mx483 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {32, 3, 3, 3}}, 483));
    migraphx::op::convolution convolution484;
    convolution484.padding  = {0, 0};
    convolution484.stride   = {2, 2};
    convolution484.dilation = {1, 1};
    convolution484.group    = 1;
    auto mx484              = p.add_instruction(convolution484, m0, mx483);
    migraphx::op::batch_norm_inference batch_norm_inference485;
    batch_norm_inference485.epsilon  = 0.001;
    batch_norm_inference485.momentum = 0.9;
    auto mx485 = p.add_instruction(batch_norm_inference485, mx484, mx482, mx481, mx480, mx479);
    migraphx::op::relu relu486;
    auto mx486 = p.add_instruction(relu486, mx485);
    migraphx::op::convolution convolution487;
    convolution487.padding  = {0, 0};
    convolution487.stride   = {1, 1};
    convolution487.dilation = {1, 1};
    convolution487.group    = 1;
    auto mx487              = p.add_instruction(convolution487, mx486, mx478);
    migraphx::op::batch_norm_inference batch_norm_inference488;
    batch_norm_inference488.epsilon  = 0.001;
    batch_norm_inference488.momentum = 0.9;
    auto mx488 = p.add_instruction(batch_norm_inference488, mx487, mx477, mx476, mx475, mx474);
    migraphx::op::relu relu489;
    auto mx489 = p.add_instruction(relu489, mx488);
    migraphx::op::convolution convolution490;
    convolution490.padding  = {1, 1};
    convolution490.stride   = {1, 1};
    convolution490.dilation = {1, 1};
    convolution490.group    = 1;
    auto mx490              = p.add_instruction(convolution490, mx489, mx473);
    migraphx::op::batch_norm_inference batch_norm_inference491;
    batch_norm_inference491.epsilon  = 0.001;
    batch_norm_inference491.momentum = 0.9;
    auto mx491 = p.add_instruction(batch_norm_inference491, mx490, mx472, mx471, mx470, mx469);
    migraphx::op::relu relu492;
    auto mx492 = p.add_instruction(relu492, mx491);
    migraphx::op::pooling pooling493;
    pooling493.mode    = "max ";
    pooling493.padding = {0, 0};
    pooling493.stride  = {2, 2};
    pooling493.lengths = {3, 3};
    auto mx493         = p.add_instruction(pooling493, mx492);
    migraphx::op::convolution convolution494;
    convolution494.padding  = {0, 0};
    convolution494.stride   = {1, 1};
    convolution494.dilation = {1, 1};
    convolution494.group    = 1;
    auto mx494              = p.add_instruction(convolution494, mx493, mx468);
    migraphx::op::batch_norm_inference batch_norm_inference495;
    batch_norm_inference495.epsilon  = 0.001;
    batch_norm_inference495.momentum = 0.9;
    auto mx495 = p.add_instruction(batch_norm_inference495, mx494, mx467, mx466, mx465, mx464);
    migraphx::op::relu relu496;
    auto mx496 = p.add_instruction(relu496, mx495);
    migraphx::op::convolution convolution497;
    convolution497.padding  = {0, 0};
    convolution497.stride   = {1, 1};
    convolution497.dilation = {1, 1};
    convolution497.group    = 1;
    auto mx497              = p.add_instruction(convolution497, mx496, mx463);
    migraphx::op::batch_norm_inference batch_norm_inference498;
    batch_norm_inference498.epsilon  = 0.001;
    batch_norm_inference498.momentum = 0.9;
    auto mx498 = p.add_instruction(batch_norm_inference498, mx497, mx462, mx461, mx460, mx459);
    migraphx::op::relu relu499;
    auto mx499 = p.add_instruction(relu499, mx498);
    migraphx::op::pooling pooling500;
    pooling500.mode    = "max ";
    pooling500.padding = {0, 0};
    pooling500.stride  = {2, 2};
    pooling500.lengths = {3, 3};
    auto mx500         = p.add_instruction(pooling500, mx499);
    migraphx::op::convolution convolution501;
    convolution501.padding  = {0, 0};
    convolution501.stride   = {1, 1};
    convolution501.dilation = {1, 1};
    convolution501.group    = 1;
    auto mx501              = p.add_instruction(convolution501, mx500, mx458);
    migraphx::op::batch_norm_inference batch_norm_inference502;
    batch_norm_inference502.epsilon  = 0.001;
    batch_norm_inference502.momentum = 0.9;
    auto mx502 = p.add_instruction(batch_norm_inference502, mx501, mx457, mx456, mx455, mx454);
    migraphx::op::relu relu503;
    auto mx503 = p.add_instruction(relu503, mx502);
    migraphx::op::convolution convolution504;
    convolution504.padding  = {0, 0};
    convolution504.stride   = {1, 1};
    convolution504.dilation = {1, 1};
    convolution504.group    = 1;
    auto mx504              = p.add_instruction(convolution504, mx500, mx453);
    migraphx::op::batch_norm_inference batch_norm_inference505;
    batch_norm_inference505.epsilon  = 0.001;
    batch_norm_inference505.momentum = 0.9;
    auto mx505 = p.add_instruction(batch_norm_inference505, mx504, mx452, mx451, mx450, mx449);
    migraphx::op::relu relu506;
    auto mx506 = p.add_instruction(relu506, mx505);
    migraphx::op::convolution convolution507;
    convolution507.padding  = {2, 2};
    convolution507.stride   = {1, 1};
    convolution507.dilation = {1, 1};
    convolution507.group    = 1;
    auto mx507              = p.add_instruction(convolution507, mx506, mx448);
    migraphx::op::batch_norm_inference batch_norm_inference508;
    batch_norm_inference508.epsilon  = 0.001;
    batch_norm_inference508.momentum = 0.9;
    auto mx508 = p.add_instruction(batch_norm_inference508, mx507, mx447, mx446, mx445, mx444);
    migraphx::op::relu relu509;
    auto mx509 = p.add_instruction(relu509, mx508);
    migraphx::op::convolution convolution510;
    convolution510.padding  = {0, 0};
    convolution510.stride   = {1, 1};
    convolution510.dilation = {1, 1};
    convolution510.group    = 1;
    auto mx510              = p.add_instruction(convolution510, mx500, mx443);
    migraphx::op::batch_norm_inference batch_norm_inference511;
    batch_norm_inference511.epsilon  = 0.001;
    batch_norm_inference511.momentum = 0.9;
    auto mx511 = p.add_instruction(batch_norm_inference511, mx510, mx442, mx441, mx440, mx439);
    migraphx::op::relu relu512;
    auto mx512 = p.add_instruction(relu512, mx511);
    migraphx::op::convolution convolution513;
    convolution513.padding  = {1, 1};
    convolution513.stride   = {1, 1};
    convolution513.dilation = {1, 1};
    convolution513.group    = 1;
    auto mx513              = p.add_instruction(convolution513, mx512, mx438);
    migraphx::op::batch_norm_inference batch_norm_inference514;
    batch_norm_inference514.epsilon  = 0.001;
    batch_norm_inference514.momentum = 0.9;
    auto mx514 = p.add_instruction(batch_norm_inference514, mx513, mx437, mx436, mx435, mx434);
    migraphx::op::relu relu515;
    auto mx515 = p.add_instruction(relu515, mx514);
    migraphx::op::convolution convolution516;
    convolution516.padding  = {1, 1};
    convolution516.stride   = {1, 1};
    convolution516.dilation = {1, 1};
    convolution516.group    = 1;
    auto mx516              = p.add_instruction(convolution516, mx515, mx433);
    migraphx::op::batch_norm_inference batch_norm_inference517;
    batch_norm_inference517.epsilon  = 0.001;
    batch_norm_inference517.momentum = 0.9;
    auto mx517 = p.add_instruction(batch_norm_inference517, mx516, mx432, mx431, mx430, mx429);
    migraphx::op::relu relu518;
    auto mx518 = p.add_instruction(relu518, mx517);
    migraphx::op::pooling pooling519;
    pooling519.mode    = "average ";
    pooling519.padding = {1, 1};
    pooling519.stride  = {1, 1};
    pooling519.lengths = {3, 3};
    auto mx519         = p.add_instruction(pooling519, mx500);
    migraphx::op::convolution convolution520;
    convolution520.padding  = {0, 0};
    convolution520.stride   = {1, 1};
    convolution520.dilation = {1, 1};
    convolution520.group    = 1;
    auto mx520              = p.add_instruction(convolution520, mx519, mx428);
    migraphx::op::batch_norm_inference batch_norm_inference521;
    batch_norm_inference521.epsilon  = 0.001;
    batch_norm_inference521.momentum = 0.9;
    auto mx521 = p.add_instruction(batch_norm_inference521, mx520, mx427, mx426, mx425, mx424);
    migraphx::op::relu relu522;
    auto mx522 = p.add_instruction(relu522, mx521);
    migraphx::op::concat concat523;
    concat523.axis = 1;
    auto mx523     = p.add_instruction(concat523, mx503, mx509, mx518, mx522);
    migraphx::op::convolution convolution524;
    convolution524.padding  = {0, 0};
    convolution524.stride   = {1, 1};
    convolution524.dilation = {1, 1};
    convolution524.group    = 1;
    auto mx524              = p.add_instruction(convolution524, mx523, mx423);
    migraphx::op::batch_norm_inference batch_norm_inference525;
    batch_norm_inference525.epsilon  = 0.001;
    batch_norm_inference525.momentum = 0.9;
    auto mx525 = p.add_instruction(batch_norm_inference525, mx524, mx422, mx421, mx420, mx419);
    migraphx::op::relu relu526;
    auto mx526 = p.add_instruction(relu526, mx525);
    migraphx::op::convolution convolution527;
    convolution527.padding  = {0, 0};
    convolution527.stride   = {1, 1};
    convolution527.dilation = {1, 1};
    convolution527.group    = 1;
    auto mx527              = p.add_instruction(convolution527, mx523, mx418);
    migraphx::op::batch_norm_inference batch_norm_inference528;
    batch_norm_inference528.epsilon  = 0.001;
    batch_norm_inference528.momentum = 0.9;
    auto mx528 = p.add_instruction(batch_norm_inference528, mx527, mx417, mx416, mx415, mx414);
    migraphx::op::relu relu529;
    auto mx529 = p.add_instruction(relu529, mx528);
    migraphx::op::convolution convolution530;
    convolution530.padding  = {2, 2};
    convolution530.stride   = {1, 1};
    convolution530.dilation = {1, 1};
    convolution530.group    = 1;
    auto mx530              = p.add_instruction(convolution530, mx529, mx413);
    migraphx::op::batch_norm_inference batch_norm_inference531;
    batch_norm_inference531.epsilon  = 0.001;
    batch_norm_inference531.momentum = 0.9;
    auto mx531 = p.add_instruction(batch_norm_inference531, mx530, mx412, mx411, mx410, mx409);
    migraphx::op::relu relu532;
    auto mx532 = p.add_instruction(relu532, mx531);
    migraphx::op::convolution convolution533;
    convolution533.padding  = {0, 0};
    convolution533.stride   = {1, 1};
    convolution533.dilation = {1, 1};
    convolution533.group    = 1;
    auto mx533              = p.add_instruction(convolution533, mx523, mx408);
    migraphx::op::batch_norm_inference batch_norm_inference534;
    batch_norm_inference534.epsilon  = 0.001;
    batch_norm_inference534.momentum = 0.9;
    auto mx534 = p.add_instruction(batch_norm_inference534, mx533, mx407, mx406, mx405, mx404);
    migraphx::op::relu relu535;
    auto mx535 = p.add_instruction(relu535, mx534);
    migraphx::op::convolution convolution536;
    convolution536.padding  = {1, 1};
    convolution536.stride   = {1, 1};
    convolution536.dilation = {1, 1};
    convolution536.group    = 1;
    auto mx536              = p.add_instruction(convolution536, mx535, mx403);
    migraphx::op::batch_norm_inference batch_norm_inference537;
    batch_norm_inference537.epsilon  = 0.001;
    batch_norm_inference537.momentum = 0.9;
    auto mx537 = p.add_instruction(batch_norm_inference537, mx536, mx402, mx401, mx400, mx399);
    migraphx::op::relu relu538;
    auto mx538 = p.add_instruction(relu538, mx537);
    migraphx::op::convolution convolution539;
    convolution539.padding  = {1, 1};
    convolution539.stride   = {1, 1};
    convolution539.dilation = {1, 1};
    convolution539.group    = 1;
    auto mx539              = p.add_instruction(convolution539, mx538, mx398);
    migraphx::op::batch_norm_inference batch_norm_inference540;
    batch_norm_inference540.epsilon  = 0.001;
    batch_norm_inference540.momentum = 0.9;
    auto mx540 = p.add_instruction(batch_norm_inference540, mx539, mx397, mx396, mx395, mx394);
    migraphx::op::relu relu541;
    auto mx541 = p.add_instruction(relu541, mx540);
    migraphx::op::pooling pooling542;
    pooling542.mode    = "average ";
    pooling542.padding = {1, 1};
    pooling542.stride  = {1, 1};
    pooling542.lengths = {3, 3};
    auto mx542         = p.add_instruction(pooling542, mx523);
    migraphx::op::convolution convolution543;
    convolution543.padding  = {0, 0};
    convolution543.stride   = {1, 1};
    convolution543.dilation = {1, 1};
    convolution543.group    = 1;
    auto mx543              = p.add_instruction(convolution543, mx542, mx393);
    migraphx::op::batch_norm_inference batch_norm_inference544;
    batch_norm_inference544.epsilon  = 0.001;
    batch_norm_inference544.momentum = 0.9;
    auto mx544 = p.add_instruction(batch_norm_inference544, mx543, mx392, mx391, mx390, mx389);
    migraphx::op::relu relu545;
    auto mx545 = p.add_instruction(relu545, mx544);
    migraphx::op::concat concat546;
    concat546.axis = 1;
    auto mx546     = p.add_instruction(concat546, mx526, mx532, mx541, mx545);
    migraphx::op::convolution convolution547;
    convolution547.padding  = {0, 0};
    convolution547.stride   = {1, 1};
    convolution547.dilation = {1, 1};
    convolution547.group    = 1;
    auto mx547              = p.add_instruction(convolution547, mx546, mx388);
    migraphx::op::batch_norm_inference batch_norm_inference548;
    batch_norm_inference548.epsilon  = 0.001;
    batch_norm_inference548.momentum = 0.9;
    auto mx548 = p.add_instruction(batch_norm_inference548, mx547, mx387, mx386, mx385, mx384);
    migraphx::op::relu relu549;
    auto mx549 = p.add_instruction(relu549, mx548);
    migraphx::op::convolution convolution550;
    convolution550.padding  = {0, 0};
    convolution550.stride   = {1, 1};
    convolution550.dilation = {1, 1};
    convolution550.group    = 1;
    auto mx550              = p.add_instruction(convolution550, mx546, mx383);
    migraphx::op::batch_norm_inference batch_norm_inference551;
    batch_norm_inference551.epsilon  = 0.001;
    batch_norm_inference551.momentum = 0.9;
    auto mx551 = p.add_instruction(batch_norm_inference551, mx550, mx382, mx381, mx380, mx379);
    migraphx::op::relu relu552;
    auto mx552 = p.add_instruction(relu552, mx551);
    migraphx::op::convolution convolution553;
    convolution553.padding  = {2, 2};
    convolution553.stride   = {1, 1};
    convolution553.dilation = {1, 1};
    convolution553.group    = 1;
    auto mx553              = p.add_instruction(convolution553, mx552, mx378);
    migraphx::op::batch_norm_inference batch_norm_inference554;
    batch_norm_inference554.epsilon  = 0.001;
    batch_norm_inference554.momentum = 0.9;
    auto mx554 = p.add_instruction(batch_norm_inference554, mx553, mx377, mx376, mx375, mx374);
    migraphx::op::relu relu555;
    auto mx555 = p.add_instruction(relu555, mx554);
    migraphx::op::convolution convolution556;
    convolution556.padding  = {0, 0};
    convolution556.stride   = {1, 1};
    convolution556.dilation = {1, 1};
    convolution556.group    = 1;
    auto mx556              = p.add_instruction(convolution556, mx546, mx373);
    migraphx::op::batch_norm_inference batch_norm_inference557;
    batch_norm_inference557.epsilon  = 0.001;
    batch_norm_inference557.momentum = 0.9;
    auto mx557 = p.add_instruction(batch_norm_inference557, mx556, mx372, mx371, mx370, mx369);
    migraphx::op::relu relu558;
    auto mx558 = p.add_instruction(relu558, mx557);
    migraphx::op::convolution convolution559;
    convolution559.padding  = {1, 1};
    convolution559.stride   = {1, 1};
    convolution559.dilation = {1, 1};
    convolution559.group    = 1;
    auto mx559              = p.add_instruction(convolution559, mx558, mx368);
    migraphx::op::batch_norm_inference batch_norm_inference560;
    batch_norm_inference560.epsilon  = 0.001;
    batch_norm_inference560.momentum = 0.9;
    auto mx560 = p.add_instruction(batch_norm_inference560, mx559, mx367, mx366, mx365, mx364);
    migraphx::op::relu relu561;
    auto mx561 = p.add_instruction(relu561, mx560);
    migraphx::op::convolution convolution562;
    convolution562.padding  = {1, 1};
    convolution562.stride   = {1, 1};
    convolution562.dilation = {1, 1};
    convolution562.group    = 1;
    auto mx562              = p.add_instruction(convolution562, mx561, mx363);
    migraphx::op::batch_norm_inference batch_norm_inference563;
    batch_norm_inference563.epsilon  = 0.001;
    batch_norm_inference563.momentum = 0.9;
    auto mx563 = p.add_instruction(batch_norm_inference563, mx562, mx362, mx361, mx360, mx359);
    migraphx::op::relu relu564;
    auto mx564 = p.add_instruction(relu564, mx563);
    migraphx::op::pooling pooling565;
    pooling565.mode    = "average ";
    pooling565.padding = {1, 1};
    pooling565.stride  = {1, 1};
    pooling565.lengths = {3, 3};
    auto mx565         = p.add_instruction(pooling565, mx546);
    migraphx::op::convolution convolution566;
    convolution566.padding  = {0, 0};
    convolution566.stride   = {1, 1};
    convolution566.dilation = {1, 1};
    convolution566.group    = 1;
    auto mx566              = p.add_instruction(convolution566, mx565, mx358);
    migraphx::op::batch_norm_inference batch_norm_inference567;
    batch_norm_inference567.epsilon  = 0.001;
    batch_norm_inference567.momentum = 0.9;
    auto mx567 = p.add_instruction(batch_norm_inference567, mx566, mx357, mx356, mx355, mx354);
    migraphx::op::relu relu568;
    auto mx568 = p.add_instruction(relu568, mx567);
    migraphx::op::concat concat569;
    concat569.axis = 1;
    auto mx569     = p.add_instruction(concat569, mx549, mx555, mx564, mx568);
    migraphx::op::convolution convolution570;
    convolution570.padding  = {0, 0};
    convolution570.stride   = {2, 2};
    convolution570.dilation = {1, 1};
    convolution570.group    = 1;
    auto mx570              = p.add_instruction(convolution570, mx569, mx353);
    migraphx::op::batch_norm_inference batch_norm_inference571;
    batch_norm_inference571.epsilon  = 0.001;
    batch_norm_inference571.momentum = 0.9;
    auto mx571 = p.add_instruction(batch_norm_inference571, mx570, mx352, mx351, mx350, mx349);
    migraphx::op::relu relu572;
    auto mx572 = p.add_instruction(relu572, mx571);
    migraphx::op::convolution convolution573;
    convolution573.padding  = {0, 0};
    convolution573.stride   = {1, 1};
    convolution573.dilation = {1, 1};
    convolution573.group    = 1;
    auto mx573              = p.add_instruction(convolution573, mx569, mx348);
    migraphx::op::batch_norm_inference batch_norm_inference574;
    batch_norm_inference574.epsilon  = 0.001;
    batch_norm_inference574.momentum = 0.9;
    auto mx574 = p.add_instruction(batch_norm_inference574, mx573, mx347, mx346, mx345, mx344);
    migraphx::op::relu relu575;
    auto mx575 = p.add_instruction(relu575, mx574);
    migraphx::op::convolution convolution576;
    convolution576.padding  = {1, 1};
    convolution576.stride   = {1, 1};
    convolution576.dilation = {1, 1};
    convolution576.group    = 1;
    auto mx576              = p.add_instruction(convolution576, mx575, mx343);
    migraphx::op::batch_norm_inference batch_norm_inference577;
    batch_norm_inference577.epsilon  = 0.001;
    batch_norm_inference577.momentum = 0.9;
    auto mx577 = p.add_instruction(batch_norm_inference577, mx576, mx342, mx341, mx340, mx339);
    migraphx::op::relu relu578;
    auto mx578 = p.add_instruction(relu578, mx577);
    migraphx::op::convolution convolution579;
    convolution579.padding  = {0, 0};
    convolution579.stride   = {2, 2};
    convolution579.dilation = {1, 1};
    convolution579.group    = 1;
    auto mx579              = p.add_instruction(convolution579, mx578, mx338);
    migraphx::op::batch_norm_inference batch_norm_inference580;
    batch_norm_inference580.epsilon  = 0.001;
    batch_norm_inference580.momentum = 0.9;
    auto mx580 = p.add_instruction(batch_norm_inference580, mx579, mx337, mx336, mx335, mx334);
    migraphx::op::relu relu581;
    auto mx581 = p.add_instruction(relu581, mx580);
    migraphx::op::pooling pooling582;
    pooling582.mode    = "max ";
    pooling582.padding = {0, 0};
    pooling582.stride  = {2, 2};
    pooling582.lengths = {3, 3};
    auto mx582         = p.add_instruction(pooling582, mx569);
    migraphx::op::concat concat583;
    concat583.axis = 1;
    auto mx583     = p.add_instruction(concat583, mx572, mx581, mx582);
    migraphx::op::convolution convolution584;
    convolution584.padding  = {0, 0};
    convolution584.stride   = {1, 1};
    convolution584.dilation = {1, 1};
    convolution584.group    = 1;
    auto mx584              = p.add_instruction(convolution584, mx583, mx333);
    migraphx::op::batch_norm_inference batch_norm_inference585;
    batch_norm_inference585.epsilon  = 0.001;
    batch_norm_inference585.momentum = 0.9;
    auto mx585 = p.add_instruction(batch_norm_inference585, mx584, mx332, mx331, mx330, mx329);
    migraphx::op::relu relu586;
    auto mx586 = p.add_instruction(relu586, mx585);
    migraphx::op::convolution convolution587;
    convolution587.padding  = {0, 0};
    convolution587.stride   = {1, 1};
    convolution587.dilation = {1, 1};
    convolution587.group    = 1;
    auto mx587              = p.add_instruction(convolution587, mx583, mx328);
    migraphx::op::batch_norm_inference batch_norm_inference588;
    batch_norm_inference588.epsilon  = 0.001;
    batch_norm_inference588.momentum = 0.9;
    auto mx588 = p.add_instruction(batch_norm_inference588, mx587, mx327, mx326, mx325, mx324);
    migraphx::op::relu relu589;
    auto mx589 = p.add_instruction(relu589, mx588);
    migraphx::op::convolution convolution590;
    convolution590.padding  = {0, 3};
    convolution590.stride   = {1, 1};
    convolution590.dilation = {1, 1};
    convolution590.group    = 1;
    auto mx590              = p.add_instruction(convolution590, mx589, mx323);
    migraphx::op::batch_norm_inference batch_norm_inference591;
    batch_norm_inference591.epsilon  = 0.001;
    batch_norm_inference591.momentum = 0.9;
    auto mx591 = p.add_instruction(batch_norm_inference591, mx590, mx322, mx321, mx320, mx319);
    migraphx::op::relu relu592;
    auto mx592 = p.add_instruction(relu592, mx591);
    migraphx::op::convolution convolution593;
    convolution593.padding  = {3, 0};
    convolution593.stride   = {1, 1};
    convolution593.dilation = {1, 1};
    convolution593.group    = 1;
    auto mx593              = p.add_instruction(convolution593, mx592, mx318);
    migraphx::op::batch_norm_inference batch_norm_inference594;
    batch_norm_inference594.epsilon  = 0.001;
    batch_norm_inference594.momentum = 0.9;
    auto mx594 = p.add_instruction(batch_norm_inference594, mx593, mx317, mx316, mx315, mx314);
    migraphx::op::relu relu595;
    auto mx595 = p.add_instruction(relu595, mx594);
    migraphx::op::convolution convolution596;
    convolution596.padding  = {0, 0};
    convolution596.stride   = {1, 1};
    convolution596.dilation = {1, 1};
    convolution596.group    = 1;
    auto mx596              = p.add_instruction(convolution596, mx583, mx313);
    migraphx::op::batch_norm_inference batch_norm_inference597;
    batch_norm_inference597.epsilon  = 0.001;
    batch_norm_inference597.momentum = 0.9;
    auto mx597 = p.add_instruction(batch_norm_inference597, mx596, mx312, mx311, mx310, mx309);
    migraphx::op::relu relu598;
    auto mx598 = p.add_instruction(relu598, mx597);
    migraphx::op::convolution convolution599;
    convolution599.padding  = {3, 0};
    convolution599.stride   = {1, 1};
    convolution599.dilation = {1, 1};
    convolution599.group    = 1;
    auto mx599              = p.add_instruction(convolution599, mx598, mx308);
    migraphx::op::batch_norm_inference batch_norm_inference600;
    batch_norm_inference600.epsilon  = 0.001;
    batch_norm_inference600.momentum = 0.9;
    auto mx600 = p.add_instruction(batch_norm_inference600, mx599, mx307, mx306, mx305, mx304);
    migraphx::op::relu relu601;
    auto mx601 = p.add_instruction(relu601, mx600);
    migraphx::op::convolution convolution602;
    convolution602.padding  = {0, 3};
    convolution602.stride   = {1, 1};
    convolution602.dilation = {1, 1};
    convolution602.group    = 1;
    auto mx602              = p.add_instruction(convolution602, mx601, mx303);
    migraphx::op::batch_norm_inference batch_norm_inference603;
    batch_norm_inference603.epsilon  = 0.001;
    batch_norm_inference603.momentum = 0.9;
    auto mx603 = p.add_instruction(batch_norm_inference603, mx602, mx302, mx301, mx300, mx299);
    migraphx::op::relu relu604;
    auto mx604 = p.add_instruction(relu604, mx603);
    migraphx::op::convolution convolution605;
    convolution605.padding  = {3, 0};
    convolution605.stride   = {1, 1};
    convolution605.dilation = {1, 1};
    convolution605.group    = 1;
    auto mx605              = p.add_instruction(convolution605, mx604, mx298);
    migraphx::op::batch_norm_inference batch_norm_inference606;
    batch_norm_inference606.epsilon  = 0.001;
    batch_norm_inference606.momentum = 0.9;
    auto mx606 = p.add_instruction(batch_norm_inference606, mx605, mx297, mx296, mx295, mx294);
    migraphx::op::relu relu607;
    auto mx607 = p.add_instruction(relu607, mx606);
    migraphx::op::convolution convolution608;
    convolution608.padding  = {0, 3};
    convolution608.stride   = {1, 1};
    convolution608.dilation = {1, 1};
    convolution608.group    = 1;
    auto mx608              = p.add_instruction(convolution608, mx607, mx293);
    migraphx::op::batch_norm_inference batch_norm_inference609;
    batch_norm_inference609.epsilon  = 0.001;
    batch_norm_inference609.momentum = 0.9;
    auto mx609 = p.add_instruction(batch_norm_inference609, mx608, mx292, mx291, mx290, mx289);
    migraphx::op::relu relu610;
    auto mx610 = p.add_instruction(relu610, mx609);
    migraphx::op::pooling pooling611;
    pooling611.mode    = "average ";
    pooling611.padding = {1, 1};
    pooling611.stride  = {1, 1};
    pooling611.lengths = {3, 3};
    auto mx611         = p.add_instruction(pooling611, mx583);
    migraphx::op::convolution convolution612;
    convolution612.padding  = {0, 0};
    convolution612.stride   = {1, 1};
    convolution612.dilation = {1, 1};
    convolution612.group    = 1;
    auto mx612              = p.add_instruction(convolution612, mx611, mx288);
    migraphx::op::batch_norm_inference batch_norm_inference613;
    batch_norm_inference613.epsilon  = 0.001;
    batch_norm_inference613.momentum = 0.9;
    auto mx613 = p.add_instruction(batch_norm_inference613, mx612, mx287, mx286, mx285, mx284);
    migraphx::op::relu relu614;
    auto mx614 = p.add_instruction(relu614, mx613);
    migraphx::op::concat concat615;
    concat615.axis = 1;
    auto mx615     = p.add_instruction(concat615, mx586, mx595, mx610, mx614);
    migraphx::op::convolution convolution616;
    convolution616.padding  = {0, 0};
    convolution616.stride   = {1, 1};
    convolution616.dilation = {1, 1};
    convolution616.group    = 1;
    auto mx616              = p.add_instruction(convolution616, mx615, mx283);
    migraphx::op::batch_norm_inference batch_norm_inference617;
    batch_norm_inference617.epsilon  = 0.001;
    batch_norm_inference617.momentum = 0.9;
    auto mx617 = p.add_instruction(batch_norm_inference617, mx616, mx282, mx281, mx280, mx279);
    migraphx::op::relu relu618;
    auto mx618 = p.add_instruction(relu618, mx617);
    migraphx::op::convolution convolution619;
    convolution619.padding  = {0, 0};
    convolution619.stride   = {1, 1};
    convolution619.dilation = {1, 1};
    convolution619.group    = 1;
    auto mx619              = p.add_instruction(convolution619, mx615, mx278);
    migraphx::op::batch_norm_inference batch_norm_inference620;
    batch_norm_inference620.epsilon  = 0.001;
    batch_norm_inference620.momentum = 0.9;
    auto mx620 = p.add_instruction(batch_norm_inference620, mx619, mx277, mx276, mx275, mx274);
    migraphx::op::relu relu621;
    auto mx621 = p.add_instruction(relu621, mx620);
    migraphx::op::convolution convolution622;
    convolution622.padding  = {0, 3};
    convolution622.stride   = {1, 1};
    convolution622.dilation = {1, 1};
    convolution622.group    = 1;
    auto mx622              = p.add_instruction(convolution622, mx621, mx273);
    migraphx::op::batch_norm_inference batch_norm_inference623;
    batch_norm_inference623.epsilon  = 0.001;
    batch_norm_inference623.momentum = 0.9;
    auto mx623 = p.add_instruction(batch_norm_inference623, mx622, mx272, mx271, mx270, mx269);
    migraphx::op::relu relu624;
    auto mx624 = p.add_instruction(relu624, mx623);
    migraphx::op::convolution convolution625;
    convolution625.padding  = {3, 0};
    convolution625.stride   = {1, 1};
    convolution625.dilation = {1, 1};
    convolution625.group    = 1;
    auto mx625              = p.add_instruction(convolution625, mx624, mx268);
    migraphx::op::batch_norm_inference batch_norm_inference626;
    batch_norm_inference626.epsilon  = 0.001;
    batch_norm_inference626.momentum = 0.9;
    auto mx626 = p.add_instruction(batch_norm_inference626, mx625, mx267, mx266, mx265, mx264);
    migraphx::op::relu relu627;
    auto mx627 = p.add_instruction(relu627, mx626);
    migraphx::op::convolution convolution628;
    convolution628.padding  = {0, 0};
    convolution628.stride   = {1, 1};
    convolution628.dilation = {1, 1};
    convolution628.group    = 1;
    auto mx628              = p.add_instruction(convolution628, mx615, mx263);
    migraphx::op::batch_norm_inference batch_norm_inference629;
    batch_norm_inference629.epsilon  = 0.001;
    batch_norm_inference629.momentum = 0.9;
    auto mx629 = p.add_instruction(batch_norm_inference629, mx628, mx262, mx261, mx260, mx259);
    migraphx::op::relu relu630;
    auto mx630 = p.add_instruction(relu630, mx629);
    migraphx::op::convolution convolution631;
    convolution631.padding  = {3, 0};
    convolution631.stride   = {1, 1};
    convolution631.dilation = {1, 1};
    convolution631.group    = 1;
    auto mx631              = p.add_instruction(convolution631, mx630, mx258);
    migraphx::op::batch_norm_inference batch_norm_inference632;
    batch_norm_inference632.epsilon  = 0.001;
    batch_norm_inference632.momentum = 0.9;
    auto mx632 = p.add_instruction(batch_norm_inference632, mx631, mx257, mx256, mx255, mx254);
    migraphx::op::relu relu633;
    auto mx633 = p.add_instruction(relu633, mx632);
    migraphx::op::convolution convolution634;
    convolution634.padding  = {0, 3};
    convolution634.stride   = {1, 1};
    convolution634.dilation = {1, 1};
    convolution634.group    = 1;
    auto mx634              = p.add_instruction(convolution634, mx633, mx253);
    migraphx::op::batch_norm_inference batch_norm_inference635;
    batch_norm_inference635.epsilon  = 0.001;
    batch_norm_inference635.momentum = 0.9;
    auto mx635 = p.add_instruction(batch_norm_inference635, mx634, mx252, mx251, mx250, mx249);
    migraphx::op::relu relu636;
    auto mx636 = p.add_instruction(relu636, mx635);
    migraphx::op::convolution convolution637;
    convolution637.padding  = {3, 0};
    convolution637.stride   = {1, 1};
    convolution637.dilation = {1, 1};
    convolution637.group    = 1;
    auto mx637              = p.add_instruction(convolution637, mx636, mx248);
    migraphx::op::batch_norm_inference batch_norm_inference638;
    batch_norm_inference638.epsilon  = 0.001;
    batch_norm_inference638.momentum = 0.9;
    auto mx638 = p.add_instruction(batch_norm_inference638, mx637, mx247, mx246, mx245, mx244);
    migraphx::op::relu relu639;
    auto mx639 = p.add_instruction(relu639, mx638);
    migraphx::op::convolution convolution640;
    convolution640.padding  = {0, 3};
    convolution640.stride   = {1, 1};
    convolution640.dilation = {1, 1};
    convolution640.group    = 1;
    auto mx640              = p.add_instruction(convolution640, mx639, mx243);
    migraphx::op::batch_norm_inference batch_norm_inference641;
    batch_norm_inference641.epsilon  = 0.001;
    batch_norm_inference641.momentum = 0.9;
    auto mx641 = p.add_instruction(batch_norm_inference641, mx640, mx242, mx241, mx240, mx239);
    migraphx::op::relu relu642;
    auto mx642 = p.add_instruction(relu642, mx641);
    migraphx::op::pooling pooling643;
    pooling643.mode    = "average ";
    pooling643.padding = {1, 1};
    pooling643.stride  = {1, 1};
    pooling643.lengths = {3, 3};
    auto mx643         = p.add_instruction(pooling643, mx615);
    migraphx::op::convolution convolution644;
    convolution644.padding  = {0, 0};
    convolution644.stride   = {1, 1};
    convolution644.dilation = {1, 1};
    convolution644.group    = 1;
    auto mx644              = p.add_instruction(convolution644, mx643, mx238);
    migraphx::op::batch_norm_inference batch_norm_inference645;
    batch_norm_inference645.epsilon  = 0.001;
    batch_norm_inference645.momentum = 0.9;
    auto mx645 = p.add_instruction(batch_norm_inference645, mx644, mx237, mx236, mx235, mx234);
    migraphx::op::relu relu646;
    auto mx646 = p.add_instruction(relu646, mx645);
    migraphx::op::concat concat647;
    concat647.axis = 1;
    auto mx647     = p.add_instruction(concat647, mx618, mx627, mx642, mx646);
    migraphx::op::convolution convolution648;
    convolution648.padding  = {0, 0};
    convolution648.stride   = {1, 1};
    convolution648.dilation = {1, 1};
    convolution648.group    = 1;
    auto mx648              = p.add_instruction(convolution648, mx647, mx233);
    migraphx::op::batch_norm_inference batch_norm_inference649;
    batch_norm_inference649.epsilon  = 0.001;
    batch_norm_inference649.momentum = 0.9;
    auto mx649 = p.add_instruction(batch_norm_inference649, mx648, mx232, mx231, mx230, mx229);
    migraphx::op::relu relu650;
    auto mx650 = p.add_instruction(relu650, mx649);
    migraphx::op::convolution convolution651;
    convolution651.padding  = {0, 0};
    convolution651.stride   = {1, 1};
    convolution651.dilation = {1, 1};
    convolution651.group    = 1;
    auto mx651              = p.add_instruction(convolution651, mx647, mx228);
    migraphx::op::batch_norm_inference batch_norm_inference652;
    batch_norm_inference652.epsilon  = 0.001;
    batch_norm_inference652.momentum = 0.9;
    auto mx652 = p.add_instruction(batch_norm_inference652, mx651, mx227, mx226, mx225, mx224);
    migraphx::op::relu relu653;
    auto mx653 = p.add_instruction(relu653, mx652);
    migraphx::op::convolution convolution654;
    convolution654.padding  = {0, 3};
    convolution654.stride   = {1, 1};
    convolution654.dilation = {1, 1};
    convolution654.group    = 1;
    auto mx654              = p.add_instruction(convolution654, mx653, mx223);
    migraphx::op::batch_norm_inference batch_norm_inference655;
    batch_norm_inference655.epsilon  = 0.001;
    batch_norm_inference655.momentum = 0.9;
    auto mx655 = p.add_instruction(batch_norm_inference655, mx654, mx222, mx221, mx220, mx219);
    migraphx::op::relu relu656;
    auto mx656 = p.add_instruction(relu656, mx655);
    migraphx::op::convolution convolution657;
    convolution657.padding  = {3, 0};
    convolution657.stride   = {1, 1};
    convolution657.dilation = {1, 1};
    convolution657.group    = 1;
    auto mx657              = p.add_instruction(convolution657, mx656, mx218);
    migraphx::op::batch_norm_inference batch_norm_inference658;
    batch_norm_inference658.epsilon  = 0.001;
    batch_norm_inference658.momentum = 0.9;
    auto mx658 = p.add_instruction(batch_norm_inference658, mx657, mx217, mx216, mx215, mx214);
    migraphx::op::relu relu659;
    auto mx659 = p.add_instruction(relu659, mx658);
    migraphx::op::convolution convolution660;
    convolution660.padding  = {0, 0};
    convolution660.stride   = {1, 1};
    convolution660.dilation = {1, 1};
    convolution660.group    = 1;
    auto mx660              = p.add_instruction(convolution660, mx647, mx213);
    migraphx::op::batch_norm_inference batch_norm_inference661;
    batch_norm_inference661.epsilon  = 0.001;
    batch_norm_inference661.momentum = 0.9;
    auto mx661 = p.add_instruction(batch_norm_inference661, mx660, mx212, mx211, mx210, mx209);
    migraphx::op::relu relu662;
    auto mx662 = p.add_instruction(relu662, mx661);
    migraphx::op::convolution convolution663;
    convolution663.padding  = {3, 0};
    convolution663.stride   = {1, 1};
    convolution663.dilation = {1, 1};
    convolution663.group    = 1;
    auto mx663              = p.add_instruction(convolution663, mx662, mx208);
    migraphx::op::batch_norm_inference batch_norm_inference664;
    batch_norm_inference664.epsilon  = 0.001;
    batch_norm_inference664.momentum = 0.9;
    auto mx664 = p.add_instruction(batch_norm_inference664, mx663, mx207, mx206, mx205, mx204);
    migraphx::op::relu relu665;
    auto mx665 = p.add_instruction(relu665, mx664);
    migraphx::op::convolution convolution666;
    convolution666.padding  = {0, 3};
    convolution666.stride   = {1, 1};
    convolution666.dilation = {1, 1};
    convolution666.group    = 1;
    auto mx666              = p.add_instruction(convolution666, mx665, mx203);
    migraphx::op::batch_norm_inference batch_norm_inference667;
    batch_norm_inference667.epsilon  = 0.001;
    batch_norm_inference667.momentum = 0.9;
    auto mx667 = p.add_instruction(batch_norm_inference667, mx666, mx202, mx201, mx200, mx199);
    migraphx::op::relu relu668;
    auto mx668 = p.add_instruction(relu668, mx667);
    migraphx::op::convolution convolution669;
    convolution669.padding  = {3, 0};
    convolution669.stride   = {1, 1};
    convolution669.dilation = {1, 1};
    convolution669.group    = 1;
    auto mx669              = p.add_instruction(convolution669, mx668, mx198);
    migraphx::op::batch_norm_inference batch_norm_inference670;
    batch_norm_inference670.epsilon  = 0.001;
    batch_norm_inference670.momentum = 0.9;
    auto mx670 = p.add_instruction(batch_norm_inference670, mx669, mx197, mx196, mx195, mx194);
    migraphx::op::relu relu671;
    auto mx671 = p.add_instruction(relu671, mx670);
    migraphx::op::convolution convolution672;
    convolution672.padding  = {0, 3};
    convolution672.stride   = {1, 1};
    convolution672.dilation = {1, 1};
    convolution672.group    = 1;
    auto mx672              = p.add_instruction(convolution672, mx671, mx193);
    migraphx::op::batch_norm_inference batch_norm_inference673;
    batch_norm_inference673.epsilon  = 0.001;
    batch_norm_inference673.momentum = 0.9;
    auto mx673 = p.add_instruction(batch_norm_inference673, mx672, mx192, mx191, mx190, mx189);
    migraphx::op::relu relu674;
    auto mx674 = p.add_instruction(relu674, mx673);
    migraphx::op::pooling pooling675;
    pooling675.mode    = "average ";
    pooling675.padding = {1, 1};
    pooling675.stride  = {1, 1};
    pooling675.lengths = {3, 3};
    auto mx675         = p.add_instruction(pooling675, mx647);
    migraphx::op::convolution convolution676;
    convolution676.padding  = {0, 0};
    convolution676.stride   = {1, 1};
    convolution676.dilation = {1, 1};
    convolution676.group    = 1;
    auto mx676              = p.add_instruction(convolution676, mx675, mx188);
    migraphx::op::batch_norm_inference batch_norm_inference677;
    batch_norm_inference677.epsilon  = 0.001;
    batch_norm_inference677.momentum = 0.9;
    auto mx677 = p.add_instruction(batch_norm_inference677, mx676, mx187, mx186, mx185, mx184);
    migraphx::op::relu relu678;
    auto mx678 = p.add_instruction(relu678, mx677);
    migraphx::op::concat concat679;
    concat679.axis = 1;
    auto mx679     = p.add_instruction(concat679, mx650, mx659, mx674, mx678);
    migraphx::op::convolution convolution680;
    convolution680.padding  = {0, 0};
    convolution680.stride   = {1, 1};
    convolution680.dilation = {1, 1};
    convolution680.group    = 1;
    auto mx680              = p.add_instruction(convolution680, mx679, mx183);
    migraphx::op::batch_norm_inference batch_norm_inference681;
    batch_norm_inference681.epsilon  = 0.001;
    batch_norm_inference681.momentum = 0.9;
    auto mx681 = p.add_instruction(batch_norm_inference681, mx680, mx182, mx181, mx180, mx179);
    migraphx::op::relu relu682;
    auto mx682 = p.add_instruction(relu682, mx681);
    migraphx::op::convolution convolution683;
    convolution683.padding  = {0, 0};
    convolution683.stride   = {1, 1};
    convolution683.dilation = {1, 1};
    convolution683.group    = 1;
    auto mx683              = p.add_instruction(convolution683, mx679, mx178);
    migraphx::op::batch_norm_inference batch_norm_inference684;
    batch_norm_inference684.epsilon  = 0.001;
    batch_norm_inference684.momentum = 0.9;
    auto mx684 = p.add_instruction(batch_norm_inference684, mx683, mx177, mx176, mx175, mx174);
    migraphx::op::relu relu685;
    auto mx685 = p.add_instruction(relu685, mx684);
    migraphx::op::convolution convolution686;
    convolution686.padding  = {0, 3};
    convolution686.stride   = {1, 1};
    convolution686.dilation = {1, 1};
    convolution686.group    = 1;
    auto mx686              = p.add_instruction(convolution686, mx685, mx173);
    migraphx::op::batch_norm_inference batch_norm_inference687;
    batch_norm_inference687.epsilon  = 0.001;
    batch_norm_inference687.momentum = 0.9;
    auto mx687 = p.add_instruction(batch_norm_inference687, mx686, mx172, mx171, mx170, mx169);
    migraphx::op::relu relu688;
    auto mx688 = p.add_instruction(relu688, mx687);
    migraphx::op::convolution convolution689;
    convolution689.padding  = {3, 0};
    convolution689.stride   = {1, 1};
    convolution689.dilation = {1, 1};
    convolution689.group    = 1;
    auto mx689              = p.add_instruction(convolution689, mx688, mx168);
    migraphx::op::batch_norm_inference batch_norm_inference690;
    batch_norm_inference690.epsilon  = 0.001;
    batch_norm_inference690.momentum = 0.9;
    auto mx690 = p.add_instruction(batch_norm_inference690, mx689, mx167, mx166, mx165, mx164);
    migraphx::op::relu relu691;
    auto mx691 = p.add_instruction(relu691, mx690);
    migraphx::op::convolution convolution692;
    convolution692.padding  = {0, 0};
    convolution692.stride   = {1, 1};
    convolution692.dilation = {1, 1};
    convolution692.group    = 1;
    auto mx692              = p.add_instruction(convolution692, mx679, mx163);
    migraphx::op::batch_norm_inference batch_norm_inference693;
    batch_norm_inference693.epsilon  = 0.001;
    batch_norm_inference693.momentum = 0.9;
    auto mx693 = p.add_instruction(batch_norm_inference693, mx692, mx162, mx161, mx160, mx159);
    migraphx::op::relu relu694;
    auto mx694 = p.add_instruction(relu694, mx693);
    migraphx::op::convolution convolution695;
    convolution695.padding  = {3, 0};
    convolution695.stride   = {1, 1};
    convolution695.dilation = {1, 1};
    convolution695.group    = 1;
    auto mx695              = p.add_instruction(convolution695, mx694, mx158);
    migraphx::op::batch_norm_inference batch_norm_inference696;
    batch_norm_inference696.epsilon  = 0.001;
    batch_norm_inference696.momentum = 0.9;
    auto mx696 = p.add_instruction(batch_norm_inference696, mx695, mx157, mx156, mx155, mx154);
    migraphx::op::relu relu697;
    auto mx697 = p.add_instruction(relu697, mx696);
    migraphx::op::convolution convolution698;
    convolution698.padding  = {0, 3};
    convolution698.stride   = {1, 1};
    convolution698.dilation = {1, 1};
    convolution698.group    = 1;
    auto mx698              = p.add_instruction(convolution698, mx697, mx153);
    migraphx::op::batch_norm_inference batch_norm_inference699;
    batch_norm_inference699.epsilon  = 0.001;
    batch_norm_inference699.momentum = 0.9;
    auto mx699 = p.add_instruction(batch_norm_inference699, mx698, mx152, mx151, mx150, mx149);
    migraphx::op::relu relu700;
    auto mx700 = p.add_instruction(relu700, mx699);
    migraphx::op::convolution convolution701;
    convolution701.padding  = {3, 0};
    convolution701.stride   = {1, 1};
    convolution701.dilation = {1, 1};
    convolution701.group    = 1;
    auto mx701              = p.add_instruction(convolution701, mx700, mx148);
    migraphx::op::batch_norm_inference batch_norm_inference702;
    batch_norm_inference702.epsilon  = 0.001;
    batch_norm_inference702.momentum = 0.9;
    auto mx702 = p.add_instruction(batch_norm_inference702, mx701, mx147, mx146, mx145, mx144);
    migraphx::op::relu relu703;
    auto mx703 = p.add_instruction(relu703, mx702);
    migraphx::op::convolution convolution704;
    convolution704.padding  = {0, 3};
    convolution704.stride   = {1, 1};
    convolution704.dilation = {1, 1};
    convolution704.group    = 1;
    auto mx704              = p.add_instruction(convolution704, mx703, mx143);
    migraphx::op::batch_norm_inference batch_norm_inference705;
    batch_norm_inference705.epsilon  = 0.001;
    batch_norm_inference705.momentum = 0.9;
    auto mx705 = p.add_instruction(batch_norm_inference705, mx704, mx142, mx141, mx140, mx139);
    migraphx::op::relu relu706;
    auto mx706 = p.add_instruction(relu706, mx705);
    migraphx::op::pooling pooling707;
    pooling707.mode    = "average ";
    pooling707.padding = {1, 1};
    pooling707.stride  = {1, 1};
    pooling707.lengths = {3, 3};
    auto mx707         = p.add_instruction(pooling707, mx679);
    migraphx::op::convolution convolution708;
    convolution708.padding  = {0, 0};
    convolution708.stride   = {1, 1};
    convolution708.dilation = {1, 1};
    convolution708.group    = 1;
    auto mx708              = p.add_instruction(convolution708, mx707, mx138);
    migraphx::op::batch_norm_inference batch_norm_inference709;
    batch_norm_inference709.epsilon  = 0.001;
    batch_norm_inference709.momentum = 0.9;
    auto mx709 = p.add_instruction(batch_norm_inference709, mx708, mx137, mx136, mx135, mx134);
    migraphx::op::relu relu710;
    auto mx710 = p.add_instruction(relu710, mx709);
    migraphx::op::concat concat711;
    concat711.axis = 1;
    auto mx711     = p.add_instruction(concat711, mx682, mx691, mx706, mx710);
    migraphx::op::convolution convolution712;
    convolution712.padding  = {0, 0};
    convolution712.stride   = {1, 1};
    convolution712.dilation = {1, 1};
    convolution712.group    = 1;
    auto mx712              = p.add_instruction(convolution712, mx711, mx121);
    migraphx::op::batch_norm_inference batch_norm_inference713;
    batch_norm_inference713.epsilon  = 0.001;
    batch_norm_inference713.momentum = 0.9;
    auto mx713 = p.add_instruction(batch_norm_inference713, mx712, mx120, mx119, mx118, mx117);
    migraphx::op::relu relu714;
    auto mx714 = p.add_instruction(relu714, mx713);
    migraphx::op::convolution convolution715;
    convolution715.padding  = {0, 0};
    convolution715.stride   = {2, 2};
    convolution715.dilation = {1, 1};
    convolution715.group    = 1;
    auto mx715              = p.add_instruction(convolution715, mx714, mx116);
    migraphx::op::batch_norm_inference batch_norm_inference716;
    batch_norm_inference716.epsilon  = 0.001;
    batch_norm_inference716.momentum = 0.9;
    auto mx716 = p.add_instruction(batch_norm_inference716, mx715, mx115, mx114, mx113, mx112);
    migraphx::op::relu relu717;
    auto mx717 = p.add_instruction(relu717, mx716);
    migraphx::op::convolution convolution718;
    convolution718.padding  = {0, 0};
    convolution718.stride   = {1, 1};
    convolution718.dilation = {1, 1};
    convolution718.group    = 1;
    auto mx718              = p.add_instruction(convolution718, mx711, mx111);
    migraphx::op::batch_norm_inference batch_norm_inference719;
    batch_norm_inference719.epsilon  = 0.001;
    batch_norm_inference719.momentum = 0.9;
    auto mx719 = p.add_instruction(batch_norm_inference719, mx718, mx110, mx109, mx108, mx107);
    migraphx::op::relu relu720;
    auto mx720 = p.add_instruction(relu720, mx719);
    migraphx::op::convolution convolution721;
    convolution721.padding  = {0, 3};
    convolution721.stride   = {1, 1};
    convolution721.dilation = {1, 1};
    convolution721.group    = 1;
    auto mx721              = p.add_instruction(convolution721, mx720, mx106);
    migraphx::op::batch_norm_inference batch_norm_inference722;
    batch_norm_inference722.epsilon  = 0.001;
    batch_norm_inference722.momentum = 0.9;
    auto mx722 = p.add_instruction(batch_norm_inference722, mx721, mx105, mx104, mx103, mx102);
    migraphx::op::relu relu723;
    auto mx723 = p.add_instruction(relu723, mx722);
    migraphx::op::convolution convolution724;
    convolution724.padding  = {3, 0};
    convolution724.stride   = {1, 1};
    convolution724.dilation = {1, 1};
    convolution724.group    = 1;
    auto mx724              = p.add_instruction(convolution724, mx723, mx101);
    migraphx::op::batch_norm_inference batch_norm_inference725;
    batch_norm_inference725.epsilon  = 0.001;
    batch_norm_inference725.momentum = 0.9;
    auto mx725 = p.add_instruction(batch_norm_inference725, mx724, mx100, mx99, mx98, mx97);
    migraphx::op::relu relu726;
    auto mx726 = p.add_instruction(relu726, mx725);
    migraphx::op::convolution convolution727;
    convolution727.padding  = {0, 0};
    convolution727.stride   = {2, 2};
    convolution727.dilation = {1, 1};
    convolution727.group    = 1;
    auto mx727              = p.add_instruction(convolution727, mx726, mx96);
    migraphx::op::batch_norm_inference batch_norm_inference728;
    batch_norm_inference728.epsilon  = 0.001;
    batch_norm_inference728.momentum = 0.9;
    auto mx728 = p.add_instruction(batch_norm_inference728, mx727, mx95, mx94, mx93, mx92);
    migraphx::op::relu relu729;
    auto mx729 = p.add_instruction(relu729, mx728);
    migraphx::op::pooling pooling730;
    pooling730.mode    = "max ";
    pooling730.padding = {0, 0};
    pooling730.stride  = {2, 2};
    pooling730.lengths = {3, 3};
    auto mx730         = p.add_instruction(pooling730, mx711);
    migraphx::op::concat concat731;
    concat731.axis = 1;
    auto mx731     = p.add_instruction(concat731, mx717, mx729, mx730);
    migraphx::op::convolution convolution732;
    convolution732.padding  = {0, 0};
    convolution732.stride   = {1, 1};
    convolution732.dilation = {1, 1};
    convolution732.group    = 1;
    auto mx732              = p.add_instruction(convolution732, mx731, mx91);
    migraphx::op::batch_norm_inference batch_norm_inference733;
    batch_norm_inference733.epsilon  = 0.001;
    batch_norm_inference733.momentum = 0.9;
    auto mx733 = p.add_instruction(batch_norm_inference733, mx732, mx90, mx89, mx88, mx87);
    migraphx::op::relu relu734;
    auto mx734 = p.add_instruction(relu734, mx733);
    migraphx::op::convolution convolution735;
    convolution735.padding  = {0, 0};
    convolution735.stride   = {1, 1};
    convolution735.dilation = {1, 1};
    convolution735.group    = 1;
    auto mx735              = p.add_instruction(convolution735, mx731, mx86);
    migraphx::op::batch_norm_inference batch_norm_inference736;
    batch_norm_inference736.epsilon  = 0.001;
    batch_norm_inference736.momentum = 0.9;
    auto mx736 = p.add_instruction(batch_norm_inference736, mx735, mx85, mx84, mx83, mx82);
    migraphx::op::relu relu737;
    auto mx737 = p.add_instruction(relu737, mx736);
    migraphx::op::convolution convolution738;
    convolution738.padding  = {0, 1};
    convolution738.stride   = {1, 1};
    convolution738.dilation = {1, 1};
    convolution738.group    = 1;
    auto mx738              = p.add_instruction(convolution738, mx737, mx81);
    migraphx::op::batch_norm_inference batch_norm_inference739;
    batch_norm_inference739.epsilon  = 0.001;
    batch_norm_inference739.momentum = 0.9;
    auto mx739 = p.add_instruction(batch_norm_inference739, mx738, mx80, mx79, mx78, mx77);
    migraphx::op::relu relu740;
    auto mx740 = p.add_instruction(relu740, mx739);
    migraphx::op::convolution convolution741;
    convolution741.padding  = {1, 0};
    convolution741.stride   = {1, 1};
    convolution741.dilation = {1, 1};
    convolution741.group    = 1;
    auto mx741              = p.add_instruction(convolution741, mx737, mx76);
    migraphx::op::batch_norm_inference batch_norm_inference742;
    batch_norm_inference742.epsilon  = 0.001;
    batch_norm_inference742.momentum = 0.9;
    auto mx742 = p.add_instruction(batch_norm_inference742, mx741, mx75, mx74, mx73, mx72);
    migraphx::op::relu relu743;
    auto mx743 = p.add_instruction(relu743, mx742);
    migraphx::op::concat concat744;
    concat744.axis = 1;
    auto mx744     = p.add_instruction(concat744, mx740, mx743);
    migraphx::op::convolution convolution745;
    convolution745.padding  = {0, 0};
    convolution745.stride   = {1, 1};
    convolution745.dilation = {1, 1};
    convolution745.group    = 1;
    auto mx745              = p.add_instruction(convolution745, mx731, mx71);
    migraphx::op::batch_norm_inference batch_norm_inference746;
    batch_norm_inference746.epsilon  = 0.001;
    batch_norm_inference746.momentum = 0.9;
    auto mx746 = p.add_instruction(batch_norm_inference746, mx745, mx70, mx69, mx68, mx67);
    migraphx::op::relu relu747;
    auto mx747 = p.add_instruction(relu747, mx746);
    migraphx::op::convolution convolution748;
    convolution748.padding  = {1, 1};
    convolution748.stride   = {1, 1};
    convolution748.dilation = {1, 1};
    convolution748.group    = 1;
    auto mx748              = p.add_instruction(convolution748, mx747, mx66);
    migraphx::op::batch_norm_inference batch_norm_inference749;
    batch_norm_inference749.epsilon  = 0.001;
    batch_norm_inference749.momentum = 0.9;
    auto mx749 = p.add_instruction(batch_norm_inference749, mx748, mx65, mx64, mx63, mx62);
    migraphx::op::relu relu750;
    auto mx750 = p.add_instruction(relu750, mx749);
    migraphx::op::convolution convolution751;
    convolution751.padding  = {0, 1};
    convolution751.stride   = {1, 1};
    convolution751.dilation = {1, 1};
    convolution751.group    = 1;
    auto mx751              = p.add_instruction(convolution751, mx750, mx61);
    migraphx::op::batch_norm_inference batch_norm_inference752;
    batch_norm_inference752.epsilon  = 0.001;
    batch_norm_inference752.momentum = 0.9;
    auto mx752 = p.add_instruction(batch_norm_inference752, mx751, mx60, mx59, mx58, mx57);
    migraphx::op::relu relu753;
    auto mx753 = p.add_instruction(relu753, mx752);
    migraphx::op::convolution convolution754;
    convolution754.padding  = {1, 0};
    convolution754.stride   = {1, 1};
    convolution754.dilation = {1, 1};
    convolution754.group    = 1;
    auto mx754              = p.add_instruction(convolution754, mx750, mx56);
    migraphx::op::batch_norm_inference batch_norm_inference755;
    batch_norm_inference755.epsilon  = 0.001;
    batch_norm_inference755.momentum = 0.9;
    auto mx755 = p.add_instruction(batch_norm_inference755, mx754, mx55, mx54, mx53, mx52);
    migraphx::op::relu relu756;
    auto mx756 = p.add_instruction(relu756, mx755);
    migraphx::op::concat concat757;
    concat757.axis = 1;
    auto mx757     = p.add_instruction(concat757, mx753, mx756);
    migraphx::op::pooling pooling758;
    pooling758.mode    = "average ";
    pooling758.padding = {1, 1};
    pooling758.stride  = {1, 1};
    pooling758.lengths = {3, 3};
    auto mx758         = p.add_instruction(pooling758, mx731);
    migraphx::op::convolution convolution759;
    convolution759.padding  = {0, 0};
    convolution759.stride   = {1, 1};
    convolution759.dilation = {1, 1};
    convolution759.group    = 1;
    auto mx759              = p.add_instruction(convolution759, mx758, mx51);
    migraphx::op::batch_norm_inference batch_norm_inference760;
    batch_norm_inference760.epsilon  = 0.001;
    batch_norm_inference760.momentum = 0.9;
    auto mx760 = p.add_instruction(batch_norm_inference760, mx759, mx50, mx49, mx48, mx47);
    migraphx::op::relu relu761;
    auto mx761 = p.add_instruction(relu761, mx760);
    migraphx::op::concat concat762;
    concat762.axis = 1;
    auto mx762     = p.add_instruction(concat762, mx734, mx744, mx757, mx761);
    migraphx::op::convolution convolution763;
    convolution763.padding  = {0, 0};
    convolution763.stride   = {1, 1};
    convolution763.dilation = {1, 1};
    convolution763.group    = 1;
    auto mx763              = p.add_instruction(convolution763, mx762, mx46);
    migraphx::op::batch_norm_inference batch_norm_inference764;
    batch_norm_inference764.epsilon  = 0.001;
    batch_norm_inference764.momentum = 0.9;
    auto mx764 = p.add_instruction(batch_norm_inference764, mx763, mx45, mx44, mx43, mx42);
    migraphx::op::relu relu765;
    auto mx765 = p.add_instruction(relu765, mx764);
    migraphx::op::convolution convolution766;
    convolution766.padding  = {0, 0};
    convolution766.stride   = {1, 1};
    convolution766.dilation = {1, 1};
    convolution766.group    = 1;
    auto mx766              = p.add_instruction(convolution766, mx762, mx41);
    migraphx::op::batch_norm_inference batch_norm_inference767;
    batch_norm_inference767.epsilon  = 0.001;
    batch_norm_inference767.momentum = 0.9;
    auto mx767 = p.add_instruction(batch_norm_inference767, mx766, mx40, mx39, mx38, mx37);
    migraphx::op::relu relu768;
    auto mx768 = p.add_instruction(relu768, mx767);
    migraphx::op::convolution convolution769;
    convolution769.padding  = {0, 1};
    convolution769.stride   = {1, 1};
    convolution769.dilation = {1, 1};
    convolution769.group    = 1;
    auto mx769              = p.add_instruction(convolution769, mx768, mx36);
    migraphx::op::batch_norm_inference batch_norm_inference770;
    batch_norm_inference770.epsilon  = 0.001;
    batch_norm_inference770.momentum = 0.9;
    auto mx770 = p.add_instruction(batch_norm_inference770, mx769, mx35, mx34, mx33, mx32);
    migraphx::op::relu relu771;
    auto mx771 = p.add_instruction(relu771, mx770);
    migraphx::op::convolution convolution772;
    convolution772.padding  = {1, 0};
    convolution772.stride   = {1, 1};
    convolution772.dilation = {1, 1};
    convolution772.group    = 1;
    auto mx772              = p.add_instruction(convolution772, mx768, mx31);
    migraphx::op::batch_norm_inference batch_norm_inference773;
    batch_norm_inference773.epsilon  = 0.001;
    batch_norm_inference773.momentum = 0.9;
    auto mx773 = p.add_instruction(batch_norm_inference773, mx772, mx30, mx29, mx28, mx27);
    migraphx::op::relu relu774;
    auto mx774 = p.add_instruction(relu774, mx773);
    migraphx::op::concat concat775;
    concat775.axis = 1;
    auto mx775     = p.add_instruction(concat775, mx771, mx774);
    migraphx::op::convolution convolution776;
    convolution776.padding  = {0, 0};
    convolution776.stride   = {1, 1};
    convolution776.dilation = {1, 1};
    convolution776.group    = 1;
    auto mx776              = p.add_instruction(convolution776, mx762, mx26);
    migraphx::op::batch_norm_inference batch_norm_inference777;
    batch_norm_inference777.epsilon  = 0.001;
    batch_norm_inference777.momentum = 0.9;
    auto mx777 = p.add_instruction(batch_norm_inference777, mx776, mx25, mx24, mx23, mx22);
    migraphx::op::relu relu778;
    auto mx778 = p.add_instruction(relu778, mx777);
    migraphx::op::convolution convolution779;
    convolution779.padding  = {1, 1};
    convolution779.stride   = {1, 1};
    convolution779.dilation = {1, 1};
    convolution779.group    = 1;
    auto mx779              = p.add_instruction(convolution779, mx778, mx21);
    migraphx::op::batch_norm_inference batch_norm_inference780;
    batch_norm_inference780.epsilon  = 0.001;
    batch_norm_inference780.momentum = 0.9;
    auto mx780 = p.add_instruction(batch_norm_inference780, mx779, mx20, mx19, mx18, mx17);
    migraphx::op::relu relu781;
    auto mx781 = p.add_instruction(relu781, mx780);
    migraphx::op::convolution convolution782;
    convolution782.padding  = {0, 1};
    convolution782.stride   = {1, 1};
    convolution782.dilation = {1, 1};
    convolution782.group    = 1;
    auto mx782              = p.add_instruction(convolution782, mx781, mx16);
    migraphx::op::batch_norm_inference batch_norm_inference783;
    batch_norm_inference783.epsilon  = 0.001;
    batch_norm_inference783.momentum = 0.9;
    auto mx783 = p.add_instruction(batch_norm_inference783, mx782, mx15, mx14, mx13, mx12);
    migraphx::op::relu relu784;
    auto mx784 = p.add_instruction(relu784, mx783);
    migraphx::op::convolution convolution785;
    convolution785.padding  = {1, 0};
    convolution785.stride   = {1, 1};
    convolution785.dilation = {1, 1};
    convolution785.group    = 1;
    auto mx785              = p.add_instruction(convolution785, mx781, mx11);
    migraphx::op::batch_norm_inference batch_norm_inference786;
    batch_norm_inference786.epsilon  = 0.001;
    batch_norm_inference786.momentum = 0.9;
    auto mx786 = p.add_instruction(batch_norm_inference786, mx785, mx10, mx9, mx8, mx7);
    migraphx::op::relu relu787;
    auto mx787 = p.add_instruction(relu787, mx786);
    migraphx::op::concat concat788;
    concat788.axis = 1;
    auto mx788     = p.add_instruction(concat788, mx784, mx787);
    migraphx::op::pooling pooling789;
    pooling789.mode    = "average ";
    pooling789.padding = {1, 1};
    pooling789.stride  = {1, 1};
    pooling789.lengths = {3, 3};
    auto mx789         = p.add_instruction(pooling789, mx762);
    migraphx::op::convolution convolution790;
    convolution790.padding  = {0, 0};
    convolution790.stride   = {1, 1};
    convolution790.dilation = {1, 1};
    convolution790.group    = 1;
    auto mx790              = p.add_instruction(convolution790, mx789, mx6);
    migraphx::op::batch_norm_inference batch_norm_inference791;
    batch_norm_inference791.epsilon  = 0.001;
    batch_norm_inference791.momentum = 0.9;
    auto mx791 = p.add_instruction(batch_norm_inference791, mx790, mx5, mx4, mx3, mx2);
    migraphx::op::relu relu792;
    auto mx792 = p.add_instruction(relu792, mx791);
    migraphx::op::concat concat793;
    concat793.axis = 1;
    auto mx793     = p.add_instruction(concat793, mx765, mx775, mx788, mx792);
    migraphx::op::pooling pooling794;
    pooling794.mode    = "average ";
    pooling794.padding = {0, 0};
    pooling794.stride  = {8, 8};
    pooling794.lengths = {8, 8};
    auto mx794         = p.add_instruction(pooling794, mx793);
    migraphx::op::identity identity795;
    auto mx795 = p.add_instruction(identity795, mx794);
    migraphx::op::flatten flatten796;
    flatten796.axis = 1;
    auto mx796      = p.add_instruction(flatten796, mx795);
    migraphx::op::transpose transpose797;
    transpose797.dims = {1, 0};
    auto mx797        = p.add_instruction(transpose797, mx1);
    migraphx::op::multibroadcast multibroadcast798;
    multibroadcast798.output_lens = {batch, 1000};
    auto mx798                    = p.add_instruction(multibroadcast798, mx0);
    migraphx::op::dot dot799;
    dot799.alpha = 1;
    dot799.beta  = 1;
    p.add_instruction(dot799, mx796, mx797, mx798);

    return p;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx
