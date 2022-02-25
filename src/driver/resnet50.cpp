#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/apply_alpha_beta.hpp>
#include "models.hpp"

namespace migraphx {
namespace driver {
inline namespace MIGRAPHX_INLINE_NS {

migraphx::program resnet50(unsigned batch) // NOLINT(readability-function-size)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto m0 =
        mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {batch, 3, 224, 224}});
    auto mx0 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1000}}, 0));
    auto mx1 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1000, 2048}}, 1));
    auto mx2 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 2)));
    auto mx3 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 3));
    auto mx4 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 4));
    auto mx5 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 5)));
    auto mx6 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {2048, 512, 1, 1}}, 6));
    auto mx7 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 7)));
    auto mx8 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 8));
    auto mx9 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 9));
    auto mx10 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 10)));
    auto mx11 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 512, 3, 3}}, 11));
    auto mx12 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 12)));
    auto mx13 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 13));
    auto mx14 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 14));
    auto mx15 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 15)));
    auto mx16 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 2048, 1, 1}}, 16));
    auto mx17 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 17)));
    auto mx18 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 18));
    auto mx19 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 19));
    auto mx20 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 20)));
    auto mx21 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {2048, 512, 1, 1}}, 21));
    auto mx22 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 22)));
    auto mx23 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 23));
    auto mx24 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 24));
    auto mx25 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 25)));
    auto mx26 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 512, 3, 3}}, 26));
    auto mx27 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 27)));
    auto mx28 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 28));
    auto mx29 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 29));
    auto mx30 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 30)));
    auto mx31 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 2048, 1, 1}}, 31));
    auto mx32 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 32)));
    auto mx33 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 33));
    auto mx34 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 34));
    auto mx35 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 35)));
    auto mx36 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {2048, 1024, 1, 1}}, 36));
    auto mx37 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 37)));
    auto mx38 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 38));
    auto mx39 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 39));
    auto mx40 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2048}}, 40)));
    auto mx41 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {2048, 512, 1, 1}}, 41));
    auto mx42 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 42)));
    auto mx43 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 43));
    auto mx44 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 44));
    auto mx45 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 45)));
    auto mx46 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 512, 3, 3}}, 46));
    auto mx47 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 47)));
    auto mx48 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 48));
    auto mx49 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 49));
    auto mx50 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 50)));
    auto mx51 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 1024, 1, 1}}, 51));
    auto mx52 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 52)));
    auto mx53 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 53));
    auto mx54 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 54));
    auto mx55 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 55));
    auto mx56 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 256, 1, 1}}, 56));
    auto mx57 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 57)));
    auto mx58 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 58));
    auto mx59 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 59));
    auto mx60 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 60)));
    auto mx61 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 256, 3, 3}}, 61));
    auto mx62 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 62)));
    auto mx63 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 63));
    auto mx64 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 64));
    auto mx65 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 65)));
    auto mx66 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 1024, 1, 1}}, 66));
    auto mx67 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 67)));
    auto mx68 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 68));
    auto mx69 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 69));
    auto mx70 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 70));
    auto mx71 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 256, 1, 1}}, 71));
    auto mx72 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 72)));
    auto mx73 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 73));
    auto mx74 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 74));
    auto mx75 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 75)));
    auto mx76 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 256, 3, 3}}, 76));
    auto mx77 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 77)));
    auto mx78 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 78));
    auto mx79 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 79));
    auto mx80 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 80)));
    auto mx81 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 1024, 1, 1}}, 81));
    auto mx82 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 82)));
    auto mx83 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 83));
    auto mx84 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 84));
    auto mx85 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 85));
    auto mx86 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 256, 1, 1}}, 86));
    auto mx87 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 87)));
    auto mx88 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 88));
    auto mx89 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 89));
    auto mx90 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 90)));
    auto mx91 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 256, 3, 3}}, 91));
    auto mx92 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 92)));
    auto mx93 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 93));
    auto mx94 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 94));
    auto mx95 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 95)));
    auto mx96 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 1024, 1, 1}}, 96));
    auto mx97 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 97)));
    auto mx98 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 98));
    auto mx99 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 99));
    auto mx100 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 100));
    auto mx101 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 256, 1, 1}}, 101));
    auto mx102 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 102)));
    auto mx103 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 103));
    auto mx104 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 104));
    auto mx105 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 105)));
    auto mx106 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 256, 3, 3}}, 106));
    auto mx107 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 107)));
    auto mx108 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 108));
    auto mx109 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 109));
    auto mx110 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 110)));
    auto mx111 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 1024, 1, 1}}, 111));
    auto mx112 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 112)));
    auto mx113 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 113));
    auto mx114 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 114));
    auto mx115 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 115));
    auto mx116 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 256, 1, 1}}, 116));
    auto mx117 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 117)));
    auto mx118 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 118));
    auto mx119 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 119));
    auto mx120 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 120)));
    auto mx121 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 256, 3, 3}}, 121));
    auto mx122 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 122)));
    auto mx123 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 123));
    auto mx124 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 124));
    auto mx125 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 125)));
    auto mx126 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 1024, 1, 1}}, 126));
    auto mx127 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 127)));
    auto mx128 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 128));
    auto mx129 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 129));
    auto mx130 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 130));
    auto mx131 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 512, 1, 1}}, 131));
    auto mx132 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 132)));
    auto mx133 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 133));
    auto mx134 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 134));
    auto mx135 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1024}}, 135));
    auto mx136 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {1024, 256, 1, 1}}, 136));
    auto mx137 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 137)));
    auto mx138 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 138));
    auto mx139 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 139));
    auto mx140 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 140)));
    auto mx141 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 256, 3, 3}}, 141));
    auto mx142 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 142)));
    auto mx143 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 143));
    auto mx144 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 144));
    auto mx145 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 145)));
    auto mx146 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 512, 1, 1}}, 146));
    auto mx147 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 147)));
    auto mx148 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 148));
    auto mx149 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 149));
    auto mx150 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 150));
    auto mx151 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 128, 1, 1}}, 151));
    auto mx152 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 152)));
    auto mx153 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 153));
    auto mx154 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 154));
    auto mx155 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 155)));
    auto mx156 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 3, 3}}, 156));
    auto mx157 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 157)));
    auto mx158 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 158));
    auto mx159 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 159));
    auto mx160 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 160)));
    auto mx161 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 512, 1, 1}}, 161));
    auto mx162 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 162)));
    auto mx163 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 163));
    auto mx164 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 164));
    auto mx165 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 165));
    auto mx166 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 128, 1, 1}}, 166));
    auto mx167 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 167)));
    auto mx168 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 168));
    auto mx169 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 169));
    auto mx170 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 170)));
    auto mx171 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 3, 3}}, 171));
    auto mx172 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 172)));
    auto mx173 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 173));
    auto mx174 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 174));
    auto mx175 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 175)));
    auto mx176 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 512, 1, 1}}, 176));
    auto mx177 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 177)));
    auto mx178 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 178));
    auto mx179 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 179));
    auto mx180 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 180));
    auto mx181 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 128, 1, 1}}, 181));
    auto mx182 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 182)));
    auto mx183 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 183));
    auto mx184 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 184));
    auto mx185 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 185)));
    auto mx186 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 3, 3}}, 186));
    auto mx187 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 187)));
    auto mx188 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 188));
    auto mx189 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 189));
    auto mx190 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 190)));
    auto mx191 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 512, 1, 1}}, 191));
    auto mx192 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 192)));
    auto mx193 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 193));
    auto mx194 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 194));
    auto mx195 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 195));
    auto mx196 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 256, 1, 1}}, 196));
    auto mx197 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 197)));
    auto mx198 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 198));
    auto mx199 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 199));
    auto mx200 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {512}}, 200));
    auto mx201 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {512, 128, 1, 1}}, 201));
    auto mx202 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 202)));
    auto mx203 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 203));
    auto mx204 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 204));
    auto mx205 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 205)));
    auto mx206 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 128, 3, 3}}, 206));
    auto mx207 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 207)));
    auto mx208 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 208));
    auto mx209 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 209));
    auto mx210 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {128}}, 210)));
    auto mx211 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {128, 256, 1, 1}}, 211));
    auto mx212 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 212)));
    auto mx213 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 213));
    auto mx214 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 214));
    auto mx215 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 215));
    auto mx216 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 64, 1, 1}}, 216));
    auto mx217 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 217)));
    auto mx218 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 218));
    auto mx219 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 219));
    auto mx220 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 220)));
    auto mx221 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 64, 3, 3}}, 221));
    auto mx222 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 222)));
    auto mx223 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 223));
    auto mx224 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 224));
    auto mx225 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 225)));
    auto mx226 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 256, 1, 1}}, 226));
    auto mx227 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 227)));
    auto mx228 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 228));
    auto mx229 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 229));
    auto mx230 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 230));
    auto mx231 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 64, 1, 1}}, 231));
    auto mx232 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 232)));
    auto mx233 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 233));
    auto mx234 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 234));
    auto mx235 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 235)));
    auto mx236 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 64, 3, 3}}, 236));
    auto mx237 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 237)));
    auto mx238 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 238));
    auto mx239 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 239));
    auto mx240 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 240)));
    auto mx241 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 256, 1, 1}}, 241));
    auto mx242 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 242)));
    auto mx243 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 243));
    auto mx244 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 244));
    auto mx245 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 245));
    auto mx246 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 64, 1, 1}}, 246));
    auto mx247 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 247)));
    auto mx248 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 248));
    auto mx249 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 249));
    auto mx250 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 250));
    auto mx251 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 64, 1, 1}}, 251));
    auto mx252 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 252)));
    auto mx253 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 253));
    auto mx254 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 254));
    auto mx255 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 255)));
    auto mx256 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 64, 3, 3}}, 256));
    auto mx257 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 257)));
    auto mx258 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 258));
    auto mx259 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 259));
    auto mx260 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 260)));
    auto mx261 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 64, 1, 1}}, 261));
    auto mx262 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 262)));
    auto mx263 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 263));
    auto mx264 = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 264));
    auto mx265 = mm->add_literal(migraphx::abs(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 265)));
    auto mx266 = mm->add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 3, 7, 7}}, 266));
    migraphx::op::convolution convolution267;
    convolution267.padding  = {3, 3};
    convolution267.stride   = {2, 2};
    convolution267.dilation = {1, 1};
    convolution267.group    = 1;
    auto mx267              = mm->add_instruction(convolution267, m0, mx266);
    migraphx::op::batch_norm_inference batch_norm_inference268;
    batch_norm_inference268.epsilon  = 1e-05;
    batch_norm_inference268.momentum = 0.9;
    auto mx268 = mm->add_instruction(batch_norm_inference268, mx267, mx265, mx264, mx263, mx262);
    migraphx::op::relu relu269;
    auto mx269 = mm->add_instruction(relu269, mx268);
    migraphx::op::pooling pooling270;
    pooling270.mode    = migraphx::op::pooling_mode::max;
    pooling270.padding = {1, 1};
    pooling270.stride  = {2, 2};
    pooling270.lengths = {3, 3};
    auto mx270         = mm->add_instruction(pooling270, mx269);
    migraphx::op::convolution convolution271;
    convolution271.padding  = {0, 0};
    convolution271.stride   = {1, 1};
    convolution271.dilation = {1, 1};
    convolution271.group    = 1;
    auto mx271              = mm->add_instruction(convolution271, mx270, mx261);
    migraphx::op::batch_norm_inference batch_norm_inference272;
    batch_norm_inference272.epsilon  = 1e-05;
    batch_norm_inference272.momentum = 0.9;
    auto mx272 = mm->add_instruction(batch_norm_inference272, mx271, mx260, mx259, mx258, mx257);
    migraphx::op::relu relu273;
    auto mx273 = mm->add_instruction(relu273, mx272);
    migraphx::op::convolution convolution274;
    convolution274.padding  = {1, 1};
    convolution274.stride   = {1, 1};
    convolution274.dilation = {1, 1};
    convolution274.group    = 1;
    auto mx274              = mm->add_instruction(convolution274, mx273, mx256);
    migraphx::op::batch_norm_inference batch_norm_inference275;
    batch_norm_inference275.epsilon  = 1e-05;
    batch_norm_inference275.momentum = 0.9;
    auto mx275 = mm->add_instruction(batch_norm_inference275, mx274, mx255, mx254, mx253, mx252);
    migraphx::op::relu relu276;
    auto mx276 = mm->add_instruction(relu276, mx275);
    migraphx::op::convolution convolution277;
    convolution277.padding  = {0, 0};
    convolution277.stride   = {1, 1};
    convolution277.dilation = {1, 1};
    convolution277.group    = 1;
    auto mx277              = mm->add_instruction(convolution277, mx276, mx251);
    migraphx::op::batch_norm_inference batch_norm_inference278;
    batch_norm_inference278.epsilon  = 1e-05;
    batch_norm_inference278.momentum = 0.9;
    auto mx278 = mm->add_instruction(batch_norm_inference278, mx277, mx250, mx249, mx248, mx247);
    migraphx::op::convolution convolution279;
    convolution279.padding  = {0, 0};
    convolution279.stride   = {1, 1};
    convolution279.dilation = {1, 1};
    convolution279.group    = 1;
    auto mx279              = mm->add_instruction(convolution279, mx270, mx246);
    migraphx::op::batch_norm_inference batch_norm_inference280;
    batch_norm_inference280.epsilon  = 1e-05;
    batch_norm_inference280.momentum = 0.9;
    auto mx280 = mm->add_instruction(batch_norm_inference280, mx279, mx245, mx244, mx243, mx242);
    migraphx::op::add add281;
    auto mx281 = mm->add_instruction(add281, mx278, mx280);
    migraphx::op::relu relu282;
    auto mx282 = mm->add_instruction(relu282, mx281);
    migraphx::op::convolution convolution283;
    convolution283.padding  = {0, 0};
    convolution283.stride   = {1, 1};
    convolution283.dilation = {1, 1};
    convolution283.group    = 1;
    auto mx283              = mm->add_instruction(convolution283, mx282, mx241);
    migraphx::op::batch_norm_inference batch_norm_inference284;
    batch_norm_inference284.epsilon  = 1e-05;
    batch_norm_inference284.momentum = 0.9;
    auto mx284 = mm->add_instruction(batch_norm_inference284, mx283, mx240, mx239, mx238, mx237);
    migraphx::op::relu relu285;
    auto mx285 = mm->add_instruction(relu285, mx284);
    migraphx::op::convolution convolution286;
    convolution286.padding  = {1, 1};
    convolution286.stride   = {1, 1};
    convolution286.dilation = {1, 1};
    convolution286.group    = 1;
    auto mx286              = mm->add_instruction(convolution286, mx285, mx236);
    migraphx::op::batch_norm_inference batch_norm_inference287;
    batch_norm_inference287.epsilon  = 1e-05;
    batch_norm_inference287.momentum = 0.9;
    auto mx287 = mm->add_instruction(batch_norm_inference287, mx286, mx235, mx234, mx233, mx232);
    migraphx::op::relu relu288;
    auto mx288 = mm->add_instruction(relu288, mx287);
    migraphx::op::convolution convolution289;
    convolution289.padding  = {0, 0};
    convolution289.stride   = {1, 1};
    convolution289.dilation = {1, 1};
    convolution289.group    = 1;
    auto mx289              = mm->add_instruction(convolution289, mx288, mx231);
    migraphx::op::batch_norm_inference batch_norm_inference290;
    batch_norm_inference290.epsilon  = 1e-05;
    batch_norm_inference290.momentum = 0.9;
    auto mx290 = mm->add_instruction(batch_norm_inference290, mx289, mx230, mx229, mx228, mx227);
    migraphx::op::add add291;
    auto mx291 = mm->add_instruction(add291, mx290, mx282);
    migraphx::op::relu relu292;
    auto mx292 = mm->add_instruction(relu292, mx291);
    migraphx::op::convolution convolution293;
    convolution293.padding  = {0, 0};
    convolution293.stride   = {1, 1};
    convolution293.dilation = {1, 1};
    convolution293.group    = 1;
    auto mx293              = mm->add_instruction(convolution293, mx292, mx226);
    migraphx::op::batch_norm_inference batch_norm_inference294;
    batch_norm_inference294.epsilon  = 1e-05;
    batch_norm_inference294.momentum = 0.9;
    auto mx294 = mm->add_instruction(batch_norm_inference294, mx293, mx225, mx224, mx223, mx222);
    migraphx::op::relu relu295;
    auto mx295 = mm->add_instruction(relu295, mx294);
    migraphx::op::convolution convolution296;
    convolution296.padding  = {1, 1};
    convolution296.stride   = {1, 1};
    convolution296.dilation = {1, 1};
    convolution296.group    = 1;
    auto mx296              = mm->add_instruction(convolution296, mx295, mx221);
    migraphx::op::batch_norm_inference batch_norm_inference297;
    batch_norm_inference297.epsilon  = 1e-05;
    batch_norm_inference297.momentum = 0.9;
    auto mx297 = mm->add_instruction(batch_norm_inference297, mx296, mx220, mx219, mx218, mx217);
    migraphx::op::relu relu298;
    auto mx298 = mm->add_instruction(relu298, mx297);
    migraphx::op::convolution convolution299;
    convolution299.padding  = {0, 0};
    convolution299.stride   = {1, 1};
    convolution299.dilation = {1, 1};
    convolution299.group    = 1;
    auto mx299              = mm->add_instruction(convolution299, mx298, mx216);
    migraphx::op::batch_norm_inference batch_norm_inference300;
    batch_norm_inference300.epsilon  = 1e-05;
    batch_norm_inference300.momentum = 0.9;
    auto mx300 = mm->add_instruction(batch_norm_inference300, mx299, mx215, mx214, mx213, mx212);
    migraphx::op::add add301;
    auto mx301 = mm->add_instruction(add301, mx300, mx292);
    migraphx::op::relu relu302;
    auto mx302 = mm->add_instruction(relu302, mx301);
    migraphx::op::convolution convolution303;
    convolution303.padding  = {0, 0};
    convolution303.stride   = {1, 1};
    convolution303.dilation = {1, 1};
    convolution303.group    = 1;
    auto mx303              = mm->add_instruction(convolution303, mx302, mx211);
    migraphx::op::batch_norm_inference batch_norm_inference304;
    batch_norm_inference304.epsilon  = 1e-05;
    batch_norm_inference304.momentum = 0.9;
    auto mx304 = mm->add_instruction(batch_norm_inference304, mx303, mx210, mx209, mx208, mx207);
    migraphx::op::relu relu305;
    auto mx305 = mm->add_instruction(relu305, mx304);
    migraphx::op::convolution convolution306;
    convolution306.padding  = {1, 1};
    convolution306.stride   = {2, 2};
    convolution306.dilation = {1, 1};
    convolution306.group    = 1;
    auto mx306              = mm->add_instruction(convolution306, mx305, mx206);
    migraphx::op::batch_norm_inference batch_norm_inference307;
    batch_norm_inference307.epsilon  = 1e-05;
    batch_norm_inference307.momentum = 0.9;
    auto mx307 = mm->add_instruction(batch_norm_inference307, mx306, mx205, mx204, mx203, mx202);
    migraphx::op::relu relu308;
    auto mx308 = mm->add_instruction(relu308, mx307);
    migraphx::op::convolution convolution309;
    convolution309.padding  = {0, 0};
    convolution309.stride   = {1, 1};
    convolution309.dilation = {1, 1};
    convolution309.group    = 1;
    auto mx309              = mm->add_instruction(convolution309, mx308, mx201);
    migraphx::op::batch_norm_inference batch_norm_inference310;
    batch_norm_inference310.epsilon  = 1e-05;
    batch_norm_inference310.momentum = 0.9;
    auto mx310 = mm->add_instruction(batch_norm_inference310, mx309, mx200, mx199, mx198, mx197);
    migraphx::op::convolution convolution311;
    convolution311.padding  = {0, 0};
    convolution311.stride   = {2, 2};
    convolution311.dilation = {1, 1};
    convolution311.group    = 1;
    auto mx311              = mm->add_instruction(convolution311, mx302, mx196);
    migraphx::op::batch_norm_inference batch_norm_inference312;
    batch_norm_inference312.epsilon  = 1e-05;
    batch_norm_inference312.momentum = 0.9;
    auto mx312 = mm->add_instruction(batch_norm_inference312, mx311, mx195, mx194, mx193, mx192);
    migraphx::op::add add313;
    auto mx313 = mm->add_instruction(add313, mx310, mx312);
    migraphx::op::relu relu314;
    auto mx314 = mm->add_instruction(relu314, mx313);
    migraphx::op::convolution convolution315;
    convolution315.padding  = {0, 0};
    convolution315.stride   = {1, 1};
    convolution315.dilation = {1, 1};
    convolution315.group    = 1;
    auto mx315              = mm->add_instruction(convolution315, mx314, mx191);
    migraphx::op::batch_norm_inference batch_norm_inference316;
    batch_norm_inference316.epsilon  = 1e-05;
    batch_norm_inference316.momentum = 0.9;
    auto mx316 = mm->add_instruction(batch_norm_inference316, mx315, mx190, mx189, mx188, mx187);
    migraphx::op::relu relu317;
    auto mx317 = mm->add_instruction(relu317, mx316);
    migraphx::op::convolution convolution318;
    convolution318.padding  = {1, 1};
    convolution318.stride   = {1, 1};
    convolution318.dilation = {1, 1};
    convolution318.group    = 1;
    auto mx318              = mm->add_instruction(convolution318, mx317, mx186);
    migraphx::op::batch_norm_inference batch_norm_inference319;
    batch_norm_inference319.epsilon  = 1e-05;
    batch_norm_inference319.momentum = 0.9;
    auto mx319 = mm->add_instruction(batch_norm_inference319, mx318, mx185, mx184, mx183, mx182);
    migraphx::op::relu relu320;
    auto mx320 = mm->add_instruction(relu320, mx319);
    migraphx::op::convolution convolution321;
    convolution321.padding  = {0, 0};
    convolution321.stride   = {1, 1};
    convolution321.dilation = {1, 1};
    convolution321.group    = 1;
    auto mx321              = mm->add_instruction(convolution321, mx320, mx181);
    migraphx::op::batch_norm_inference batch_norm_inference322;
    batch_norm_inference322.epsilon  = 1e-05;
    batch_norm_inference322.momentum = 0.9;
    auto mx322 = mm->add_instruction(batch_norm_inference322, mx321, mx180, mx179, mx178, mx177);
    migraphx::op::add add323;
    auto mx323 = mm->add_instruction(add323, mx322, mx314);
    migraphx::op::relu relu324;
    auto mx324 = mm->add_instruction(relu324, mx323);
    migraphx::op::convolution convolution325;
    convolution325.padding  = {0, 0};
    convolution325.stride   = {1, 1};
    convolution325.dilation = {1, 1};
    convolution325.group    = 1;
    auto mx325              = mm->add_instruction(convolution325, mx324, mx176);
    migraphx::op::batch_norm_inference batch_norm_inference326;
    batch_norm_inference326.epsilon  = 1e-05;
    batch_norm_inference326.momentum = 0.9;
    auto mx326 = mm->add_instruction(batch_norm_inference326, mx325, mx175, mx174, mx173, mx172);
    migraphx::op::relu relu327;
    auto mx327 = mm->add_instruction(relu327, mx326);
    migraphx::op::convolution convolution328;
    convolution328.padding  = {1, 1};
    convolution328.stride   = {1, 1};
    convolution328.dilation = {1, 1};
    convolution328.group    = 1;
    auto mx328              = mm->add_instruction(convolution328, mx327, mx171);
    migraphx::op::batch_norm_inference batch_norm_inference329;
    batch_norm_inference329.epsilon  = 1e-05;
    batch_norm_inference329.momentum = 0.9;
    auto mx329 = mm->add_instruction(batch_norm_inference329, mx328, mx170, mx169, mx168, mx167);
    migraphx::op::relu relu330;
    auto mx330 = mm->add_instruction(relu330, mx329);
    migraphx::op::convolution convolution331;
    convolution331.padding  = {0, 0};
    convolution331.stride   = {1, 1};
    convolution331.dilation = {1, 1};
    convolution331.group    = 1;
    auto mx331              = mm->add_instruction(convolution331, mx330, mx166);
    migraphx::op::batch_norm_inference batch_norm_inference332;
    batch_norm_inference332.epsilon  = 1e-05;
    batch_norm_inference332.momentum = 0.9;
    auto mx332 = mm->add_instruction(batch_norm_inference332, mx331, mx165, mx164, mx163, mx162);
    migraphx::op::add add333;
    auto mx333 = mm->add_instruction(add333, mx332, mx324);
    migraphx::op::relu relu334;
    auto mx334 = mm->add_instruction(relu334, mx333);
    migraphx::op::convolution convolution335;
    convolution335.padding  = {0, 0};
    convolution335.stride   = {1, 1};
    convolution335.dilation = {1, 1};
    convolution335.group    = 1;
    auto mx335              = mm->add_instruction(convolution335, mx334, mx161);
    migraphx::op::batch_norm_inference batch_norm_inference336;
    batch_norm_inference336.epsilon  = 1e-05;
    batch_norm_inference336.momentum = 0.9;
    auto mx336 = mm->add_instruction(batch_norm_inference336, mx335, mx160, mx159, mx158, mx157);
    migraphx::op::relu relu337;
    auto mx337 = mm->add_instruction(relu337, mx336);
    migraphx::op::convolution convolution338;
    convolution338.padding  = {1, 1};
    convolution338.stride   = {1, 1};
    convolution338.dilation = {1, 1};
    convolution338.group    = 1;
    auto mx338              = mm->add_instruction(convolution338, mx337, mx156);
    migraphx::op::batch_norm_inference batch_norm_inference339;
    batch_norm_inference339.epsilon  = 1e-05;
    batch_norm_inference339.momentum = 0.9;
    auto mx339 = mm->add_instruction(batch_norm_inference339, mx338, mx155, mx154, mx153, mx152);
    migraphx::op::relu relu340;
    auto mx340 = mm->add_instruction(relu340, mx339);
    migraphx::op::convolution convolution341;
    convolution341.padding  = {0, 0};
    convolution341.stride   = {1, 1};
    convolution341.dilation = {1, 1};
    convolution341.group    = 1;
    auto mx341              = mm->add_instruction(convolution341, mx340, mx151);
    migraphx::op::batch_norm_inference batch_norm_inference342;
    batch_norm_inference342.epsilon  = 1e-05;
    batch_norm_inference342.momentum = 0.9;
    auto mx342 = mm->add_instruction(batch_norm_inference342, mx341, mx150, mx149, mx148, mx147);
    migraphx::op::add add343;
    auto mx343 = mm->add_instruction(add343, mx342, mx334);
    migraphx::op::relu relu344;
    auto mx344 = mm->add_instruction(relu344, mx343);
    migraphx::op::convolution convolution345;
    convolution345.padding  = {0, 0};
    convolution345.stride   = {1, 1};
    convolution345.dilation = {1, 1};
    convolution345.group    = 1;
    auto mx345              = mm->add_instruction(convolution345, mx344, mx146);
    migraphx::op::batch_norm_inference batch_norm_inference346;
    batch_norm_inference346.epsilon  = 1e-05;
    batch_norm_inference346.momentum = 0.9;
    auto mx346 = mm->add_instruction(batch_norm_inference346, mx345, mx145, mx144, mx143, mx142);
    migraphx::op::relu relu347;
    auto mx347 = mm->add_instruction(relu347, mx346);
    migraphx::op::convolution convolution348;
    convolution348.padding  = {1, 1};
    convolution348.stride   = {2, 2};
    convolution348.dilation = {1, 1};
    convolution348.group    = 1;
    auto mx348              = mm->add_instruction(convolution348, mx347, mx141);
    migraphx::op::batch_norm_inference batch_norm_inference349;
    batch_norm_inference349.epsilon  = 1e-05;
    batch_norm_inference349.momentum = 0.9;
    auto mx349 = mm->add_instruction(batch_norm_inference349, mx348, mx140, mx139, mx138, mx137);
    migraphx::op::relu relu350;
    auto mx350 = mm->add_instruction(relu350, mx349);
    migraphx::op::convolution convolution351;
    convolution351.padding  = {0, 0};
    convolution351.stride   = {1, 1};
    convolution351.dilation = {1, 1};
    convolution351.group    = 1;
    auto mx351              = mm->add_instruction(convolution351, mx350, mx136);
    migraphx::op::batch_norm_inference batch_norm_inference352;
    batch_norm_inference352.epsilon  = 1e-05;
    batch_norm_inference352.momentum = 0.9;
    auto mx352 = mm->add_instruction(batch_norm_inference352, mx351, mx135, mx134, mx133, mx132);
    migraphx::op::convolution convolution353;
    convolution353.padding  = {0, 0};
    convolution353.stride   = {2, 2};
    convolution353.dilation = {1, 1};
    convolution353.group    = 1;
    auto mx353              = mm->add_instruction(convolution353, mx344, mx131);
    migraphx::op::batch_norm_inference batch_norm_inference354;
    batch_norm_inference354.epsilon  = 1e-05;
    batch_norm_inference354.momentum = 0.9;
    auto mx354 = mm->add_instruction(batch_norm_inference354, mx353, mx130, mx129, mx128, mx127);
    migraphx::op::add add355;
    auto mx355 = mm->add_instruction(add355, mx352, mx354);
    migraphx::op::relu relu356;
    auto mx356 = mm->add_instruction(relu356, mx355);
    migraphx::op::convolution convolution357;
    convolution357.padding  = {0, 0};
    convolution357.stride   = {1, 1};
    convolution357.dilation = {1, 1};
    convolution357.group    = 1;
    auto mx357              = mm->add_instruction(convolution357, mx356, mx126);
    migraphx::op::batch_norm_inference batch_norm_inference358;
    batch_norm_inference358.epsilon  = 1e-05;
    batch_norm_inference358.momentum = 0.9;
    auto mx358 = mm->add_instruction(batch_norm_inference358, mx357, mx125, mx124, mx123, mx122);
    migraphx::op::relu relu359;
    auto mx359 = mm->add_instruction(relu359, mx358);
    migraphx::op::convolution convolution360;
    convolution360.padding  = {1, 1};
    convolution360.stride   = {1, 1};
    convolution360.dilation = {1, 1};
    convolution360.group    = 1;
    auto mx360              = mm->add_instruction(convolution360, mx359, mx121);
    migraphx::op::batch_norm_inference batch_norm_inference361;
    batch_norm_inference361.epsilon  = 1e-05;
    batch_norm_inference361.momentum = 0.9;
    auto mx361 = mm->add_instruction(batch_norm_inference361, mx360, mx120, mx119, mx118, mx117);
    migraphx::op::relu relu362;
    auto mx362 = mm->add_instruction(relu362, mx361);
    migraphx::op::convolution convolution363;
    convolution363.padding  = {0, 0};
    convolution363.stride   = {1, 1};
    convolution363.dilation = {1, 1};
    convolution363.group    = 1;
    auto mx363              = mm->add_instruction(convolution363, mx362, mx116);
    migraphx::op::batch_norm_inference batch_norm_inference364;
    batch_norm_inference364.epsilon  = 1e-05;
    batch_norm_inference364.momentum = 0.9;
    auto mx364 = mm->add_instruction(batch_norm_inference364, mx363, mx115, mx114, mx113, mx112);
    migraphx::op::add add365;
    auto mx365 = mm->add_instruction(add365, mx364, mx356);
    migraphx::op::relu relu366;
    auto mx366 = mm->add_instruction(relu366, mx365);
    migraphx::op::convolution convolution367;
    convolution367.padding  = {0, 0};
    convolution367.stride   = {1, 1};
    convolution367.dilation = {1, 1};
    convolution367.group    = 1;
    auto mx367              = mm->add_instruction(convolution367, mx366, mx111);
    migraphx::op::batch_norm_inference batch_norm_inference368;
    batch_norm_inference368.epsilon  = 1e-05;
    batch_norm_inference368.momentum = 0.9;
    auto mx368 = mm->add_instruction(batch_norm_inference368, mx367, mx110, mx109, mx108, mx107);
    migraphx::op::relu relu369;
    auto mx369 = mm->add_instruction(relu369, mx368);
    migraphx::op::convolution convolution370;
    convolution370.padding  = {1, 1};
    convolution370.stride   = {1, 1};
    convolution370.dilation = {1, 1};
    convolution370.group    = 1;
    auto mx370              = mm->add_instruction(convolution370, mx369, mx106);
    migraphx::op::batch_norm_inference batch_norm_inference371;
    batch_norm_inference371.epsilon  = 1e-05;
    batch_norm_inference371.momentum = 0.9;
    auto mx371 = mm->add_instruction(batch_norm_inference371, mx370, mx105, mx104, mx103, mx102);
    migraphx::op::relu relu372;
    auto mx372 = mm->add_instruction(relu372, mx371);
    migraphx::op::convolution convolution373;
    convolution373.padding  = {0, 0};
    convolution373.stride   = {1, 1};
    convolution373.dilation = {1, 1};
    convolution373.group    = 1;
    auto mx373              = mm->add_instruction(convolution373, mx372, mx101);
    migraphx::op::batch_norm_inference batch_norm_inference374;
    batch_norm_inference374.epsilon  = 1e-05;
    batch_norm_inference374.momentum = 0.9;
    auto mx374 = mm->add_instruction(batch_norm_inference374, mx373, mx100, mx99, mx98, mx97);
    migraphx::op::add add375;
    auto mx375 = mm->add_instruction(add375, mx374, mx366);
    migraphx::op::relu relu376;
    auto mx376 = mm->add_instruction(relu376, mx375);
    migraphx::op::convolution convolution377;
    convolution377.padding  = {0, 0};
    convolution377.stride   = {1, 1};
    convolution377.dilation = {1, 1};
    convolution377.group    = 1;
    auto mx377              = mm->add_instruction(convolution377, mx376, mx96);
    migraphx::op::batch_norm_inference batch_norm_inference378;
    batch_norm_inference378.epsilon  = 1e-05;
    batch_norm_inference378.momentum = 0.9;
    auto mx378 = mm->add_instruction(batch_norm_inference378, mx377, mx95, mx94, mx93, mx92);
    migraphx::op::relu relu379;
    auto mx379 = mm->add_instruction(relu379, mx378);
    migraphx::op::convolution convolution380;
    convolution380.padding  = {1, 1};
    convolution380.stride   = {1, 1};
    convolution380.dilation = {1, 1};
    convolution380.group    = 1;
    auto mx380              = mm->add_instruction(convolution380, mx379, mx91);
    migraphx::op::batch_norm_inference batch_norm_inference381;
    batch_norm_inference381.epsilon  = 1e-05;
    batch_norm_inference381.momentum = 0.9;
    auto mx381 = mm->add_instruction(batch_norm_inference381, mx380, mx90, mx89, mx88, mx87);
    migraphx::op::relu relu382;
    auto mx382 = mm->add_instruction(relu382, mx381);
    migraphx::op::convolution convolution383;
    convolution383.padding  = {0, 0};
    convolution383.stride   = {1, 1};
    convolution383.dilation = {1, 1};
    convolution383.group    = 1;
    auto mx383              = mm->add_instruction(convolution383, mx382, mx86);
    migraphx::op::batch_norm_inference batch_norm_inference384;
    batch_norm_inference384.epsilon  = 1e-05;
    batch_norm_inference384.momentum = 0.9;
    auto mx384 = mm->add_instruction(batch_norm_inference384, mx383, mx85, mx84, mx83, mx82);
    migraphx::op::add add385;
    auto mx385 = mm->add_instruction(add385, mx384, mx376);
    migraphx::op::relu relu386;
    auto mx386 = mm->add_instruction(relu386, mx385);
    migraphx::op::convolution convolution387;
    convolution387.padding  = {0, 0};
    convolution387.stride   = {1, 1};
    convolution387.dilation = {1, 1};
    convolution387.group    = 1;
    auto mx387              = mm->add_instruction(convolution387, mx386, mx81);
    migraphx::op::batch_norm_inference batch_norm_inference388;
    batch_norm_inference388.epsilon  = 1e-05;
    batch_norm_inference388.momentum = 0.9;
    auto mx388 = mm->add_instruction(batch_norm_inference388, mx387, mx80, mx79, mx78, mx77);
    migraphx::op::relu relu389;
    auto mx389 = mm->add_instruction(relu389, mx388);
    migraphx::op::convolution convolution390;
    convolution390.padding  = {1, 1};
    convolution390.stride   = {1, 1};
    convolution390.dilation = {1, 1};
    convolution390.group    = 1;
    auto mx390              = mm->add_instruction(convolution390, mx389, mx76);
    migraphx::op::batch_norm_inference batch_norm_inference391;
    batch_norm_inference391.epsilon  = 1e-05;
    batch_norm_inference391.momentum = 0.9;
    auto mx391 = mm->add_instruction(batch_norm_inference391, mx390, mx75, mx74, mx73, mx72);
    migraphx::op::relu relu392;
    auto mx392 = mm->add_instruction(relu392, mx391);
    migraphx::op::convolution convolution393;
    convolution393.padding  = {0, 0};
    convolution393.stride   = {1, 1};
    convolution393.dilation = {1, 1};
    convolution393.group    = 1;
    auto mx393              = mm->add_instruction(convolution393, mx392, mx71);
    migraphx::op::batch_norm_inference batch_norm_inference394;
    batch_norm_inference394.epsilon  = 1e-05;
    batch_norm_inference394.momentum = 0.9;
    auto mx394 = mm->add_instruction(batch_norm_inference394, mx393, mx70, mx69, mx68, mx67);
    migraphx::op::add add395;
    auto mx395 = mm->add_instruction(add395, mx394, mx386);
    migraphx::op::relu relu396;
    auto mx396 = mm->add_instruction(relu396, mx395);
    migraphx::op::convolution convolution397;
    convolution397.padding  = {0, 0};
    convolution397.stride   = {1, 1};
    convolution397.dilation = {1, 1};
    convolution397.group    = 1;
    auto mx397              = mm->add_instruction(convolution397, mx396, mx66);
    migraphx::op::batch_norm_inference batch_norm_inference398;
    batch_norm_inference398.epsilon  = 1e-05;
    batch_norm_inference398.momentum = 0.9;
    auto mx398 = mm->add_instruction(batch_norm_inference398, mx397, mx65, mx64, mx63, mx62);
    migraphx::op::relu relu399;
    auto mx399 = mm->add_instruction(relu399, mx398);
    migraphx::op::convolution convolution400;
    convolution400.padding  = {1, 1};
    convolution400.stride   = {1, 1};
    convolution400.dilation = {1, 1};
    convolution400.group    = 1;
    auto mx400              = mm->add_instruction(convolution400, mx399, mx61);
    migraphx::op::batch_norm_inference batch_norm_inference401;
    batch_norm_inference401.epsilon  = 1e-05;
    batch_norm_inference401.momentum = 0.9;
    auto mx401 = mm->add_instruction(batch_norm_inference401, mx400, mx60, mx59, mx58, mx57);
    migraphx::op::relu relu402;
    auto mx402 = mm->add_instruction(relu402, mx401);
    migraphx::op::convolution convolution403;
    convolution403.padding  = {0, 0};
    convolution403.stride   = {1, 1};
    convolution403.dilation = {1, 1};
    convolution403.group    = 1;
    auto mx403              = mm->add_instruction(convolution403, mx402, mx56);
    migraphx::op::batch_norm_inference batch_norm_inference404;
    batch_norm_inference404.epsilon  = 1e-05;
    batch_norm_inference404.momentum = 0.9;
    auto mx404 = mm->add_instruction(batch_norm_inference404, mx403, mx55, mx54, mx53, mx52);
    migraphx::op::add add405;
    auto mx405 = mm->add_instruction(add405, mx404, mx396);
    migraphx::op::relu relu406;
    auto mx406 = mm->add_instruction(relu406, mx405);
    migraphx::op::convolution convolution407;
    convolution407.padding  = {0, 0};
    convolution407.stride   = {1, 1};
    convolution407.dilation = {1, 1};
    convolution407.group    = 1;
    auto mx407              = mm->add_instruction(convolution407, mx406, mx51);
    migraphx::op::batch_norm_inference batch_norm_inference408;
    batch_norm_inference408.epsilon  = 1e-05;
    batch_norm_inference408.momentum = 0.9;
    auto mx408 = mm->add_instruction(batch_norm_inference408, mx407, mx50, mx49, mx48, mx47);
    migraphx::op::relu relu409;
    auto mx409 = mm->add_instruction(relu409, mx408);
    migraphx::op::convolution convolution410;
    convolution410.padding  = {1, 1};
    convolution410.stride   = {2, 2};
    convolution410.dilation = {1, 1};
    convolution410.group    = 1;
    auto mx410              = mm->add_instruction(convolution410, mx409, mx46);
    migraphx::op::batch_norm_inference batch_norm_inference411;
    batch_norm_inference411.epsilon  = 1e-05;
    batch_norm_inference411.momentum = 0.9;
    auto mx411 = mm->add_instruction(batch_norm_inference411, mx410, mx45, mx44, mx43, mx42);
    migraphx::op::relu relu412;
    auto mx412 = mm->add_instruction(relu412, mx411);
    migraphx::op::convolution convolution413;
    convolution413.padding  = {0, 0};
    convolution413.stride   = {1, 1};
    convolution413.dilation = {1, 1};
    convolution413.group    = 1;
    auto mx413              = mm->add_instruction(convolution413, mx412, mx41);
    migraphx::op::batch_norm_inference batch_norm_inference414;
    batch_norm_inference414.epsilon  = 1e-05;
    batch_norm_inference414.momentum = 0.9;
    auto mx414 = mm->add_instruction(batch_norm_inference414, mx413, mx40, mx39, mx38, mx37);
    migraphx::op::convolution convolution415;
    convolution415.padding  = {0, 0};
    convolution415.stride   = {2, 2};
    convolution415.dilation = {1, 1};
    convolution415.group    = 1;
    auto mx415              = mm->add_instruction(convolution415, mx406, mx36);
    migraphx::op::batch_norm_inference batch_norm_inference416;
    batch_norm_inference416.epsilon  = 1e-05;
    batch_norm_inference416.momentum = 0.9;
    auto mx416 = mm->add_instruction(batch_norm_inference416, mx415, mx35, mx34, mx33, mx32);
    migraphx::op::add add417;
    auto mx417 = mm->add_instruction(add417, mx414, mx416);
    migraphx::op::relu relu418;
    auto mx418 = mm->add_instruction(relu418, mx417);
    migraphx::op::convolution convolution419;
    convolution419.padding  = {0, 0};
    convolution419.stride   = {1, 1};
    convolution419.dilation = {1, 1};
    convolution419.group    = 1;
    auto mx419              = mm->add_instruction(convolution419, mx418, mx31);
    migraphx::op::batch_norm_inference batch_norm_inference420;
    batch_norm_inference420.epsilon  = 1e-05;
    batch_norm_inference420.momentum = 0.9;
    auto mx420 = mm->add_instruction(batch_norm_inference420, mx419, mx30, mx29, mx28, mx27);
    migraphx::op::relu relu421;
    auto mx421 = mm->add_instruction(relu421, mx420);
    migraphx::op::convolution convolution422;
    convolution422.padding  = {1, 1};
    convolution422.stride   = {1, 1};
    convolution422.dilation = {1, 1};
    convolution422.group    = 1;
    auto mx422              = mm->add_instruction(convolution422, mx421, mx26);
    migraphx::op::batch_norm_inference batch_norm_inference423;
    batch_norm_inference423.epsilon  = 1e-05;
    batch_norm_inference423.momentum = 0.9;
    auto mx423 = mm->add_instruction(batch_norm_inference423, mx422, mx25, mx24, mx23, mx22);
    migraphx::op::relu relu424;
    auto mx424 = mm->add_instruction(relu424, mx423);
    migraphx::op::convolution convolution425;
    convolution425.padding  = {0, 0};
    convolution425.stride   = {1, 1};
    convolution425.dilation = {1, 1};
    convolution425.group    = 1;
    auto mx425              = mm->add_instruction(convolution425, mx424, mx21);
    migraphx::op::batch_norm_inference batch_norm_inference426;
    batch_norm_inference426.epsilon  = 1e-05;
    batch_norm_inference426.momentum = 0.9;
    auto mx426 = mm->add_instruction(batch_norm_inference426, mx425, mx20, mx19, mx18, mx17);
    migraphx::op::add add427;
    auto mx427 = mm->add_instruction(add427, mx426, mx418);
    migraphx::op::relu relu428;
    auto mx428 = mm->add_instruction(relu428, mx427);
    migraphx::op::convolution convolution429;
    convolution429.padding  = {0, 0};
    convolution429.stride   = {1, 1};
    convolution429.dilation = {1, 1};
    convolution429.group    = 1;
    auto mx429              = mm->add_instruction(convolution429, mx428, mx16);
    migraphx::op::batch_norm_inference batch_norm_inference430;
    batch_norm_inference430.epsilon  = 1e-05;
    batch_norm_inference430.momentum = 0.9;
    auto mx430 = mm->add_instruction(batch_norm_inference430, mx429, mx15, mx14, mx13, mx12);
    migraphx::op::relu relu431;
    auto mx431 = mm->add_instruction(relu431, mx430);
    migraphx::op::convolution convolution432;
    convolution432.padding  = {1, 1};
    convolution432.stride   = {1, 1};
    convolution432.dilation = {1, 1};
    convolution432.group    = 1;
    auto mx432              = mm->add_instruction(convolution432, mx431, mx11);
    migraphx::op::batch_norm_inference batch_norm_inference433;
    batch_norm_inference433.epsilon  = 1e-05;
    batch_norm_inference433.momentum = 0.9;
    auto mx433 = mm->add_instruction(batch_norm_inference433, mx432, mx10, mx9, mx8, mx7);
    migraphx::op::relu relu434;
    auto mx434 = mm->add_instruction(relu434, mx433);
    migraphx::op::convolution convolution435;
    convolution435.padding  = {0, 0};
    convolution435.stride   = {1, 1};
    convolution435.dilation = {1, 1};
    convolution435.group    = 1;
    auto mx435              = mm->add_instruction(convolution435, mx434, mx6);
    migraphx::op::batch_norm_inference batch_norm_inference436;
    batch_norm_inference436.epsilon  = 1e-05;
    batch_norm_inference436.momentum = 0.9;
    auto mx436 = mm->add_instruction(batch_norm_inference436, mx435, mx5, mx4, mx3, mx2);
    migraphx::op::add add437;
    auto mx437 = mm->add_instruction(add437, mx436, mx428);
    migraphx::op::relu relu438;
    auto mx438 = mm->add_instruction(relu438, mx437);
    migraphx::op::pooling pooling439;
    pooling439.mode    = migraphx::op::pooling_mode::average;
    pooling439.padding = {0, 0};
    pooling439.stride  = {1, 1};
    pooling439.lengths = {7, 7};
    auto mx439         = mm->add_instruction(pooling439, mx438);
    migraphx::op::flatten flatten440;
    flatten440.axis = 1;
    auto mx440      = mm->add_instruction(flatten440, mx439);
    migraphx::op::transpose transpose441;
    transpose441.dims = {1, 0};
    auto mx441        = mm->add_instruction(transpose441, mx1);
    migraphx::op::multibroadcast multibroadcast442;
    multibroadcast442.output_lens = {batch, 1000};
    auto mx442                    = mm->add_instruction(multibroadcast442, mx0);
    float dot443_alpha            = 1;
    float dot443_beta             = 1;
    migraphx::add_apply_alpha_beta(
        *mm, {mx440, mx441, mx442}, migraphx::make_op("dot"), dot443_alpha, dot443_beta);
    return p;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx
