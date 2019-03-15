#include <iostream>
#include <vector>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/onnx.hpp>
#include "test.hpp"
#include <migraphx/half.hpp>


template <class T>
void matmul_test()
{
    migraphx::program p;
    std::vector<T> a     = {-0.00925222, 0.56250403, 0.70107397,  0.75402161,  -0.505885,
                        1.33628943,  -0.11413,   -0.31270559, 1.59336732,  -0.19361027,
                        -0.91620867, 0.40108416, -0.06969921, 0.68483471,  -0.39906632,
                        -1.66423624, 0.69040076, -1.31490171, -0.11282616, -0.79391814};
    std::vector<float> b = {6.09568541e-01,
                            -6.10527007e-01,
                            3.66646462e-01,
                            1.18951101e-01,
                            5.58777432e-01,
                            -3.21296298e-01,
                            -5.95997198e-01,
                            -5.01425721e-01,
                            -2.84606807e-01,
                            -5.73673557e-01,
                            -8.99430260e-01,
                            -4.25103093e-01,
                            1.53027987e+00,
                            -3.81407415e-04,
                            -3.29650255e-01};
    std::vector<float> c = {-1.56327541e+00,
                            -7.09570140e-01,
                            -5.37424982e-01,
                            -2.22994831e-01,
                            -2.15586437e+00,
                            2.09177941e-03,
                            -1.47279677e+00,
                            2.02627040e-01,
                            -6.04527691e-01,
                            -1.29885596e+00,
                            2.16294914e+00,
                            -1.48101497e-01};
    migraphx::shape a_shape{migraphx::shape::get_type<T>{}, {4, 5}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::get_type<T>{}, {5, 3}};
    auto bl = p.add_literal(migraphx::literal{b_shape, b});
    p.add_instruction(migraphx::op::dot{}, al, bl);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<T> results_vector(12);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(c, results_vector));
}
TEST_CASE_REGISTER(matmul_test<float>)
TEST_CASE_REGISTER(matmul_test<double>)

template <class T>
void matmul_test_ex()
{
    migraphx::program p;
    std::vector<T> a     = {-0.00925222, 0.56250403, 0.70107397,  0.75402161,  -0.505885,
                        1.33628943,  -0.11413,   -0.31270559, 1.59336732,  -0.19361027,
                        -0.91620867, 0.40108416, -0.06969921, 0.68483471,  -0.39906632,
                        -1.66423624, 0.69040076, -1.31490171, -0.11282616, -0.79391814};
    std::vector<float> b = {6.09568541e-01,
                            -6.10527007e-01,
                            3.66646462e-01,
                            1.18951101e-01,
                            5.58777432e-01,
                            -3.21296298e-01,
                            -5.95997198e-01,
                            -5.01425721e-01,
                            -2.84606807e-01,
                            -5.73673557e-01,
                            -8.99430260e-01,
                            -4.25103093e-01,
                            1.53027987e+00,
                            -3.81407415e-04,
                            -3.29650255e-01};
    std::vector<float> c = {-1.56327541e+00,
                            -7.09570140e-01,
                            -5.37424982e-01,
                            -2.22994831e-01,
                            -2.15586437e+00,
                            2.09177941e-03,
                            -1.47279677e+00,
                            2.02627040e-01,
                            -6.04527691e-01,
                            -1.29885596e+00,
                            2.16294914e+00,
                            -1.48101497e-01};
    migraphx::shape a_shape{migraphx::shape::get_type<T>{}, {1, 1, 4, 5}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::get_type<T>{}, {1, 1, 5, 3}};
    auto bl = p.add_literal(migraphx::literal{b_shape, b});
    p.add_instruction(migraphx::op::dot{}, al, bl);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<T> results_vector(12);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(c, results_vector));
}
TEST_CASE_REGISTER(matmul_test_ex<float>)
TEST_CASE_REGISTER(matmul_test_ex<double>)

TEST_CASE(matmul_mutli_dim_2)
{
    migraphx::program p;
    std::vector<float> m1 = {-0.76234141,
                             0.01368910,
                             -0.86343423,
                             -0.99465282,
                             0.76133268,
                             0.96507140,
                             -0.55893585,
                             0.02625652,
                             0.75171776,
                             0.23112578,
                             0.25624787,
                             -1.50442161};
    migraphx::shape m1_shape{migraphx::shape::float_type, {2, 2, 3}};
    std::vector<float> m2 = {-0.15933632, -0.69594712, -0.06198966, -1.23905184, -0.83672704,
                             -1.06971832, -0.12272917, 1.07094116,  -0.08346820, 1.16820693,
                             -0.95700874, 0.24059691,  0.43326023,  0.78305235,  -0.53506601,
                             -0.69359678, -0.26334436, 1.56292796,  -0.33629175, -1.72693469,
                             0.41435494,  1.52136843,  -0.40699791, -1.59839430};
    migraphx::shape m2_shape{migraphx::shape::float_type, {2, 3, 4}};
    auto l1 = p.add_literal(migraphx::literal{m1_shape, m1});
    auto l2 = p.add_literal(migraphx::literal{m2_shape, m2});

    p.add_instruction(migraphx::op::dot{}, l1, l2);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });

    std::vector<float> m_res = {0.18208394,
                                -0.49276402,
                                0.87189133,
                                0.75150114,
                                -0.55909610,
                                1.00521735,
                                -0.95536130,
                                2.27996211,
                                0.06239879,
                                0.74700068,
                                -0.01570983,
                                -0.85920856,
                                -0.59070835,
                                -1.70729902,
                                0.40245487,
                                1.80182751};

    EXPECT(migraphx::verify_range(m, m_res));
}

TEST_CASE(gemm_mutli_dim_2_beta0)
{
    migraphx::program p;
    std::vector<float> m1 = {-0.76234141,
                             0.01368910,
                             -0.86343423,
                             -0.99465282,
                             0.76133268,
                             0.96507140,
                             -0.55893585,
                             0.02625652,
                             0.75171776,
                             0.23112578,
                             0.25624787,
                             -1.50442161};
    migraphx::shape m1_shape{migraphx::shape::float_type, {2, 2, 3}};
    std::vector<float> m2 = {-0.15933632, -0.69594712, -0.06198966, -1.23905184, -0.83672704,
                             -1.06971832, -0.12272917, 1.07094116,  -0.08346820, 1.16820693,
                             -0.95700874, 0.24059691,  0.43326023,  0.78305235,  -0.53506601,
                             -0.69359678, -0.26334436, 1.56292796,  -0.33629175, -1.72693469,
                             0.41435494,  1.52136843,  -0.40699791, -1.59839430};
    migraphx::shape m2_shape{migraphx::shape::float_type, {2, 3, 4}};
    std::vector<float> m3 = {0.18208394,
                             -0.49276402,
                             0.87189133,
                             0.75150114,
                             -0.55909610,
                             1.00521735,
                             -0.95536130,
                             2.27996211,
                             0.06239879,
                             0.74700068,
                             -0.01570983,
                             -0.85920856,
                             -0.59070835,
                             -1.70729902,
                             0.40245487,
                             1.80182751};
    migraphx::shape m3_shape{migraphx::shape::float_type, {2, 2, 4}};
    auto l1     = p.add_literal(migraphx::literal{m1_shape, m1});
    auto l2     = p.add_literal(migraphx::literal{m2_shape, m2});
    auto l3     = p.add_literal(migraphx::literal{m3_shape, m3});
    float alpha = 1.0f;
    float beta  = 0.0f;
    p.add_instruction(migraphx::op::dot{alpha, beta}, l1, l2, l3);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });

    std::vector<float> m_res = {0.18208394,
                                -0.49276402,
                                0.87189133,
                                0.75150114,
                                -0.55909610,
                                1.00521735,
                                -0.95536130,
                                2.27996211,
                                0.06239879,
                                0.74700068,
                                -0.01570983,
                                -0.85920856,
                                -0.59070835,
                                -1.70729902,
                                0.40245487,
                                1.80182751};

    EXPECT(migraphx::verify_range(m, m_res));
}

TEST_CASE(gemm_beta_0)
{
    migraphx::program p;
    std::vector<float> m1 = {
        -0.76234141, 0.01368910, -0.86343423, -0.99465282, 0.76133268, 0.96507140};
    migraphx::shape m1_shape{migraphx::shape::float_type, {1, 2, 3}};
    std::vector<float> m2 = {-0.15933632,
                             -0.69594712,
                             -0.06198966,
                             -1.23905184,
                             -0.83672704,
                             -1.06971832,
                             -0.12272917,
                             1.07094116,
                             -0.08346820,
                             1.16820693,
                             -0.95700874,
                             0.24059691};
    migraphx::shape m2_shape{migraphx::shape::float_type, {1, 3, 4}};

    migraphx::shape m3_shape{migraphx::shape::float_type, {1, 2, 4}};
    std::vector<float> m3 = {0.18208394,
                             -0.49276402,
                             0.87189133,
                             0.75150114,
                             -0.55909610,
                             1.00521735,
                             -0.95536130,
                             2.27996211};
    auto l1               = p.add_literal(migraphx::literal{m1_shape, m1});
    auto l2               = p.add_literal(migraphx::literal{m2_shape, m2});
    auto l3               = p.add_literal(migraphx::literal{m3_shape, m3});

    float alpha = 1.0f;
    float beta  = 0.0f;
    p.add_instruction(migraphx::op::dot{alpha, beta}, l1, l2, l3);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });

    std::vector<float> m_res = {0.18208394,
                                -0.49276402,
                                0.87189133,
                                0.75150114,
                                -0.55909610,
                                1.00521735,
                                -0.95536130,
                                2.27996211};

    EXPECT(migraphx::verify_range(m, m_res));
}

TEST_CASE(matmul_mutli_dim_2_3)
{
    migraphx::program p;
    std::vector<float> m1 = {
        -1.93300070, 0.33902698,  -0.45173527, -0.72283069, -0.17177134, 1.62199882,
        0.87052847,  0.14989811,  -0.88969184, -0.18131398, 0.72654339,  -0.57123693,
        0.03852506,  -0.72332085, -1.81844083, -0.33465167, -0.71400352, 0.36883161,
        0.08698452,  0.94974586,  0.40087323,  -0.05448534, 0.03220677,  -1.22494296,
        0.97938472,  -1.43714454, -0.80430904, -0.08098728, 0.31520301,  0.49642169,
        -1.63471091, 0.34390096,  2.81292176,  -0.22666528, 1.54559556,  -1.51075762};
    migraphx::shape m1_shape{migraphx::shape::float_type, {2, 3, 2, 3}};
    std::vector<float> m2 = {
        -0.33170529, 2.26325120,  -0.50639461, 0.64802947,  0.44748888,  0.33768068,
        -0.53621075, 0.34341460,  0.58742520,  -1.13995790, -0.99322535, 0.35447353,
        0.01977110,  -0.10155016, -1.02288245, -0.16575791, -1.47870374, 0.29300008,
        -0.39112198, 1.42303608,  -0.02853060, 1.52610164,  0.53540909,  0.75618998,
        -0.26877787, -1.90886366, 0.30622790,  0.59794535,  1.29795331,  -0.37805803,
        -1.58167176, -1.26966832, 0.27435891,  0.89430347,  0.22854926,  -0.50317658};
    migraphx::shape m2_shape{migraphx::shape::float_type, {2, 3, 3, 2}};
    auto l1 = p.add_literal(migraphx::literal{m1_shape, m1});
    auto l2 = p.add_literal(migraphx::literal{m2_shape, m2});

    p.add_instruction(migraphx::op::dot{}, l1, l2);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });

    std::vector<float> m_res = {0.26735861,  -4.30770895, 1.05257728,  -1.19954265, 0.50493170,
                                -0.18729756, 1.09137941,  -1.09298312, 3.42956915,  -0.41681939,
                                0.17833257,  0.26040336,  0.15351280,  1.87632715,  -0.63545406,
                                -0.95467340, -1.74728628, -2.42477030, 0.76262372,  0.15539164,
                                3.32281958,  0.96769613,  0.43727545,  2.43019906};

    EXPECT(migraphx::verify_range(m, m_res));
}

TEST_CASE(gemm_mutli_dim1_2_3)
{
    migraphx::program p;
    std::vector<float> m1 = {
        1.23636469,  -0.47041261, -0.14375651, -0.48371852, 1.16479301,  -0.89361055,
        -0.18569086, 1.10700457,  -1.02632638, 0.82277012,  0.33525769,  0.52825145,
        -1.00141689, 0.45510090,  -0.02675039, -0.60454439, 0.38551153,  -0.01658514,
        0.93059292,  -0.54595188, -0.04911005, -0.91397221, -0.83127477, -1.57685603,
        -1.36200452, 2.25822236,  -1.23416970, 0.12312496,  0.76232760,  -0.83594234,
        1.67418145,  -0.19412936, 1.05261378,  0.66246074,  -1.15233398, 0.16429736};
    migraphx::shape m1_shape{migraphx::shape::float_type, {2, 3, 2, 3}};
    std::vector<float> m2 = {
        -0.87300530, -0.07112838, 0.19196860,  -1.04986840, 1.20348200,  0.31966893,
        1.04805440,  -2.04777729, -0.67906052, -1.17250760, 0.34305044,  -1.01957785,
        -1.12694862, 0.18431338,  -1.63712290, 0.27566931,  -1.11282021, 1.41738919,
        0.47871283,  -1.01980420, 1.00212436,  -0.78740444, -1.65636133, 1.51466547,
        -0.12470397, 0.70404393,  -0.15244797, 0.74288871,  0.07339926,  -1.45811623,
        0.27185845,  0.08804596,  0.99061977,  -1.61752428, 0.29191159,  0.87271953};
    migraphx::shape m2_shape{migraphx::shape::float_type, {2, 3, 3, 2}};
    std::vector<float> m3 = {-1.07692443, 0.85223457,  -0.37266530, 2.31511577,  0.04227017,
                             1.13229428,  -0.52769242, 0.27307182,  -0.47779843, -0.08023168,
                             -0.22862823, 0.81489871,  1.13139581,  1.13860467,  0.24309065,
                             0.26533729,  0.49106772,  -1.18860493, 0.27842449,  1.03568141,
                             0.49759611,  0.10021662,  0.00592602,  0.90862000};
    migraphx::shape m3_shape{migraphx::shape::float_type, {2, 3, 2, 2}};

    auto l1        = p.add_literal(migraphx::literal{m1_shape, m1});
    auto l2        = p.add_literal(migraphx::literal{m2_shape, m2});
    auto l3        = p.add_literal(migraphx::literal{m3_shape, m3});
    float alpha    = 0.35;
    float beta     = 0.41;
    auto m12_alpha = p.add_instruction(migraphx::op::dot{alpha, beta}, l1, l2);
    auto l_beta    = p.add_literal(beta);
    auto b_beta    = p.add_instruction(migraphx::op::scalar{m12_alpha->get_shape()}, l_beta);
    auto m3_beta   = p.add_instruction(migraphx::op::mul{}, b_beta, l3);
    p.add_instruction(migraphx::op::add{}, m3_beta, m12_alpha);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });

    std::vector<float> m_res = {-0.91147203, 0.47540785, -0.30313587, 0.43325099,  -0.43711586,
                                0.50928632,  0.06919868, -0.80382802, -0.05125718, -0.06685650,
                                -0.06972163, 0.32407764, 0.45677396,  0.25909489,  0.56911252,
                                -0.17183724, 0.10858734, 0.39406289,  0.04662959,  1.07979824,
                                0.40355016,  0.52410648, -0.31728447, 1.09550845};

    EXPECT(migraphx::verify_range(m, m_res));
}

TEST_CASE(gemm_mutli_3args)
{
    migraphx::program p;
    std::vector<float> m1 = {
        1.23636469,  -0.47041261, -0.14375651, -0.48371852, 1.16479301,  -0.89361055,
        -0.18569086, 1.10700457,  -1.02632638, 0.82277012,  0.33525769,  0.52825145,
        -1.00141689, 0.45510090,  -0.02675039, -0.60454439, 0.38551153,  -0.01658514,
        0.93059292,  -0.54595188, -0.04911005, -0.91397221, -0.83127477, -1.57685603,
        -1.36200452, 2.25822236,  -1.23416970, 0.12312496,  0.76232760,  -0.83594234,
        1.67418145,  -0.19412936, 1.05261378,  0.66246074,  -1.15233398, 0.16429736};
    migraphx::shape m1_shape{migraphx::shape::float_type, {2, 3, 2, 3}};
    std::vector<float> m2 = {
        -0.87300530, -0.07112838, 0.19196860,  -1.04986840, 1.20348200,  0.31966893,
        1.04805440,  -2.04777729, -0.67906052, -1.17250760, 0.34305044,  -1.01957785,
        -1.12694862, 0.18431338,  -1.63712290, 0.27566931,  -1.11282021, 1.41738919,
        0.47871283,  -1.01980420, 1.00212436,  -0.78740444, -1.65636133, 1.51466547,
        -0.12470397, 0.70404393,  -0.15244797, 0.74288871,  0.07339926,  -1.45811623,
        0.27185845,  0.08804596,  0.99061977,  -1.61752428, 0.29191159,  0.87271953};
    migraphx::shape m2_shape{migraphx::shape::float_type, {2, 3, 3, 2}};
    std::vector<float> m3 = {-1.07692443, 0.85223457,  -0.37266530, 2.31511577,  0.04227017,
                             1.13229428,  -0.52769242, 0.27307182,  -0.47779843, -0.08023168,
                             -0.22862823, 0.81489871,  1.13139581,  1.13860467,  0.24309065,
                             0.26533729,  0.49106772,  -1.18860493, 0.27842449,  1.03568141,
                             0.49759611,  0.10021662,  0.00592602,  0.90862000};
    migraphx::shape m3_shape{migraphx::shape::float_type, {2, 3, 2, 2}};

    auto l1     = p.add_literal(migraphx::literal{m1_shape, m1});
    auto l2     = p.add_literal(migraphx::literal{m2_shape, m2});
    auto l3     = p.add_literal(migraphx::literal{m3_shape, m3});
    float alpha = 0.35;
    float beta  = 0.41;
    p.add_instruction(migraphx::op::dot{alpha, beta}, l1, l2, l3);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> m;
    result.visit([&](auto output) { m.assign(output.begin(), output.end()); });

    std::vector<float> m_res = {-0.91147203, 0.47540785, -0.30313587, 0.43325099,  -0.43711586,
                                0.50928632,  0.06919868, -0.80382802, -0.05125718, -0.06685650,
                                -0.06972163, 0.32407764, 0.45677396,  0.25909489,  0.56911252,
                                -0.17183724, 0.10858734, 0.39406289,  0.04662959,  1.07979824,
                                0.40355016,  0.52410648, -0.31728447, 1.09550845};

    EXPECT(migraphx::verify_range(m, m_res));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
