/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/op/pooling.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

TEST_CASE(conv_dyn_batch_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape input_dyn_shape{migraphx::shape::float_type,
                                    {{1, 100}, {3, 3}, {4, 4}, {4, 4}}};
    migraphx::shape weights_shape{migraphx::shape::float_type, {2, 3, 3, 3}};

    auto input   = mm->add_parameter("X", input_dyn_shape);
    auto weights = mm->add_parameter("W", weights_shape);
    mm->add_instruction(migraphx::make_op("convolution",
                                          {
                                              {"padding", {1, 1}},
                                              {"stride", {2, 2}},
                                          }),
                        input,
                        weights);

    p.compile(migraphx::make_target("ref"));

    std::vector<float> a = {
        2.71567607,  -0.9960829,  0.91671127,  0.28140706,  0.63235772,  0.08077253,  0.80927712,
        -0.59108931, -1.05421555, -2.76622486, -0.85044265, -0.52049929, 0.67726439,  -0.65290606,
        0.02345525,  -0.33579525, 0.38901961,  1.05473483,  -1.31188095, 1.8963089,   -0.07265259,
        0.947339,    0.41949373,  -0.70814759, 0.25892952,  1.07311416,  1.2571274,   -0.62318051,
        -0.19951548, -0.94232577, -0.29393643, 0.42292568,  -0.80230367, 1.40909171,  0.63617158,
        0.13900366,  1.09253144,  -0.15265895, 1.54781747,  0.72780299,  1.09189606,  -0.38068101,
        0.97057933,  -0.58958799, 1.56188643,  0.21474874,  0.58725154,  -1.27097559, -0.03024297,
        1.09437096,  -0.4897908,  0.34838957,  -1.31042492, -1.69069934, 0.86956722,  -0.40457946,
        0.46691212,  1.29273605,  0.26464137,  0.22073045,  -1.02178168, 0.22163901,  -1.84387338,
        0.75522131,  -0.45775682, -0.42241111, -1.50944722, 1.07256448,  -1.95876884, -0.28106022,
        0.3341668,   2.13129425,  -1.14728117, -1.06555498, -0.298444,   -0.88322699, -0.65866792,
        -2.06007552, 0.01374334,  0.45612028,  0.52715492,  1.01914406,  -1.72659791, 0.80650896,
        0.16860051,  2.24112225,  -0.78620857, 0.36566174,  -0.07020134, -0.47976932, -0.68230027,
        -0.94711417, -0.54506505, 1.66504931,  -0.71860826, 0.61132306};

    std::vector<float> c = {
        -0.14601797, -0.13000923, 0.06521662,  0.06178288,  -0.11083675, 0.10154136,  0.09990512,
        0.06030385,  -0.11374587, -0.17523311, -0.14344215, 0.17802463,  0.06300922,  -0.15325832,
        0.07066704,  0.05166031,  0.00615084,  -0.02606523, 0.08083995,  -0.17913306, 0.0624622,
        0.0735731,   -0.04198661, -0.0164391,  -0.06374192, 0.16569914,  0.10681538,  0.07370754,
        0.02802075,  0.00282027,  0.15104802,  -0.11084409, -0.00197773, 0.07924436,  0.03528272,
        0.04765259,  -0.15896152, 0.07917164,  0.12125669,  -0.1154705,  -0.11999125, 0.12749968,
        -0.06269585, 0.18658121,  -0.03944227, 0.0111798,   -0.17731084, 0.11789055,  -0.09982193,
        0.08142821,  0.0729029,   0.11303909,  0.12735154,  0.03885292};

    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {2, 3, 4, 4}};

    migraphx::parameter_map params0;
    params0["X"] = migraphx::argument(input_fixed_shape0, a.data());
    params0["W"] = migraphx::argument(weights_shape, c.data());

    auto result = p.eval(params0).back();
    std::vector<float> results_vector(64);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-0.20817225,
                               0.87965256,
                               0.14958936,
                               -1.24887264,
                               -0.06540672,
                               0.20778663,
                               0.40456355,
                               -0.99900877,
                               0.4917807,
                               0.1994698,
                               0.64205718,
                               0.37798831,
                               -0.25315839,
                               0.44276932,
                               -0.16138598,
                               0.79344082};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(conv_dyn_img_shape_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape input_dyn_shape{migraphx::shape::float_type, {{1, 1}, {3, 3}, {4, 6}, {4, 6}}};
    migraphx::shape weights_shape{migraphx::shape::float_type, {1, 3, 3, 3}};

    auto input   = mm->add_parameter("X", input_dyn_shape);
    auto weights = mm->add_parameter("W", weights_shape);
    mm->add_instruction(migraphx::make_op("convolution", {{"padding", {0, 0}}, {"stride", {1, 1}}}),
                        input,
                        weights);

    p.compile(migraphx::make_target("ref"));

    std::vector<float> a = {0.28007596, 0.46114671, 0.12171969, 0.52260835, 0.40916841, 0.07163955,
                            0.09896668, 0.98628836, 0.69406788, 0.44868846, 0.64017681, 0.27048886,
                            0.30187397, 0.07334207, 0.05258557, 0.80747513, 0.81330534, 0.00497161,
                            0.33005534, 0.08908686, 0.46794691, 0.61768946, 0.55104806, 0.13406187,
                            0.70244284, 0.61296941, 0.46742536, 0.29712714, 0.91839388, 0.0834397,
                            0.14476327, 0.37857075, 0.25922384, 0.61620963, 0.69455439, 0.70389431,
                            0.77388606, 0.1752363,  0.74631394, 0.24604889, 0.53600244, 0.22116457,
                            0.81217463, 0.10789447, 0.43083784, 0.63371852, 0.69742316, 0.09536905};

    std::vector<float> c = {0.98411968, 0.2899219,  0.44638833, 0.30390816, 0.03989896, 0.2445332,
                            0.32700131, 0.57517075, 0.06956476, 0.93079306, 0.19882314, 0.52940601,
                            0.35624753, 0.35938406, 0.9111428,  0.88923574, 0.61040283, 0.2797513,
                            0.15479768, 0.46534674, 0.16970931, 0.49704618, 0.07062198, 0.01678321,
                            0.53150934, 0.39244495, 0.9963813};

    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {1, 3, 4, 4}};

    migraphx::parameter_map params0;
    params0["X"] = migraphx::argument(input_fixed_shape0, a.data());
    params0["W"] = migraphx::argument(weights_shape, c.data());

    auto result = p.eval(params0).back();
    std::vector<float> results_vector(72);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {6.1329393, 4.3199925, 5.448438, 3.8497565};

    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));

    a = {0.95600171, 0.20768181, 0.82844489, 0.14928212, 0.51280462, 0.1359196,  0.68903648,
         0.84174772, 0.425509,   0.956926,   0.82533291, 0.33821531, 0.57576055, 0.75330186,
         0.82710394, 0.93343847, 0.14499469, 0.74558021, 0.13935139, 0.90652876, 0.22611443,
         0.85323975, 0.30631787, 0.96983037, 0.51783421, 0.32247456, 0.28243352, 0.605865,
         0.33376446, 0.67864877, 0.15442507, 0.24977552, 0.86989425, 0.60036782, 0.26198306,
         0.1494149,  0.13678915, 0.24892094, 0.38282467, 0.64907906, 0.83756376, 0.77603195,
         0.33951558, 0.14856874, 0.45701939, 0.43786436, 0.57421759, 0.37326922, 0.63382506,
         0.11464436, 0.23309047, 0.76724102, 0.98712427, 0.80800108, 0.84296564, 0.79568268,
         0.45684131, 0.73867068, 0.57845499, 0.45073557, 0.27102442, 0.86460315, 0.06865567,
         0.81673446, 0.881835,   0.42351639, 0.83322931, 0.34101671, 0.51979151, 0.54920645,
         0.19287718, 0.33321689, 0.27752456, 0.45755893, 0.67484562, 0.68383122, 0.52361312,
         0.46437257, 0.50862936, 0.32460429, 0.1726007,  0.29933345, 0.64856728, 0.06471591,
         0.63370843, 0.27900152, 0.18595992, 0.48904812, 0.35368508, 0.09620202};

    c = {0.709561,   0.7916206,  0.0443115,  0.62592275, 0.2498623,  0.42725624, 0.7905135,
         0.53160169, 0.01303743, 0.01987505, 0.39041803, 0.89530203, 0.23155373, 0.44435213,
         0.14407301, 0.80968594, 0.38216188, 0.35692557, 0.2568538,  0.83587388, 0.43654904,
         0.04974508, 0.80375029, 0.25350374, 0.1820275,  0.23369029, 0.54358755};

    gold = {6.305986,
            5.564665,
            6.122996,
            5.7262855,
            5.5546584,
            5.779489,
            5.798161,
            5.160476,
            6.702436,
            5.4851074,
            6.227567,
            5.2016754};
    migraphx::shape input_fixed_shape1{migraphx::shape::float_type, {1, 3, 6, 5}};

    migraphx::parameter_map params1;
    params1["X"] = migraphx::argument(input_fixed_shape1, a.data());
    params1["W"] = migraphx::argument(weights_shape, c.data());

    result = p.eval(params1).back();
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(conv_dyn_weights_shape_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape input_shape{migraphx::shape::float_type, {1, 3, 4, 4}};
    migraphx::shape weights_shape{migraphx::shape::float_type, {{1, 1}, {3, 3}, {2, 3}, {2, 3}}};

    auto input   = mm->add_parameter("X", input_shape);
    auto weights = mm->add_parameter("W", weights_shape);
    mm->add_instruction(migraphx::make_op("convolution", {{"padding", {0, 0}}, {"stride", {1, 1}}}),
                        input,
                        weights);

    p.compile(migraphx::make_target("ref"));

    std::vector<float> a = {0.28007596, 0.46114671, 0.12171969, 0.52260835, 0.40916841, 0.07163955,
                            0.09896668, 0.98628836, 0.69406788, 0.44868846, 0.64017681, 0.27048886,
                            0.30187397, 0.07334207, 0.05258557, 0.80747513, 0.81330534, 0.00497161,
                            0.33005534, 0.08908686, 0.46794691, 0.61768946, 0.55104806, 0.13406187,
                            0.70244284, 0.61296941, 0.46742536, 0.29712714, 0.91839388, 0.0834397,
                            0.14476327, 0.37857075, 0.25922384, 0.61620963, 0.69455439, 0.70389431,
                            0.77388606, 0.1752363,  0.74631394, 0.24604889, 0.53600244, 0.22116457,
                            0.81217463, 0.10789447, 0.43083784, 0.63371852, 0.69742316, 0.09536905};

    std::vector<float> c = {0.98411968,
                            0.2899219,
                            0.44638833,
                            0.30390816,
                            0.03989896,
                            0.2445332,
                            0.32700131,
                            0.57517075,
                            0.06956476,
                            0.93079306,
                            0.19882314,
                            0.52940601};

    migraphx::shape weight_fixed_shape0{migraphx::shape::float_type, {1, 3, 2, 2}};

    migraphx::parameter_map params0;
    params0["X"] = migraphx::argument(input_shape, a.data());
    params0["W"] = migraphx::argument(weight_fixed_shape0, c.data());

    auto result = p.eval(params0).back();
    std::vector<float> results_vector(72);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.9939406,
                               2.2703054,
                               1.8896171,
                               2.062202,
                               2.3035214,
                               1.629366,
                               2.1606991,
                               2.1917608,
                               1.6797699};

    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));

    c    = {0.98411968, 0.2899219,  0.44638833, 0.30390816, 0.03989896, 0.2445332,  0.32700131,
            0.57517075, 0.06956476, 0.93079306, 0.19882314, 0.52940601, 0.35624753, 0.35938406,
            0.9111428,  0.88923574, 0.61040283, 0.2797513,  0.15479768, 0.46534674, 0.16970931,
            0.49704618, 0.07062198, 0.01678321, 0.53150934, 0.39244495, 0.9963813};
    gold = {6.1329393, 4.3199925, 5.448438, 3.8497565};
    migraphx::shape weights_fixed_shape1{migraphx::shape::float_type, {1, 3, 3, 3}};

    migraphx::parameter_map params1;
    params1["X"] = migraphx::argument(input_shape, a.data());
    params1["W"] = migraphx::argument(weights_fixed_shape1, c.data());

    result = p.eval(params1).back();
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(conv_dyn_img_same_upper_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape input_dyn_shape{migraphx::shape::float_type, {{1, 1}, {3, 3}, {4, 6}, {4, 6}}};
    migraphx::shape weights_shape{migraphx::shape::float_type, {1, 3, 3, 3}};

    auto input   = mm->add_parameter("X", input_dyn_shape);
    auto weights = mm->add_parameter("W", weights_shape);
    mm->add_instruction(
        migraphx::make_op(
            "convolution",
            {{"stride", {1, 1}}, {"padding_mode", migraphx::op::padding_mode_t::same_upper}}),
        input,
        weights);

    p.compile(migraphx::make_target("ref"));

    std::vector<float> a = {0.63321185, 0.6466339,  0.8515352,  0.44240063, 0.5018913,  0.5068494,
                            0.75330657, 0.7383877,  0.15870683, 0.8171611,  0.56118083, 0.87004256,
                            0.24401724, 0.8815178,  0.4222333,  0.27191755,

                            0.41633207, 0.2460619,  0.32004243, 0.6962248,  0.12284133, 0.2620491,
                            0.96931046, 0.6030955,  0.7623861,  0.2395751,  0.61440414, 0.577285,
                            0.80087787, 0.12776066, 0.26566318, 0.46569306,

                            0.96701574, 0.3850145,  0.14165345, 0.5887347,  0.7152134,  0.5295342,
                            0.6303507,  0.4037548,  0.18556239, 0.79416305, 0.29107493, 0.18770285,
                            0.6870904,  0.30701008, 0.314684,   0.91075855};

    std::vector<float> c = {
        2.8150102e-01, 3.3198616e-01, 9.5149356e-01, 7.4039467e-02, 9.6555042e-01,
        2.8815505e-01, 2.5100240e-01, 5.2186239e-01, 2.3850012e-01,

        8.2963020e-01, 3.0763101e-04, 6.7026985e-01, 1.4260857e-01, 9.7517288e-01,
        3.6847427e-02, 8.5804445e-01, 7.3440993e-01, 6.7948365e-01,

        7.9253986e-02, 7.3943835e-01, 1.7813577e-01, 1.0780835e-01, 4.2304707e-01,
        4.0084350e-01, 1.1114500e-01, 4.4846520e-01, 5.0109702e-01};

    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {1, 3, 4, 4}};

    migraphx::parameter_map params0;
    params0["X"] = migraphx::argument(input_fixed_shape0, a.data());
    params0["W"] = migraphx::argument(weights_shape, c.data());

    auto result = p.eval(params0).back();
    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {3.013387,
                               3.7111127,
                               4.2946506,
                               3.579301,
                               4.5306826,
                               6.1262493,
                               6.332169,
                               4.495293,
                               4.46013,
                               6.0938954,
                               5.848162,
                               4.514299,
                               2.9587686,
                               4.117671,
                               3.5187216,
                               2.3236327};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(conv_dyn_kernel_same_upper_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape input_shape{migraphx::shape::float_type, {1, 3, 4, 4}};
    migraphx::shape weights_shape{migraphx::shape::float_type, {{1, 1}, {3, 3}, {2, 3}, {2, 3}}};

    auto input   = mm->add_parameter("X", input_shape);
    auto weights = mm->add_parameter("W", weights_shape);
    mm->add_instruction(
        migraphx::make_op(
            "convolution",
            {{"stride", {1, 1}}, {"padding_mode", migraphx::op::padding_mode_t::same_upper}}),
        input,
        weights);

    p.compile(migraphx::make_target("ref"));

    std::vector<float> a = {0.63321185, 0.6466339,  0.8515352,  0.44240063, 0.5018913,  0.5068494,
                            0.75330657, 0.7383877,  0.15870683, 0.8171611,  0.56118083, 0.87004256,
                            0.24401724, 0.8815178,  0.4222333,  0.27191755,

                            0.41633207, 0.2460619,  0.32004243, 0.6962248,  0.12284133, 0.2620491,
                            0.96931046, 0.6030955,  0.7623861,  0.2395751,  0.61440414, 0.577285,
                            0.80087787, 0.12776066, 0.26566318, 0.46569306,

                            0.96701574, 0.3850145,  0.14165345, 0.5887347,  0.7152134,  0.5295342,
                            0.6303507,  0.4037548,  0.18556239, 0.79416305, 0.29107493, 0.18770285,
                            0.6870904,  0.30701008, 0.314684,   0.91075855};
    std::vector<float> c = {2.8150102e-01,
                            3.3198616e-01,
                            9.5149356e-01,
                            7.4039467e-02,

                            9.6555042e-01,
                            2.8815505e-01,
                            2.5100240e-01,
                            5.2186239e-01,

                            2.3850012e-01,
                            8.2963020e-01,
                            3.0763101e-04,
                            6.7026985e-01};

    migraphx::shape weight_fixed_shape0{migraphx::shape::float_type, {1, 3, 2, 2}};

    migraphx::parameter_map params0;
    params0["X"] = migraphx::argument(input_shape, a.data());
    params0["W"] = migraphx::argument(weight_fixed_shape0, c.data());

    auto result = p.eval(params0).back();
    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {2.453681,
                               2.536207,
                               3.0187201,
                               1.7912633,
                               2.1738236,
                               2.9695358,
                               3.2319589,
                               1.859269,
                               2.5953722,
                               2.50734,
                               2.7736917,
                               1.2229807,
                               1.5900216,
                               0.9225286,
                               1.43048,
                               0.74341124};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(conv_dyn_kernel_same_lower_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape input_shape{migraphx::shape::float_type, {1, 3, 4, 4}};
    migraphx::shape weights_shape{migraphx::shape::float_type, {{1, 1}, {3, 3}, {2, 3}, {2, 3}}};

    auto input   = mm->add_parameter("X", input_shape);
    auto weights = mm->add_parameter("W", weights_shape);
    mm->add_instruction(
        migraphx::make_op(
            "convolution",
            {{"stride", {1, 1}}, {"padding_mode", migraphx::op::padding_mode_t::same_lower}}),
        input,
        weights);

    p.compile(migraphx::make_target("ref"));

    std::vector<float> a = {0.63321185, 0.6466339,  0.8515352,  0.44240063, 0.5018913,  0.5068494,
                            0.75330657, 0.7383877,  0.15870683, 0.8171611,  0.56118083, 0.87004256,
                            0.24401724, 0.8815178,  0.4222333,  0.27191755,

                            0.41633207, 0.2460619,  0.32004243, 0.6962248,  0.12284133, 0.2620491,
                            0.96931046, 0.6030955,  0.7623861,  0.2395751,  0.61440414, 0.577285,
                            0.80087787, 0.12776066, 0.26566318, 0.46569306,

                            0.96701574, 0.3850145,  0.14165345, 0.5887347,  0.7152134,  0.5295342,
                            0.6303507,  0.4037548,  0.18556239, 0.79416305, 0.29107493, 0.18770285,
                            0.6870904,  0.30701008, 0.314684,   0.91075855};
    std::vector<float> c = {2.8150102e-01,
                            3.3198616e-01,
                            9.5149356e-01,
                            7.4039467e-02,

                            9.6555042e-01,
                            2.8815505e-01,
                            2.5100240e-01,
                            5.2186239e-01,

                            2.3850012e-01,
                            8.2963020e-01,
                            3.0763101e-04,
                            6.7026985e-01};

    migraphx::shape weight_fixed_shape0{migraphx::shape::float_type, {1, 3, 2, 2}};

    migraphx::parameter_map params0;
    params0["X"] = migraphx::argument(input_shape, a.data());
    params0["W"] = migraphx::argument(weight_fixed_shape0, c.data());

    auto result = p.eval(params0).back();
    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0.91231215,
                               1.1416453,
                               1.00216,
                               1.6813052,
                               1.7131033,
                               2.453681,
                               2.536207,
                               3.0187201,
                               1.3293691,
                               2.1738236,
                               2.9695358,
                               3.2319589,
                               1.3228729,
                               2.5953722,
                               2.50734,
                               2.7736917};

    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(conv2d_padding_stride_test)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        2.71567607,  -0.9960829,  0.91671127,  0.28140706,  0.63235772,  0.08077253,  0.80927712,
        -0.59108931, -1.05421555, -2.76622486, -0.85044265, -0.52049929, 0.67726439,  -0.65290606,
        0.02345525,  -0.33579525, 0.38901961,  1.05473483,  -1.31188095, 1.8963089,   -0.07265259,
        0.947339,    0.41949373,  -0.70814759, 0.25892952,  1.07311416,  1.2571274,   -0.62318051,
        -0.19951548, -0.94232577, -0.29393643, 0.42292568,  -0.80230367, 1.40909171,  0.63617158,
        0.13900366,  1.09253144,  -0.15265895, 1.54781747,  0.72780299,  1.09189606,  -0.38068101,
        0.97057933,  -0.58958799, 1.56188643,  0.21474874,  0.58725154,  -1.27097559, -0.03024297,
        1.09437096,  -0.4897908,  0.34838957,  -1.31042492, -1.69069934, 0.86956722,  -0.40457946,
        0.46691212,  1.29273605,  0.26464137,  0.22073045,  -1.02178168, 0.22163901,  -1.84387338,
        0.75522131,  -0.45775682, -0.42241111, -1.50944722, 1.07256448,  -1.95876884, -0.28106022,
        0.3341668,   2.13129425,  -1.14728117, -1.06555498, -0.298444,   -0.88322699, -0.65866792,
        -2.06007552, 0.01374334,  0.45612028,  0.52715492,  1.01914406,  -1.72659791, 0.80650896,
        0.16860051,  2.24112225,  -0.78620857, 0.36566174,  -0.07020134, -0.47976932, -0.68230027,
        -0.94711417, -0.54506505, 1.66504931,  -0.71860826, 0.61132306};

    std::vector<float> c = {
        -0.14601797, -0.13000923, 0.06521662,  0.06178288,  -0.11083675, 0.10154136,  0.09990512,
        0.06030385,  -0.11374587, -0.17523311, -0.14344215, 0.17802463,  0.06300922,  -0.15325832,
        0.07066704,  0.05166031,  0.00615084,  -0.02606523, 0.08083995,  -0.17913306, 0.0624622,
        0.0735731,   -0.04198661, -0.0164391,  -0.06374192, 0.16569914,  0.10681538,  0.07370754,
        0.02802075,  0.00282027,  0.15104802,  -0.11084409, -0.00197773, 0.07924436,  0.03528272,
        0.04765259,  -0.15896152, 0.07917164,  0.12125669,  -0.1154705,  -0.11999125, 0.12749968,
        -0.06269585, 0.18658121,  -0.03944227, 0.0111798,   -0.17731084, 0.11789055,  -0.09982193,
        0.08142821,  0.0729029,   0.11303909,  0.12735154,  0.03885292};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 4, 4}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});

    migraphx::shape c_shape{migraphx::shape::float_type, {2, 3, 3, 3}};
    auto cl = mm->add_literal(migraphx::literal{c_shape, c});

    mm->add_instruction(
        migraphx::make_op("convolution", {{"padding", {1, 1}}, {"stride", {2, 2}}}), al, cl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-0.20817225,
                               0.87965256,
                               0.14958936,
                               -1.24887264,
                               -0.06540672,
                               0.20778663,
                               0.40456355,
                               -0.99900877,
                               0.4917807,
                               0.1994698,
                               0.64205718,
                               0.37798831,
                               -0.25315839,
                               0.44276932,
                               -0.16138598,
                               0.79344082};

    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(conv2d_padding_test)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        2.71567607,  -0.9960829,  0.91671127,  0.28140706,  0.63235772,  0.08077253,  0.80927712,
        -0.59108931, -1.05421555, -2.76622486, -0.85044265, -0.52049929, 0.67726439,  -0.65290606,
        0.02345525,  -0.33579525, 0.38901961,  1.05473483,  -1.31188095, 1.8963089,   -0.07265259,
        0.947339,    0.41949373,  -0.70814759, 0.25892952,  1.07311416,  1.2571274,   -0.62318051,
        -0.19951548, -0.94232577, -0.29393643, 0.42292568,  -0.80230367, 1.40909171,  0.63617158,
        0.13900366,  1.09253144,  -0.15265895, 1.54781747,  0.72780299,  1.09189606,  -0.38068101,
        0.97057933,  -0.58958799, 1.56188643,  0.21474874,  0.58725154,  -1.27097559, -0.03024297,
        1.09437096,  -0.4897908,  0.34838957,  -1.31042492, -1.69069934, 0.86956722,  -0.40457946,
        0.46691212,  1.29273605,  0.26464137,  0.22073045,  -1.02178168, 0.22163901,  -1.84387338,
        0.75522131,  -0.45775682, -0.42241111, -1.50944722, 1.07256448,  -1.95876884, -0.28106022,
        0.3341668,   2.13129425,  -1.14728117, -1.06555498, -0.298444,   -0.88322699, -0.65866792,
        -2.06007552, 0.01374334,  0.45612028,  0.52715492,  1.01914406,  -1.72659791, 0.80650896,
        0.16860051,  2.24112225,  -0.78620857, 0.36566174,  -0.07020134, -0.47976932, -0.68230027,
        -0.94711417, -0.54506505, 1.66504931,  -0.71860826, 0.61132306};

    std::vector<float> c = {
        -0.16115488, -0.09800646, -0.05412646, 0.10475694,  0.00555485,  -0.12667653, 0.0458357,
        -0.02656217, -0.16338061, 0.15037455,  0.0102711,   0.01303349,  0.05242859,  0.02034754,
        0.04751867,  -0.17038961, -0.1434752,  -0.10770349, 0.05676742,  -0.15838449, 0.10128359,
        -0.18958683, 0.11954515,  0.10758857,  -0.01058291, -0.12797487, 0.08971019,  0.18793164,
        -0.00881396, -0.06588994, -0.13321903, -0.03300409, 0.01439607,  0.07618178,  -0.11556662,
        0.00764295,  0.12956454,  -0.08937147, -0.12763587, 0.04674943,  0.05765297,  0.11336918,
        0.14747436,  -0.06199479, -0.01166052, -0.12432006, -0.04494537, -0.17581205, 0.09475745,
        0.1149437,   -0.1014564,  0.0274073,   -0.01323579, -0.11092556};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 4, 4}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});

    migraphx::shape c_shape{migraphx::shape::float_type, {2, 3, 3, 3}};
    auto cl = mm->add_literal(migraphx::literal{c_shape, c});

    mm->add_instruction(
        migraphx::make_op("convolution", {{"padding", {1, 1}}, {"stride", {1, 1}}}), al, cl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector(64);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {
        -0.0201216,  0.40407312,  -0.39005592, -0.0631946,  0.37963012,  -0.64611685, 0.1349397,
        -0.54113752, 0.28533003,  0.27667275,  -0.16442731, -0.181494,   0.30564839,  0.58744538,
        0.32015014,  0.24969585,  -0.27367792, -0.53308117, 0.41236052,  0.26136363,  -0.01489828,
        0.57652152,  -0.38506854, 0.119615,    0.0437076,   0.04779706,  0.57887721,  0.23126155,
        0.05695833,  -0.68200272, 0.02063358,  -0.10267162, 0.8062973,   -0.38149622, -0.40134856,
        -0.03353126, 0.38991132,  -0.3478111,  0.03661491,  0.25783631,  0.62772679,  -0.1961118,
        0.76423508,  -0.36241418, -0.20994355, -0.12368261, -0.9406727,  0.02340185,  -0.08793129,
        -0.02471633, -0.58163726, -0.02211772, -0.42014724, 0.77525634,  0.504951,    -0.20537445,
        -0.20369984, -0.83037728, -1.40423918, -0.46160448, -0.22944322, 0.36074194,  0.49579027,
        0.46527559};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(conv2d_test)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        2.71567607,  -0.9960829,  0.91671127,  0.28140706,  0.63235772,  0.08077253,  0.80927712,
        -0.59108931, -1.05421555, -2.76622486, -0.85044265, -0.52049929, 0.67726439,  -0.65290606,
        0.02345525,  -0.33579525, 0.38901961,  1.05473483,  -1.31188095, 1.8963089,   -0.07265259,
        0.947339,    0.41949373,  -0.70814759, 0.25892952,  1.07311416,  1.2571274,   -0.62318051,
        -0.19951548, -0.94232577, -0.29393643, 0.42292568,  -0.80230367, 1.40909171,  0.63617158,
        0.13900366,  1.09253144,  -0.15265895, 1.54781747,  0.72780299,  1.09189606,  -0.38068101,
        0.97057933,  -0.58958799, 1.56188643,  0.21474874,  0.58725154,  -1.27097559, -0.03024297,
        1.09437096,  -0.4897908,  0.34838957,  -1.31042492, -1.69069934, 0.86956722,  -0.40457946,
        0.46691212,  1.29273605,  0.26464137,  0.22073045,  -1.02178168, 0.22163901,  -1.84387338,
        0.75522131,  -0.45775682, -0.42241111, -1.50944722, 1.07256448,  -1.95876884, -0.28106022,
        0.3341668,   2.13129425,  -1.14728117, -1.06555498, -0.298444,   -0.88322699, -0.65866792,
        -2.06007552, 0.01374334,  0.45612028,  0.52715492,  1.01914406,  -1.72659791, 0.80650896,
        0.16860051,  2.24112225,  -0.78620857, 0.36566174,  -0.07020134, -0.47976932, -0.68230027,
        -0.94711417, -0.54506505, 1.66504931,  -0.71860826, 0.61132306};

    std::vector<float> c = {
        2.82721668e-02,  6.44195229e-02,  1.53499246e-02,  1.72468081e-01,  -6.33238107e-02,
        9.49496776e-02,  1.40258059e-01,  -7.92879611e-02, -1.29301161e-01, 3.11307609e-03,
        -1.90624535e-01, 1.13238767e-01,  -2.80647576e-02, 3.12882811e-02,  -3.52091640e-02,
        3.33581865e-02,  6.43158704e-02,  7.40238279e-02,  -1.00106120e-01, -9.56912562e-02,
        1.44342467e-01,  9.40258950e-02,  6.36333972e-02,  1.66158378e-03,  -8.91554281e-02,
        2.58734226e-02,  1.70919895e-02,  1.78214177e-01,  8.84564668e-02,  8.98126513e-02,
        -1.63809001e-01, 1.37802169e-01,  1.66439757e-01,  -1.45631135e-02, 1.88469887e-04,
        4.76950556e-02,  -1.91969007e-01, -1.76233292e-01, -7.70473927e-02, 1.14828631e-01,
        1.76608220e-01,  -1.50728196e-01, 1.99946314e-02,  -5.88052124e-02, 1.31612435e-01,
        1.61106288e-02,  -1.35080189e-01, 1.49512306e-01,  3.86456847e-02,  1.29330024e-01,
        -3.22975963e-02, -5.60784787e-02, -5.41997552e-02, 4.78562862e-02};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 4, 4}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});

    migraphx::shape c_shape{migraphx::shape::float_type, {2, 3, 3, 3}};
    auto cl = mm->add_literal(migraphx::literal{c_shape, c});

    mm->add_instruction(migraphx::make_op("convolution"), al, cl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.27039781,
                               0.19105849,
                               -0.06339942,
                               -0.65087199,
                               0.40867025,
                               0.05063812,
                               -0.14907975,
                               0.49018705,
                               -0.49197209,
                               0.33236548,
                               -0.39374301,
                               0.16012701,
                               0.06574871,
                               0.71606487,
                               -0.55201721,
                               -0.46427044};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(conv2d_dilation_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<float> a(16);
    std::iota(a.begin(), a.end(), 0.0f);
    std::vector<float> c(9);
    std::iota(c.begin(), c.end(), 0.0f);

    migraphx::shape a_shape{migraphx::shape::float_type, {1, 1, 4, 4}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});

    migraphx::shape c_shape{migraphx::shape::float_type, {1, 1, 3, 3}};
    auto cl = mm->add_literal(migraphx::literal{c_shape, c});

    mm->add_instruction(
        migraphx::make_op("convolution",
                          {{"padding", {1, 1}}, {"stride", {1, 1}}, {"dilation", {2, 2}}}),
        al,
        cl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {266, 206, 98, 66};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(conv1d_grouped_test)
{
    // values generated using torch.nn.Conv1d()
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> a = {
        -1.8177425861e-01, -1.2626364231e+00, 1.6530422866e-01,  3.7044650316e-01,
        1.5661215782e-01,  -4.8390197754e-01, -1.6961209476e-01, -6.8163943291e-01,
        -1.2525339127e+00, 8.5230982304e-01,  1.2412364483e+00,  -6.5990972519e-01,
        -1.1501162052e+00, 6.8731909990e-01,  -3.2288840413e-01, 1.1013785005e-01,
        -2.6825296879e-01, 2.4173924327e-01,  -6.6990071535e-01, -1.5921423435e+00};

    std::vector<float> c = {4.1833680868e-01,
                            -2.5043043494e-01,
                            7.6881540008e-03,
                            2.5368443131e-01,
                            8.7439371273e-03,
                            -5.3004939109e-02,
                            -7.4075937271e-02,
                            2.1147231758e-01,
                            -4.2490333319e-01,
                            3.2308679819e-01,
                            -5.1823651791e-01,
                            -4.1125145555e-01};

    migraphx::shape a_shape{migraphx::shape::float_type, {1, 4, 5}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});

    migraphx::shape c_shape{migraphx::shape::float_type, {4, 1, 3}};
    auto cl = mm->add_literal(migraphx::literal{c_shape, c});

    mm->add_instruction(
        migraphx::make_op("convolution",
                          {{"padding", {0, 0}}, {"stride", {1}}, {"dilation", {1}}, {"group", 4}}),
        al,
        cl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector(12);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {2.4143062532e-01,
                               -5.6675648689e-01,
                               -2.2414173931e-02,
                               -8.8111214340e-02,
                               1.7402324826e-02,
                               -2.2905001044e-01,
                               2.5718981028e-01,
                               -4.8637849092e-01,
                               3.6774125695e-01,
                               7.5186952949e-02,
                               6.3550546765e-02,
                               1.0800406933e+00};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(conv2d_grouped_test)
{
    // values generated using torch.nn.Conv2d()
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> a = {
        -1.8177425861e-01, -1.2626364231e+00, 1.6530422866e-01,  3.7044650316e-01,
        -1.0667574406e+00, -2.2788441181e-01, -6.6322600842e-01, -7.9737424850e-01,
        -2.4671976566e+00, -2.2557601333e-01, 9.1932612658e-01,  -2.3837232590e+00,
        3.5043731332e-01,  1.5977258682e+00,  4.5377662778e-01,  -8.0431783199e-01,
        -1.2409560680e+00, -5.4294246435e-01, -5.3603738546e-01, 1.3082869351e-01,
        1.5661215782e-01,  -4.8390197754e-01, -1.6961209476e-01, -6.8163943291e-01,
        -4.8902845383e-01, 1.0724093914e+00,  -5.7206004858e-01, -8.7797451019e-01,
        -1.1501162052e+00, 6.8731909990e-01,  -3.2288840413e-01, 1.1013785005e-01,
        -1.2336454391e+00, -2.8755193949e-01, -2.0737731457e+00, -6.2202119827e-01,
        1.3445684910e+00,  1.5188463032e-01,  1.1277725697e+00,  9.8665112257e-01,
        -5.8740414679e-02, 8.5314877331e-02,  8.8430750370e-01,  -1.3925373554e+00,
        1.4687923193e+00,  -8.0164027214e-01, 3.7581253052e-01,  1.2531118393e+00,
        -2.2203404903e+00, -1.4825233221e+00, -3.0809181929e-01, 4.9189615250e-01,
        1.2713159323e+00,  1.7327500582e+00,  1.7120348215e+00,  3.6474102736e-01,
        1.5471659601e-01,  -9.7715801001e-01, -4.3451300263e-01, -5.8212316036e-01,
        -1.7497187853e+00, 2.8677374125e-01,  -5.3925972432e-02, 5.2807545662e-01,
        -3.0451023579e-01, -1.1294512749e+00, 9.2312860489e-01,  -2.5824943185e-01,
        -2.1930438280e-01, -1.9152478874e-01, -4.5927390456e-01, -6.1666041613e-01,
        -9.1035616398e-01, -6.9311809540e-01, -8.4171491861e-01, 1.1104973555e+00,
        1.0168340206e+00,  7.5626128912e-01,  -6.4824336767e-01, -5.8247423172e-01,
        -2.3152463436e+00, 9.1751605272e-02,  -2.6009631157e-01, 6.3553839922e-01,
        1.3880019188e+00,  1.0628105402e+00,  1.1259366572e-01,  -7.3319464922e-01,
        9.9693065882e-01,  9.2709171772e-01,  -6.9960361719e-01, 8.1043428183e-01,
        5.4199612141e-01,  2.3824901581e+00,  7.8938305378e-01,  8.0670493841e-01,
        7.2724384069e-01,  7.6283775270e-02,  3.8195171952e-01,  2.7709417045e-02};

    std::vector<float> c = {-6.2048994005e-02,
                            1.1337312311e-01,
                            2.9074615240e-01,
                            2.7810212970e-01,
                            1.5443325043e-01,
                            -2.5806081295e-01,
                            2.6786279678e-01,
                            -3.2373470068e-01,
                            -6.6714607179e-02,
                            1.3622359931e-01,
                            -3.1335243583e-01,
                            4.5211236924e-02,
                            -1.2248075008e-01,
                            2.6443171501e-01,
                            2.5018900633e-01,
                            5.7185053825e-02,
                            1.7449633777e-01,
                            -2.1992056072e-01,
                            2.0849828422e-01,
                            -8.6234018207e-02,
                            -1.5416269004e-01,
                            -1.0765707493e-01,
                            3.8317047060e-02,
                            -1.0307097435e-01,
                            5.8663964272e-02,
                            2.7788731456e-01,
                            -2.8074741364e-02,
                            -1.7016741633e-01,
                            -1.2533438206e-01,
                            2.5698691607e-01,
                            -1.8419043720e-01,
                            2.3588168621e-01,
                            -3.0398529023e-02,
                            -2.2950637341e-01,
                            -2.4686738849e-01,
                            1.0772538185e-01

    };

    migraphx::shape a_shape{migraphx::shape::float_type, {1, 4, 5, 5}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});

    migraphx::shape c_shape{migraphx::shape::float_type, {4, 1, 3, 3}};
    auto cl = mm->add_literal(migraphx::literal{c_shape, c});

    mm->add_instruction(migraphx::make_op("convolution", {{"group", 4}}), al, cl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        9.5073115826e-01,  -3.2465749979e-01, -1.2766140699e+00, -2.7326330543e-01,
        -1.9085615873e+00, -4.9353949726e-02, -2.9069831967e-01, 3.3230510354e-01,
        2.3165440559e-01,  2.1132601798e-01,  -4.1108575463e-01, -1.8341702223e-01,
        1.5251106024e-01,  1.0116391182e+00,  -1.6068032384e-01, -5.0404131413e-01,
        7.9000133276e-01,  -3.8690292835e-01, -2.5670701265e-01, -2.9649671912e-01,
        1.5239302814e-01,  6.1578142643e-01,  8.4686398506e-02,  -8.9813157916e-02,
        -3.7039354444e-01, -3.7333327532e-01, -1.9308039546e-01, -1.6771307215e-02,
        -2.6942315698e-01, -1.9471725449e-02, 1.8760623038e-01,  -1.0682649910e-01,
        3.5812236369e-02,  -4.3629702926e-01, 9.0488202870e-02,  5.6741178036e-01};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(conv3d_grouped_test)
{
    // values generated using torch.nn.Conv3d()
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> a = {
        -1.8177425861e-01, -1.2626364231e+00, 1.6530422866e-01,  3.7044650316e-01,
        -1.0667574406e+00, -2.2788441181e-01, -6.6322600842e-01, -7.9737424850e-01,
        -2.4671976566e+00, -2.2557601333e-01, 9.1932612658e-01,  -2.3837232590e+00,
        3.5043731332e-01,  1.5977258682e+00,  4.5377662778e-01,  -8.0431783199e-01,
        -1.2409560680e+00, -5.4294246435e-01, -5.3603738546e-01, 1.3082869351e-01,
        1.5661215782e-01,  -4.8390197754e-01, -1.6961209476e-01, -6.8163943291e-01,
        -4.8902845383e-01, 1.0724093914e+00,  -5.7206004858e-01, -8.7797451019e-01,
        -1.1501162052e+00, 6.8731909990e-01,  -3.2288840413e-01, 1.1013785005e-01,
        -1.2336454391e+00, -2.8755193949e-01, -2.0737731457e+00, -6.2202119827e-01,
        1.3445684910e+00,  1.5188463032e-01,  1.1277725697e+00,  9.8665112257e-01,
        -5.8740414679e-02, 8.5314877331e-02,  8.8430750370e-01,  -1.3925373554e+00,
        1.4687923193e+00,  -8.0164027214e-01, 3.7581253052e-01,  1.2531118393e+00,
        -2.2203404903e+00, -1.4825233221e+00, -3.0809181929e-01, 4.9189615250e-01,
        1.2713159323e+00,  1.7327500582e+00,  1.7120348215e+00,  3.6474102736e-01,
        1.5471659601e-01,  -9.7715801001e-01, -4.3451300263e-01, -5.8212316036e-01,
        -1.7497187853e+00, 2.8677374125e-01,  -5.3925972432e-02, 5.2807545662e-01,
        -3.0451023579e-01, -1.1294512749e+00, 9.2312860489e-01,  -2.5824943185e-01,
        -2.1930438280e-01, -1.9152478874e-01, -4.5927390456e-01, -6.1666041613e-01,
        -9.1035616398e-01, -6.9311809540e-01, -8.4171491861e-01, 1.1104973555e+00,
        1.0168340206e+00,  7.5626128912e-01,  -6.4824336767e-01, -5.8247423172e-01,
        -2.3152463436e+00, 9.1751605272e-02,  -2.6009631157e-01, 6.3553839922e-01,
        -6.5450698137e-01, 1.4767274857e+00,  8.8887333870e-01,  -5.1203089952e-01,
        -7.5896525383e-01, 1.6613142490e+00,  2.8061065078e-01,  7.2000652552e-01,
        -3.2952797413e-01, 1.5448122025e+00,  2.5642251596e-02,  -1.6934386492e+00,
        -1.1651701927e+00, -7.6255226135e-01, -9.3927308917e-02, -4.7004006803e-02,
        1.3880019188e+00,  1.0628105402e+00,  1.1259366572e-01,  -7.3319464922e-01,
        -2.4494104087e-01, 1.0328419209e+00,  6.7040407658e-01,  2.4470899999e-01,
        5.4199612141e-01,  2.3824901581e+00,  7.8938305378e-01,  8.0670493841e-01,
        -5.2894420922e-02, 1.2305746228e-01,  8.5191625357e-01,  1.0401779413e-01,
        -8.2669818401e-01, -4.2143875360e-01, 2.3040554523e+00,  -2.0323581696e+00,
        -4.5673078299e-01, 1.0564883053e-01,  -6.0439848900e-01, 9.4357421622e-03,
        6.0120218992e-01,  -1.4283125401e+00, 4.3909239769e-01,  -9.2252081633e-01,
        -1.0252687335e-01, -1.1247619390e+00, -1.9088345766e+00, -7.4897360802e-01,
        -1.4414031506e+00, 5.2919095755e-01,  -1.7667583227e+00, 3.1745910645e-02,
        8.6596679688e-01,  1.8094559908e+00,  -7.2095400095e-01, 1.0962404013e+00,
        -8.8948321342e-01, -3.0480036139e-01, 4.7870498896e-01,  9.6183389425e-01,
        4.6915197372e-01,  -2.0562076569e+00, -1.2713176012e-01, 1.9478828907e+00,
        9.6827542782e-01,  1.1638808250e-01,  5.2549332380e-01,  8.3114856482e-01,
        -7.2651541233e-01, 3.2562047243e-01,  7.0021921396e-01,  -2.2720028460e-01,
        5.2363085747e-01,  5.7012671232e-01,  -4.1096709669e-02, 1.2149795294e+00,
        3.8041347265e-01,  -7.5754743814e-01, 2.2944717109e-01,  -1.2558351755e+00,
        1.0870405287e-01,  -5.9034329653e-01, 4.5456910133e-01,  4.7747135162e-01,
        4.8018571734e-01,  -7.1751528978e-01, -6.5281885862e-01, -4.1015291214e-01,
        -5.2255600691e-01, 6.6819953918e-01,  4.3428021669e-01,  -4.6067813039e-01,
        2.7472496033e-01,  -2.3150257766e-01, -9.3255436420e-01, 1.7827729881e-01,
        7.2644609213e-01,  3.4542816877e-01,  -1.4732214808e-01, 1.8185671568e+00,
        -7.5135729276e-03, 1.2936640978e+00,  -1.5610131435e-02, 9.5702521503e-02,
        7.3214960098e-01,  -1.6284734011e+00, -2.3495199680e+00, 1.8819983304e-01,
        4.1768592596e-01,  -1.4365546703e+00, 4.3632332236e-02,  1.8172906339e-01,
        7.2902792692e-01,  -1.5549405813e+00, 1.0908687115e+00,  -2.1934424341e-01,
        -3.3904924989e-01, -5.8759814501e-01, -8.1080150604e-01, 1.1492904425e+00,
        1.3274430037e+00,  9.2008709908e-01,  2.6716779917e-02,  -1.2625107765e+00,
        -1.0983025655e-02, -2.1104571223e-01, 7.1448758245e-02,  8.0006763339e-02,
        -6.0026758909e-01, -8.0451929569e-01, -2.0860210061e-01, -1.1645827293e+00,
        1.1662654877e+00,  -1.4570258558e-01, -1.1950391531e+00, 7.9169058800e-01,
        -2.3888850212e-01, 7.2735357285e-01,  1.0767338276e+00,  7.6857763529e-01,
        1.3455440998e+00,  -1.9133212566e+00, -9.5440566540e-01, 2.2016492486e-01,
        1.4171910286e+00,  1.2438772917e+00,  2.8923720121e-01,  1.0307216644e+00,
        -1.1049110889e+00, 8.1931585073e-01,  1.2717065811e+00,  7.2451567650e-01,
        4.2772480845e-01,  6.6778925247e-03,  3.6661916971e-01,  8.4176588058e-01,
        1.7524845898e-01,  9.3251101673e-02,  1.6939337552e-01,  -5.3958308697e-01,
        -2.4632980824e+00, 2.1393857002e+00,  5.4651170969e-01,  -8.7673731148e-02,
        -1.1910932064e+00, -5.3464317322e-01, 9.0146481991e-01,  5.4449105263e-01,
        3.3046221733e-01,  1.8962550163e+00,  -2.5553891435e-02, -4.5766282082e-01};

    std::vector<float> c = {
        5.5225893855e-02,  1.7669884861e-01,  -1.4823813736e-01, -1.5093412995e-01,
        1.9241893664e-02,  8.1974424422e-02,  -1.9087575376e-01, 1.0307391733e-01,
        -1.4162765443e-01, -9.9507309496e-02, -1.6491030157e-01, -1.9087164104e-01,
        1.3383334875e-01,  -1.7198354006e-01, 1.1675053835e-01,  -1.7222750187e-01,
        -1.6482289135e-01, -1.6944560409e-01, 4.2948201299e-02,  -1.7519497871e-01,
        8.0922402442e-02,  1.6257633269e-01,  1.3960868120e-01,  -1.0616634041e-02,
        1.2709778547e-01,  -7.9442635179e-02, 1.0763156414e-01,  1.4474269748e-01,
        -1.7698179185e-01, -1.8627175689e-01, -1.3962627947e-01, 5.8670423925e-02,
        -1.6315481067e-01, -1.1026533693e-01, -1.4999851584e-01, -1.3568796217e-01,
        5.9554006904e-02,  -1.7530122399e-01, 1.1991871148e-01,  -1.0294853896e-01,
        1.1544618011e-01,  1.7758098245e-01,  -1.3080266118e-01, -8.9707314968e-02,
        5.4990760982e-02,  1.2711614370e-02,  2.5931067765e-02,  9.0181067586e-02,
        1.1194603145e-01,  3.3410452306e-02,  1.6982021928e-01,  -6.8565264344e-02,
        -6.0046706349e-02, -4.1276197881e-02, 7.6800853014e-02,  -1.7256736755e-02,
        -1.6289904714e-01, -1.7800724506e-01, 2.7882317081e-02,  -3.2509204000e-02,
        -1.6520440578e-01, 1.2653686106e-01,  1.8854382634e-01,  1.8794012070e-01,
        1.1720044911e-01,  1.2710493803e-01,  -1.4607259631e-01, 1.7739501595e-01,
        -1.2449865788e-01, -5.7230871171e-02, 1.5655733645e-01,  1.7980879545e-01,
        -1.3897077739e-01, 6.0994960368e-02,  -6.6233433783e-02, -8.2589887083e-02,
        1.5292733908e-01,  -3.3969566226e-02, 1.9211943448e-01,  1.5730638802e-01,
        -7.7081754804e-02, 1.7120510340e-01,  1.3997478783e-01,  -4.6440314502e-02,
        -4.6346023679e-02, 5.8616649359e-02,  1.3451220095e-01,  -5.1839973778e-02,
        -1.1959856004e-01, 1.2231089920e-01,  2.2015446797e-02,  -5.9318508953e-02,
        -1.6406981647e-01, -3.0361138284e-02, 1.1130692437e-03,  1.3459129632e-01,
        7.3153721169e-03,  -9.3296334147e-02, 1.6970252991e-01,  1.8314011395e-02,
        1.5286317468e-01,  -2.9979798943e-02, 5.8287456632e-02,  1.5691184998e-01,
        -1.1154287308e-01, 1.0704345256e-01,  -1.2394358218e-01, -1.6357544065e-01};

    migraphx::shape a_shape{migraphx::shape::float_type, {1, 4, 4, 4, 4}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});

    migraphx::shape c_shape{migraphx::shape::float_type, {4, 1, 3, 3, 3}};
    auto cl = mm->add_literal(migraphx::literal{c_shape, c});

    mm->add_instruction(
        migraphx::make_op(
            "convolution",
            {{"padding", {0, 0, 0}}, {"stride", {1, 1, 1}}, {"dilation", {1, 1, 1}}, {"group", 4}}),
        al,
        cl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector(32);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        7.0454144478e-01,  6.5989446640e-01,  8.0167812109e-01,  6.3199982978e-03,
        1.4711283445e+00,  6.3151007891e-01,  -6.1401832104e-01, -1.5212227106e+00,
        6.6702485085e-01,  -4.8160430789e-01, 1.7179620266e-01,  -4.2418915033e-01,
        -3.0498113483e-02, -1.1491653919e+00, -5.6819570065e-01, -5.1951372623e-01,
        2.6523154974e-01,  -2.9469928145e-01, 4.4018003345e-01,  7.3754268885e-01,
        -1.2682390399e-02, -7.2735935450e-01, 3.8021117449e-01,  -1.0337758064e+00,
        -7.3352378607e-01, -1.5659983456e-01, -3.4672033787e-01, 1.7649652064e-01,
        9.1797195375e-02,  1.8180884421e-01,  -3.7545084953e-01, 3.3764648438e-01};

    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(conv3d_test)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        2.71567607,  -0.9960829,  0.91671127,  0.28140706,  0.63235772,  0.08077253,  0.80927712,
        -0.59108931, -1.05421555, -2.76622486, -0.85044265, -0.52049929, 0.67726439,  -0.65290606,
        0.02345525,  -0.33579525, 0.38901961,  1.05473483,  -1.31188095, 1.8963089,   -0.07265259,
        0.947339,    0.41949373,  -0.70814759, 0.25892952,  1.07311416,  1.2571274,   -0.62318051,
        -0.19951548, -0.94232577, -0.29393643, 0.42292568,  -0.80230367, 1.40909171,  0.63617158,
        0.13900366,  1.09253144,  -0.15265895, 1.54781747,  0.72780299,  1.09189606,  -0.38068101,
        0.97057933,  -0.58958799, 1.56188643,  0.21474874,  0.58725154,  -1.27097559, -0.03024297,
        1.09437096,  -0.4897908,  0.34838957,  -1.31042492, -1.69069934, 0.86956722,  -0.40457946,
        0.46691212,  1.29273605,  0.26464137,  0.22073045,  -1.02178168, 0.22163901,  -1.84387338,
        0.75522131,  -0.45775682, -0.42241111, -1.50944722, 1.07256448,  -1.95876884, -0.28106022,
        0.3341668,   2.13129425,  -1.14728117, -1.06555498, -0.298444,   -0.88322699, -0.65866792,
        -2.06007552, 0.01374334,  0.45612028,  0.52715492,  1.01914406,  -1.72659791, 0.80650896,
        0.16860051,  2.24112225,  -0.78620857, 0.36566174,  -0.07020134, -0.47976932, -0.68230027,
        -0.94711417, -0.54506505, 1.66504931,  -0.71860826, 0.61132306};

    std::vector<float> c = {
        2.82721668e-02,  6.44195229e-02,  1.53499246e-02,  1.72468081e-01,  -6.33238107e-02,
        9.49496776e-02,  1.40258059e-01,  -7.92879611e-02, -1.29301161e-01, 3.11307609e-03,
        -1.90624535e-01, 1.13238767e-01,  -2.80647576e-02, 3.12882811e-02,  -3.52091640e-02,
        3.33581865e-02,  6.43158704e-02,  7.40238279e-02,  -1.00106120e-01, -9.56912562e-02,
        1.44342467e-01,  9.40258950e-02,  6.36333972e-02,  1.66158378e-03,  -8.91554281e-02,
        2.58734226e-02,  1.70919895e-02,  1.78214177e-01,  8.84564668e-02,  8.98126513e-02,
        -1.63809001e-01, 1.37802169e-01,  1.66439757e-01,  -1.45631135e-02, 1.88469887e-04,
        4.76950556e-02,  -1.91969007e-01, -1.76233292e-01, -7.70473927e-02, 1.14828631e-01,
        1.76608220e-01,  -1.50728196e-01, 1.99946314e-02,  -5.88052124e-02, 1.31612435e-01,
        1.61106288e-02,  -1.35080189e-01, 1.49512306e-01,  3.86456847e-02,  1.29330024e-01,
        -3.22975963e-02, -5.60784787e-02, -5.41997552e-02, 4.78562862e-02};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 4, 4, 1}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});

    migraphx::shape c_shape{migraphx::shape::float_type, {2, 3, 3, 3, 1}};
    auto cl = mm->add_literal(migraphx::literal{c_shape, c});

    mm->add_instruction(
        migraphx::make_op("convolution",
                          {{"padding", {0, 0, 0}}, {"stride", {1, 1, 1}}, {"dilation", {1, 1, 1}}}),
        al,
        cl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0.27039781,
                               0.19105849,
                               -0.06339942,
                               -0.65087199,
                               0.40867025,
                               0.05063812,
                               -0.14907975,
                               0.49018705,
                               -0.49197209,
                               0.33236548,
                               -0.39374301,
                               0.16012701,
                               0.06574871,
                               0.71606487,
                               -0.55201721,
                               -0.46427044};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}
