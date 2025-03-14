/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>
#include <onnx_verify_utils.hpp>

TEST_CASE(nhwcconv_test)
{
    migraphx::program p = read_onnx("nhwcconv_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::float_type, {1, 7, 7, 1}};
    std::vector<float> x_data = {
        0.45246148109436035f,   0.15498268604278564f,  0.11199361085891724f,
        -0.39421093463897705f,  0.2626858949661255f,   0.13414543867111206f,
        -0.27184486389160156f,  -0.43028733134269714f, -0.26825493574142456f,
        0.3893144130706787f,    -0.13631996512413025f, -0.009590476751327515f,
        -0.48771554231643677f,  -0.25256502628326416f, -0.2812897562980652f,
        0.4043201804161072f,    0.07795023918151855f,  0.326981782913208f,
        0.13114392757415771f,   -0.4416425824165344f,  0.12446999549865723f,
        0.36739975214004517f,   0.1698915958404541f,   0.2008744478225708f,
        0.23339951038360596f,   0.38613730669021606f,  0.11117297410964966f,
        0.3877097964286804f,    0.20812749862670898f,  -0.34297940135002136f,
        -0.029246658086776733f, -0.20483523607254028f, -0.19244328141212463f,
        -0.11104947328567505f,  -0.32830488681793213f, -0.01800677180290222f,
        0.3618946671485901f,    -0.40949052572250366f, -0.18248388171195984f,
        -0.3349453806877136f,   -0.34091079235076904f, 0.006497859954833984f,
        0.4537564516067505f,    0.08006560802459717f,  -0.14788749814033508f,
        0.034442365169525146f,  -0.33322954177856445f, 0.06049239635467529f,
        0.42619407176971436f};

    migraphx::shape w_shape{migraphx::shape::float_type, {1, 1, 1, 1}};
    std::vector<float> w_data = {-0.4406261742115021f};

    migraphx::parameter_map pm;
    pm["0"] = migraphx::argument{x_shape, x_data.data()};
    pm["1"] = migraphx::argument{w_shape, w_data.data()};

    auto result = p.eval(pm).back();
    EXPECT(result.get_shape().lens() == std::vector<std::size_t>{1, 7, 7, 1});

    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        -0.19936637580394745f,  -0.06828942894935608f,  -0.04934731498360634f,
        0.17369966208934784f,   -0.11574628204107285f,  -0.05910799279808998f,
        0.1197819635272026f,    0.18959586322307587f,   0.1182001456618309f,
        -0.17154212296009064f,  0.06006614491343498f,   0.0042258151806890965f,
        0.21490024030208588f,   0.11128675937652588f,   0.12394362688064575f,
        -0.17815405130386353f,  -0.034346915781497955f, -0.14407673478126526f,
        -0.05778544768691063f,  0.19459928572177887f,   -0.05484473705291748f,
        -0.16188594698905945f,  -0.07485868036746979f,  -0.08851054310798645f,
        -0.10284193605184555f,  -0.17014220356941223f,  -0.04898572340607643f,
        -0.17083507776260376f,  -0.09170642495155334f,  0.1511256992816925f,
        0.012886842712759972f,  0.09025576710700989f,   0.08479554951190948f,
        0.0489313043653965f,    0.14465972781181335f,   0.007934254594147205f,
        -0.15946026146411896f,  0.1804322451353073f,    0.08040717244148254f,
        0.1475857049226761f,    0.15021422505378723f,   -0.0028631272725760937f,
        -0.19993697106838226f,  -0.03527900204062462f,  0.06516310572624207f,
        -0.015176207758486271f, 0.14682966470718384f,   -0.02665453404188156f,
        -0.18779225647449493f};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
