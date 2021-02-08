#include <migraphx/program.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/operators.hpp>
#include <sstream>
#include <migraphx/make_op.hpp>

#include <migraphx/serialize.hpp>

#include "test.hpp"

template <class... Ts>
void expect_shape(const migraphx::shape& expected, const migraphx::operation& op, Ts... xs)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<migraphx::shape> shapes{xs...};
    std::vector<migraphx::instruction_ref> args(shapes.size());
    std::transform(
        shapes.begin(), shapes.end(), args.begin(), [&](auto&& s) { return mm->add_outline(s); });
    mm->add_instruction(op, args);
    if(p.get_output_shapes().back() != expected)
    {
        std::cout << "FAILED: Incorrect shape for " << op << ": ";
        std::cout << expected << " != " << p.get_output_shapes().back() << std::endl;
        for(auto&& s : shapes)
            std::cout << "    " << s << std::endl;
    }
}

template <class... Ts>
void throws_shape(const migraphx::operation& op, Ts... xs)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<migraphx::shape> shapes{xs...};
    std::vector<migraphx::instruction_ref> args(shapes.size());
    std::transform(
        shapes.begin(), shapes.end(), args.begin(), [&](auto&& s) { return mm->add_outline(s); });
    bool thrown = test::throws([&] { mm->add_instruction(op, args); });
    if(not thrown)
    {
        std::cout << "FAILED: No error found for " << op.name() << ": ";
        for(auto&& s : shapes)
            std::cout << "    " << s << std::endl;
    }
}

template <class...>
struct always_false : std::false_type
{
};

template <class... Ts>
void throws_shape(const migraphx::shape&, Ts...)
{
    static_assert(always_false<Ts...>{},
                  "An expected shape should not be passed to throws_shape function");
}

TEST_CASE(batch_norm_inference_shape)
{
    const size_t channels = 3;
    migraphx::shape s{migraphx::shape::float_type, {4, channels, 3, 3}};
    migraphx::shape vars{migraphx::shape::float_type, {channels}};
    expect_shape(s, migraphx::make_op("batch_norm_inference"), s, vars, vars, vars, vars);
    throws_shape(migraphx::make_op("batch_norm_inference"), s);
    throws_shape(migraphx::make_op("batch_norm_inference"), s, vars, vars, vars, vars, vars);
}

TEST_CASE(convolution_shape)
{
    migraphx::shape output{migraphx::shape::float_type, {4, 4, 1, 1}};
    migraphx::shape input{migraphx::shape::float_type, {4, 3, 3, 3}};
    migraphx::shape weights{migraphx::shape::float_type, {4, 3, 3, 3}};
    expect_shape(output, migraphx::make_op("convolution"), input, weights);
    throws_shape(migraphx::make_op("convolution"), input);
    throws_shape(
        migraphx::make_op("convolution", {{"padding", {0}}, {"stride", {1}}, {"dilation", {1}}}),
        input);

    migraphx::shape input2{migraphx::shape::float_type, {3, 3}};
    migraphx::shape weights2{migraphx::shape::float_type, {3, 3}};
    throws_shape(migraphx::make_op("convolution"), input2, weights2);
    throws_shape(migraphx::make_op("convolution"), input2, weights);

    migraphx::shape output_1d{migraphx::shape::float_type, {4, 4, 1}};
    migraphx::shape input_1d{migraphx::shape::float_type, {4, 3, 3}};
    migraphx::shape weights_1d{migraphx::shape::float_type, {4, 3, 3}};
    expect_shape(
        output_1d,
        migraphx::make_op("convolution", {{"padding", {0}}, {"stride", {1}}, {"dilation", {1}}}),
        input_1d,
        weights_1d);

    migraphx::shape output_3d{migraphx::shape::float_type, {4, 4, 1, 1, 1}};
    migraphx::shape input_3d{migraphx::shape::float_type, {4, 3, 3, 3, 3}};
    migraphx::shape weights_3d{migraphx::shape::float_type, {4, 3, 3, 3, 3}};
    expect_shape(
        output_3d,
        migraphx::make_op("convolution",
                          {{"padding", {0, 0, 0}}, {"stride", {1, 1, 1}}, {"dilation", {1, 1, 1}}}),
        input_3d,
        weights_3d);

    throws_shape(migraphx::make_op("convolution"), input_3d, weights_3d);
}

TEST_CASE(deconvolution_shape)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 4, 1, 1}};
    migraphx::shape output{migraphx::shape::float_type, {4, 3, 3, 3}};
    migraphx::shape weights{migraphx::shape::float_type, {4, 3, 3, 3}};
    expect_shape(output, migraphx::make_op("deconvolution"), input, weights);
    throws_shape(migraphx::make_op("deconvolution"), input);
    throws_shape(
        migraphx::make_op("deconvolution", {{"padding", {0}}, {"stride", {1}}, {"dilation", {1}}}),
        input);

    migraphx::shape input_1d{migraphx::shape::float_type, {4, 4, 1}};
    migraphx::shape output_1d{migraphx::shape::float_type, {4, 3, 3}};
    migraphx::shape weights_1d{migraphx::shape::float_type, {4, 3, 3}};
    expect_shape(
        output_1d,
        migraphx::make_op("deconvolution", {{"padding", {0}}, {"stride", {1}}, {"dilation", {1}}}),
        input_1d,
        weights_1d);

    migraphx::shape input_3d{migraphx::shape::float_type, {4, 4, 1, 1, 1}};
    migraphx::shape output_3d{migraphx::shape::float_type, {4, 3, 3, 3, 3}};
    migraphx::shape weights_3d{migraphx::shape::float_type, {4, 3, 3, 3, 3}};
    expect_shape(
        output_3d,
        migraphx::make_op("deconvolution",
                          {{"padding", {0, 0, 0}}, {"stride", {1, 1, 1}}, {"dilation", {1, 1, 1}}}),
        input_3d,
        weights_3d);
}

TEST_CASE(quant_convolution_shape)
{
    migraphx::shape output{migraphx::shape::int32_type, {4, 4, 1, 1}};
    migraphx::shape input{migraphx::shape::int8_type, {4, 3, 3, 3}};
    migraphx::shape weights{migraphx::shape::int8_type, {4, 3, 3, 3}};
    expect_shape(output, migraphx::make_op("quant_convolution"), input, weights);
    throws_shape(migraphx::make_op("quant_convolution"), input);
    throws_shape(migraphx::make_op("quant_convolution",
                                   {{"padding", {0}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
                 input,
                 weights);
    throws_shape(migraphx::make_op("quant_convolution",
                                   {{"padding", {0}}, {"stride", {1}}, {"dilation", {1}}}),
                 input,
                 weights);

    migraphx::shape input2{migraphx::shape::int32_type, {3, 3}};
    migraphx::shape weights2{migraphx::shape::float_type, {3, 3}};
    throws_shape(migraphx::make_op("quant_convolution"), input2, weights2);
    throws_shape(migraphx::make_op("quant_convolution"), input2, weights);

    migraphx::shape input3{migraphx::shape::int32_type, {4, 3, 3, 3}};
    migraphx::shape weight3{migraphx::shape::float_type, {4, 3, 3, 3}};
    throws_shape(migraphx::make_op("quant_convolution"), input3, weights);
    throws_shape(migraphx::make_op("quant_convolution"), input, weight3);
    throws_shape(migraphx::make_op("quant_convolution"), input3, weight3);
}

TEST_CASE(pooling_shape)
{
    migraphx::shape output{migraphx::shape::float_type, {4, 3, 1, 1}};
    migraphx::shape input{migraphx::shape::float_type, {4, 3, 3, 3}};
    throws_shape(
        migraphx::make_op("pooling",
                          {{"mode", "max"}, {"padding", {1}}, {"stride", {0}}, {"lengths", {1}}}),
        input);
    expect_shape(
        output,
        migraphx::make_op(
            "pooling",
            {{"mode", "max"}, {"padding", {0, 0}}, {"stride", {3, 3}}, {"lengths", {1, 1}}}),
        input);

    migraphx::shape output1{migraphx::shape::float_type, {4, 3, 2, 2}};
    expect_shape(output1,
                 migraphx::make_op("pooling",
                                   {{"mode", "max"},
                                    {"padding", {0, 0}},
                                    {"stride", {3, 3}},
                                    {"lengths", {1, 1}},
                                    {"ceil_mode", true}}),
                 input);
}

TEST_CASE(inconsistent_attr_shape)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 3, 3, 3}};
    migraphx::shape weights{migraphx::shape::float_type, {4, 3, 3, 3}};
    throws_shape(migraphx::make_op("convolution",
                                   {{"padding", {1, 1}}, {"stride", {2}}, {"dilation", {3, 3, 3}}}),
                 input,
                 weights);
    throws_shape(migraphx::make_op("deconvolution",
                                   {{"padding", {1, 1}}, {"stride", {2}}, {"dilation", {3, 3, 3}}}),
                 input,
                 weights);
    throws_shape(
        migraphx::make_op(
            "pooling", {{"mode", "max"}, {"padding", {1}}, {"stride", {0}}, {"lengths", {1, 1}}}),
        input);
}

TEST_CASE(transpose_shape)
{
    migraphx::shape input{migraphx::shape::float_type, {2, 2}};
    migraphx::shape output{migraphx::shape::float_type, {2, 2}, {1, 2}};
    expect_shape(input, migraphx::make_op("transpose", {{"dims", {0, 1}}}), input);
    expect_shape(output, migraphx::make_op("transpose", {{"dims", {1, 0}}}), input);
    expect_shape(output, migraphx::make_op("transpose"), input);
    throws_shape(migraphx::make_op("transpose", {{"dims", {1, 2}}}), input);
}

TEST_CASE(contiguous_shape)
{
    migraphx::shape output{migraphx::shape::float_type, {2, 2}};
    migraphx::shape input{migraphx::shape::float_type, {2, 2}, {1, 2}};
    expect_shape(output, migraphx::make_op("contiguous"), input);
    throws_shape(migraphx::make_op("contiguous"), input, input);

    migraphx::shape single{migraphx::shape::float_type, {2}};
    expect_shape(single, migraphx::make_op("contiguous"), single);
}

TEST_CASE(contiguous_shape_scalar)
{
    migraphx::shape output{migraphx::shape::float_type};
    migraphx::shape input{migraphx::shape::float_type};
    expect_shape(output, migraphx::make_op("contiguous"), input);
}

TEST_CASE(reshape_shape)
{
    migraphx::shape input{migraphx::shape::float_type, {24, 1, 1, 1}};
    for(auto&& new_shape :
        std::vector<std::vector<int64_t>>{{8, 3, 1, 1}, {1, 3, 4, 2}, {1, 3, 4, 2}})
    {
        std::vector<std::size_t> lens(new_shape.size());
        std::copy(new_shape.begin(), new_shape.end(), lens.begin());
        migraphx::shape output{migraphx::shape::float_type, lens};
        expect_shape(output, migraphx::make_op("reshape", {{"dims", new_shape}}), input);
    }

    for(auto&& new_shape :
        std::vector<std::vector<int64_t>>{{8, 3, 2, 2}, {1, 3, -1, -1}, {3, 0, 0}, {3, 2, 0}})
    {
        throws_shape(migraphx::make_op("reshape", {{"dims", new_shape}}), input);
    }

    std::vector<std::pair<std::vector<int64_t>, migraphx::shape>> minus1_tests{
        {{2, -1, 3}, {migraphx::shape::float_type, {2, 4, 3}}},
        {{0, -1, 0}, {migraphx::shape::float_type, {24, 1, 1}}},
        {{2, -1, 0}, {migraphx::shape::float_type, {2, 12, 1}}},
        {{0, 0, -1}, {migraphx::shape::float_type, {24, 1, 1}}},
        {{2, 0, -1}, {migraphx::shape::float_type, {2, 1, 12}}},
        {{-1, 2, 3}, {migraphx::shape::float_type, {4, 2, 3}}},
        {{-1, 0, 3}, {migraphx::shape::float_type, {8, 1, 3}}},
        {{-1, 0, 0}, {migraphx::shape::float_type, {24, 1, 1}}},
        {{-1, 3, 0}, {migraphx::shape::float_type, {8, 3, 1}}}};

    for(auto& it : minus1_tests)
    {
        expect_shape(it.second, migraphx::make_op("reshape", {{"dims", it.first}}), input);
    }
}

TEST_CASE(flatten_shape)
{
    migraphx::shape input{migraphx::shape::float_type, {2, 4, 6, 8}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 2 * 4 * 6 * 8}},
                 migraphx::make_op("flatten", {{"axis", 0}}),
                 input);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 2 * 4 * 6 * 8}},
                 migraphx::make_op("flatten", {{"axis", -4}}),
                 input);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 4 * 6 * 8}},
                 migraphx::make_op("flatten", {{"axis", 1}}),
                 input);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 4 * 6 * 8}},
                 migraphx::make_op("flatten", {{"axis", -3}}),
                 input);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {2 * 4, 6 * 8}},
                 migraphx::make_op("flatten", {{"axis", 2}}),
                 input);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {2 * 4 * 6, 8}},
                 migraphx::make_op("flatten", {{"axis", 3}}),
                 input);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {2 * 4 * 6 * 8, 1}},
                 migraphx::make_op("flatten", {{"axis", 4}}),
                 input);
    throws_shape(migraphx::make_op("flatten", {{"axis", 5}}), input);
    throws_shape(migraphx::make_op("flatten", {{"axis", -5}}), input);
}

TEST_CASE(slice_shape)
{
    migraphx::shape input{migraphx::shape::int32_type, {2, 2, 3}};
    expect_shape(migraphx::shape{migraphx::shape::int32_type, {2, 2, 2}, {6, 3, 1}},
                 migraphx::make_op("slice", {{"axes", {2}}, {"starts", {1}}, {"ends", {3}}}),
                 input);
    expect_shape(migraphx::shape{migraphx::shape::int32_type, {2, 2, 2}, {6, 3, 1}},
                 migraphx::make_op(
                     "slice", {{"axes", {0, 1, 2}}, {"starts", {0, 0, 1}}, {"ends", {2, 2, 3}}}),
                 input);
    expect_shape(migraphx::shape{migraphx::shape::int32_type, {2, 2, 1}, {6, 3, 1}},
                 migraphx::make_op("slice", {{"axes", {2}}, {"starts", {2}}, {"ends", {10}}}),
                 input);
}

TEST_CASE(multibroadcast)
{
    {
        std::vector<std::size_t> lens{4, 2, 5, 3};
        migraphx::shape input{migraphx::shape::float_type, {2, 1, 3}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, lens, {0, 3, 0, 1}},
                     migraphx::make_op("multibroadcast", {{"output_lens", lens}}),
                     input);
    }
    {
        std::vector<std::size_t> lens{4, 2, 5, 3};
        migraphx::shape input{migraphx::shape::float_type, {2, 1, 1}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, lens, {0, 1, 0, 0}},
                     migraphx::make_op("multibroadcast", {{"output_lens", lens}}),
                     input);
    }
    {
        std::vector<std::size_t> lens{4, 2, 5, 3};
        migraphx::shape input{migraphx::shape::float_type, {5, 1}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, lens, {0, 0, 1, 0}},
                     migraphx::make_op("multibroadcast", {{"output_lens", lens}}),
                     input);
    }
    {
        std::vector<std::size_t> lens{4, 2, 5, 3};
        migraphx::shape input{migraphx::shape::float_type, {4, 1, 1, 1}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, lens, {1, 0, 0, 0}},
                     migraphx::make_op("multibroadcast", {{"output_lens", lens}}),
                     input);
    }
    {
        std::vector<std::size_t> lens{4, 2, 5, 3};
        migraphx::shape input{migraphx::shape::float_type, {3}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, lens, {0, 0, 0, 1}},
                     migraphx::make_op("multibroadcast", {{"output_lens", lens}}),
                     input);
    }
    {
        std::vector<std::size_t> lens{4, 4, 1, 3};
        migraphx::shape input{migraphx::shape::float_type, {4, 1, 3}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, lens, {0, 3, 3, 1}},
                     migraphx::make_op("multibroadcast", {{"output_lens", lens}}),
                     input);
    }
    {
        std::vector<std::size_t> lens{4, 1, 1, 3};
        migraphx::shape input{migraphx::shape::float_type, {4, 1, 1, 1}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, lens, {1, 1, 1, 0}},
                     migraphx::make_op("multibroadcast", {{"output_lens", lens}}),
                     input);
    }
    {
        std::vector<std::size_t> lens{4, 1, 3};
        migraphx::shape input{migraphx::shape::float_type, {4, 1, 1, 1}};
        throws_shape(migraphx::make_op("multibroadcast", {{"output_lens", lens}}), input);
    }
    {
        std::vector<std::size_t> lens{4, 1, 3};
        migraphx::shape input{migraphx::shape::float_type, {}};
        throws_shape(migraphx::make_op("multibroadcast", {{"output_lens", lens}}), input);
    }
    {
        std::vector<std::size_t> lens{2, 3, 4, 5};
        migraphx::shape input{migraphx::shape::float_type, {3, 4}};
        throws_shape(migraphx::make_op("multibroadcast", {{"output_lens", lens}}), input);
    }
    {
        std::vector<std::size_t> lens{2, 3, 4, 5};
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4}};
        throws_shape(migraphx::make_op("multibroadcast", {{"output_lens", lens}}), input);
    }
}

TEST_CASE(broadcast)
{
    {
        std::vector<std::size_t> lens{1, 1};
        migraphx::shape input{migraphx::shape::float_type, {1}, {0}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 1}, {0, 0}},
                     migraphx::make_op("broadcast", {{"axis", 0}, {"dims", lens}}),
                     input);
    }

    {
        std::vector<std::size_t> lens{1, 1};
        migraphx::shape input{migraphx::shape::float_type, {2}};
        throws_shape(migraphx::op::broadcast{1, lens}, input);
    }

    {
        std::vector<std::size_t> lens{2, 2};
        migraphx::shape input{migraphx::shape::float_type, {1, 2}};
        throws_shape(migraphx::op::broadcast{1, lens}, input);
    }

    {
        std::vector<std::size_t> lens{3, 2, 4, 3};
        migraphx::shape input{migraphx::shape::float_type, {4, 3}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {3, 2, 4, 3}, {0, 0, 3, 1}},
                     migraphx::make_op("broadcast", {{"axis", 2}, {"dims", lens}}),
                     input);
    }

    {
        std::vector<std::size_t> lens{3, 2, 4, 3};
        migraphx::shape input{migraphx::shape::float_type, {4, 4}};
        throws_shape(migraphx::make_op("broadcast", {{"axis", 2}, {"dims", lens}}), input);
    }
}

TEST_CASE(gather)
{
    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        migraphx::shape indices{migraphx::shape::int32_type, {2, 3}};
        int axis = 1;
        expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 2, 3, 4, 5}},
                     migraphx::make_op("gather", {{"axis", axis}}),
                     input,
                     indices);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        migraphx::shape indices{migraphx::shape::int32_type, {2, 3}};
        int axis = -4;
        expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 3, 3, 4, 5}},
                     migraphx::make_op("gather", {{"axis", axis}}),
                     input,
                     indices);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        migraphx::shape indices{migraphx::shape::int32_type, {1}};
        int axis = -4;
        expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 3, 4, 5}},
                     migraphx::make_op("gather", {{"axis", axis}}),
                     input,
                     indices);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        migraphx::shape indices{migraphx::shape::int32_type};
        int axis = -4;
        expect_shape(migraphx::shape{migraphx::shape::float_type, {3, 4, 5}},
                     migraphx::make_op("gather", {{"axis", axis}}),
                     input,
                     indices);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        migraphx::shape indices{migraphx::shape::int32_type};
        int axis = 3;
        expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 3, 4}},
                     migraphx::make_op("gather", {{"axis", axis}}),
                     input,
                     indices);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {3}};
        migraphx::shape indices{migraphx::shape::int32_type};
        int axis = 0;
        expect_shape(migraphx::shape{migraphx::shape::float_type},
                     migraphx::make_op("gather", {{"axis", axis}}),
                     input,
                     indices);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {3}};
        migraphx::shape indices{migraphx::shape::int32_type, {1}};
        int axis = 0;
        expect_shape(migraphx::shape{migraphx::shape::float_type, {1}},
                     migraphx::make_op("gather", {{"axis", axis}}),
                     input,
                     indices);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        migraphx::shape indices{migraphx::shape::int32_type, {2, 3}};
        int axis = 4;
        throws_shape(migraphx::make_op("gather", {{"axis", axis}}), input, indices);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        migraphx::shape indices{migraphx::shape::int32_type, {2, 3}};
        int axis = -5;
        throws_shape(migraphx::make_op("gather", {{"axis", axis}}), input, indices);
    }
}

template <class T>
void test_softmax_variations()
{
    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}}, T{0}, input);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}}, T{1}, input);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}}, T{2}, input);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}}, T{3}, input);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        int axis = 4;
        throws_shape(T{axis}, input);
    }
}

TEST_CASE(softmax) { test_softmax_variations<migraphx::op::softmax>(); }

TEST_CASE(logsoftmax) { test_softmax_variations<migraphx::op::logsoftmax>(); }

TEST_CASE(test_argmax)
{
    {
        migraphx::shape input{migraphx::shape::half_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::int64_type, {1, 3, 4, 5}},
                     migraphx::make_op("argmax", {{"axis", 0}}),
                     input);
    }

    {
        migraphx::shape input{migraphx::shape::half_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::int64_type, {2, 1, 4, 5}},
                     migraphx::make_op("argmax", {{"axis", 1}}),
                     input);
    }

    {
        migraphx::shape input{migraphx::shape::half_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::int64_type, {2, 3, 1, 5}},
                     migraphx::make_op("argmax", {{"axis", 2}}),
                     input);
    }

    {
        migraphx::shape input{migraphx::shape::half_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::int64_type, {2, 3, 4, 1}},
                     migraphx::make_op("argmax", {{"axis", 3}}),
                     input);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        throws_shape(migraphx::make_op("argmax", {{"axis", 4}}), input);
    }
}

TEST_CASE(test_argmin)
{
    {
        migraphx::shape input{migraphx::shape::half_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::int64_type, {1, 3, 4, 5}},
                     migraphx::make_op("argmin", {{"axis", 0}}),
                     input);
    }

    {
        migraphx::shape input{migraphx::shape::half_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::int64_type, {2, 1, 4, 5}},
                     migraphx::make_op("argmin", {{"axis", 1}}),
                     input);
    }

    {
        migraphx::shape input{migraphx::shape::half_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::int64_type, {2, 3, 1, 5}},
                     migraphx::make_op("argmin", {{"axis", 2}}),
                     input);
    }

    {
        migraphx::shape input{migraphx::shape::half_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::int64_type, {2, 3, 4, 1}},
                     migraphx::make_op("argmin", {{"axis", 3}}),
                     input);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        throws_shape(migraphx::make_op("argmin", {{"axis", 4}}), input);
    }
}

TEST_CASE(test_scalar)
{
    migraphx::shape s1{migraphx::shape::float_type, {1}, {1}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 3, 4, 5}, {0, 0, 0, 0}};
    expect_shape(s2, migraphx::make_op("scalar", {{"scalar_bcst_dims", {2, 3, 4, 5}}}), s1);
}

TEST_CASE(test_scalar_nelemnts)
{
    migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
    throws_shape(migraphx::make_op("scalar", {{"scalar_bcst_dims", {2, 3, 4, 5}}}), input);
}

TEST_CASE(test_squeeze)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 1, 3, 1, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 1, 3, 3}};
    expect_shape(s2, migraphx::make_op("squeeze", {{"axes", {3}}}), s1);
}

TEST_CASE(test_squeeze_negative_axis)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 1, 3, 1, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 1, 3, 3}};
    expect_shape(s2, migraphx::make_op("squeeze", {{"axes", {-2}}}), s1);
}

TEST_CASE(test_squeeze_wrong_axis)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 1, 3, 1, 3}};
    throws_shape(migraphx::make_op("squeeze", {{"axes", {0}}}), s1);
}

TEST_CASE(test_squeeze_all)
{
    migraphx::shape s1{migraphx::shape::float_type, {1}};
    migraphx::shape s2{migraphx::shape::float_type};
    expect_shape(s2, migraphx::make_op("squeeze", {{"axes", {0}}}), s1);
}

TEST_CASE(test_unsqueeze_scalar)
{
    migraphx::shape s1{migraphx::shape::float_type, {1}, {0}};
    migraphx::shape s2{migraphx::shape::float_type, {1}, {1}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {0}}}), s1);
}

TEST_CASE(test_unsqueeze_scalar_tensor1)
{
    migraphx::shape s{migraphx::shape::float_type, {4, 3, 3}, {0, 0, 0}};
    throws_shape(migraphx::make_op("unsqueeze", {{"axes", {-2}}}), s);
}

TEST_CASE(test_unsqueeze_scalar_tensor2)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 1}, {0, 0, 0}};
    throws_shape(migraphx::make_op("unsqueeze", {{"axes", {-2}}}), s);
}

TEST_CASE(test_unsqueeze)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 3, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 3, 1, 3}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {2}}}), s1);
}

TEST_CASE(test_unsqueeze_negative_axis)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 3, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 3, 1, 3}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {-2}}}), s1);
}

template <class T>
void test_reduce_ops()
{
    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 1, 1, 1}}, T{}, input);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        expect_shape(
            migraphx::shape{migraphx::shape::float_type, {1, 1, 1, 1}}, T{{0, 1, 2, 3}}, input);
    }
    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 3, 1, 1}}, T{{2, 3}}, input);
    }
    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 3, 4, 5}}, T{{0}}, input);
    }
    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 1}}, T{{-1}}, input);
    }
    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        throws_shape(T{{4}}, input);
    }
}

TEST_CASE(reduce_sum) { test_reduce_ops<migraphx::op::reduce_sum>(); }
TEST_CASE(reduce_mean) { test_reduce_ops<migraphx::op::reduce_mean>(); }

// 2 inputs arguments
TEST_CASE(matmul)
{
    {
        migraphx::shape s_m1{migraphx::shape::float_type, {5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {5}};
        throws_shape(migraphx::make_op("dot"), s_m1, s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {5, 2}};
        throws_shape(migraphx::make_op("dot"), s_m1, s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {1, 5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {5}};
        throws_shape(migraphx::make_op("dot"), s_m1, s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {1, 5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {5, 4}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 4}},
                     migraphx::make_op("dot"),
                     s_m1,
                     s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {1, 5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {4, 4}};
        throws_shape(migraphx::make_op("dot"), s_m1, s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {1, 5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {6, 5, 4}};
        throws_shape(migraphx::make_op("dot"), s_m1, s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {6, 1, 5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {6, 5, 4}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {6, 1, 4}},
                     migraphx::make_op("dot"),
                     s_m1,
                     s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {1, 6, 1, 5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {1, 6, 5, 4}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 6, 1, 4}},
                     migraphx::make_op("dot"),
                     s_m1,
                     s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {4, 5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {5, 8}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {4, 8}},
                     migraphx::make_op("dot"),
                     s_m1,
                     s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {1, 1}};
        migraphx::shape s_m2{migraphx::shape::float_type, {1, 1}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 1}},
                     migraphx::make_op("dot"),
                     s_m1,
                     s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {1, 4, 5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {1, 5, 7}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 4, 7}},
                     migraphx::make_op("dot"),
                     s_m1,
                     s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {4, 5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {1, 1, 5, 7}};
        throws_shape(migraphx::make_op("dot"), s_m1, s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {1, 1, 4, 5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {1, 2, 5, 7}};
        throws_shape(migraphx::make_op("dot"), s_m1, s_m2);
    }
}

// 3 input arguments
TEST_CASE(gemm)
{
    {
        migraphx::shape s_m1{migraphx::shape::float_type, {4, 5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {5, 8}};
        migraphx::shape s_m3{migraphx::shape::float_type, {1}};
        throws_shape(migraphx::make_op("dot"), s_m1, s_m2, s_m3);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {4, 5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {5, 8}};
        migraphx::shape s_m3{migraphx::shape::float_type, {1, 1}};
        throws_shape(migraphx::make_op("dot"), s_m1, s_m2, s_m3);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {4, 5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {5, 8}};
        migraphx::shape s_m3{migraphx::shape::float_type, {8}};
        throws_shape(migraphx::make_op("dot"), s_m1, s_m2, s_m3);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {4, 5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {5, 8}};
        migraphx::shape s_m3{migraphx::shape::float_type, {4, 1}};
        throws_shape(migraphx::make_op("dot"), s_m1, s_m2, s_m3);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {4, 6}};
        migraphx::shape s_m2{migraphx::shape::float_type, {5, 8}};
        migraphx::shape s_m3{migraphx::shape::float_type, {4, 8}};
        throws_shape(migraphx::make_op("dot"), s_m1, s_m2, s_m3);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {4, 5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {5, 8}};
        migraphx::shape s_m3{migraphx::shape::float_type, {4}};
        throws_shape(migraphx::make_op("dot"), s_m1, s_m2, s_m3);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {4, 5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {5, 8}};
        migraphx::shape s_m3{migraphx::shape::float_type, {4, 8}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {4, 8}},
                     migraphx::make_op("dot"),
                     s_m1,
                     s_m2,
                     s_m3);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {1, 4, 5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {1, 5, 8}};
        migraphx::shape s_m3{migraphx::shape::float_type, {1, 4, 8}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 4, 8}},
                     migraphx::make_op("dot"),
                     s_m1,
                     s_m2,
                     s_m3);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {1, 4, 6}};
        migraphx::shape s_m2{migraphx::shape::float_type, {1, 5, 8}};
        migraphx::shape s_m3{migraphx::shape::float_type, {1, 4, 8}};
        throws_shape(migraphx::make_op("dot"), s_m1, s_m2, s_m3);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {1, 4, 5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {1, 5, 8}};
        migraphx::shape s_m3{migraphx::shape::float_type, {4, 8}};
        throws_shape(migraphx::make_op("dot"), s_m1, s_m2, s_m3);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {1, 4, 5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {1, 5, 8}};
        migraphx::shape s_m3{migraphx::shape::float_type};
        throws_shape(migraphx::make_op("dot"), s_m1, s_m2, s_m3);
    }
}

// quant_dot
TEST_CASE(quant_dot_2args)
{
    {
        migraphx::shape s_m1{migraphx::shape::int8_type, {2, 4}};
        migraphx::shape s_m2{migraphx::shape::int8_type, {4, 8}};
        expect_shape(migraphx::shape{migraphx::shape::int32_type, {2, 8}},
                     migraphx::make_op("quant_dot"),
                     s_m1,
                     s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::int8_type, {3, 8}};
        migraphx::shape s_m2{migraphx::shape::int8_type, {8, 7}};
        expect_shape(migraphx::shape{migraphx::shape::int32_type, {3, 7}},
                     migraphx::make_op("quant_dot", {{"alpha", 1}, {"beta", 0}}),
                     s_m1,
                     s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::int8_type, {2, 4}};
        migraphx::shape s_m2{migraphx::shape::int8_type, {8, 8}};
        throws_shape(migraphx::make_op("quant_dot"), s_m1, s_m2);
    }
}

TEST_CASE(quant_dot_3args)
{
    {
        migraphx::shape s_m1{migraphx::shape::int8_type, {2, 4}};
        migraphx::shape s_m2{migraphx::shape::int8_type, {4, 8}};
        migraphx::shape s_m3{migraphx::shape::int32_type, {2, 8}};
        expect_shape(migraphx::shape{migraphx::shape::int32_type, {2, 8}},
                     migraphx::make_op("quant_dot"),
                     s_m1,
                     s_m2,
                     s_m3);
    }

    {
        migraphx::shape s_m1{migraphx::shape::int8_type, {2, 4}};
        migraphx::shape s_m2{migraphx::shape::int8_type, {4, 8}};
        migraphx::shape s_m3{migraphx::shape::int8_type, {2, 8}};
        throws_shape(migraphx::make_op("quant_dot", {{"alpha", 1}, {"beta", 2}}), s_m1, s_m2, s_m3);
    }
}

TEST_CASE(rnn)
{
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};

        expect_shape(
            migraphx::shape{migraphx::shape::float_type,
                            {seq_len, num_dirct, batch_size, hidden_size}},
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};

        expect_shape(
            migraphx::shape{migraphx::shape::float_type,
                            {seq_len, num_dirct, batch_size, hidden_size}},
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};

        expect_shape(
            migraphx::shape{migraphx::shape::float_type,
                            {seq_len, num_dirct, batch_size, hidden_size}},
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};

        throws_shape(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size + 1},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};

        throws_shape(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};

        throws_shape(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }
}

TEST_CASE(gru)
{
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        expect_shape(
            migraphx::shape{migraphx::shape::float_type,
                            {seq_len, num_dirct, batch_size, hidden_size}},
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        expect_shape(
            migraphx::shape{migraphx::shape::float_type,
                            {seq_len, num_dirct, batch_size, hidden_size}},
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        expect_shape(
            migraphx::shape{migraphx::shape::float_type,
                            {seq_len, num_dirct, batch_size, hidden_size}},
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        throws_shape(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size + 1},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        throws_shape(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        throws_shape(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }
}

TEST_CASE(lstm)
{
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};

        expect_shape(
            migraphx::shape{migraphx::shape::float_type,
                            {seq_len, num_dirct, batch_size, hidden_size}},
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        expect_shape(
            migraphx::shape{migraphx::shape::float_type,
                            {seq_len, num_dirct, batch_size, hidden_size}},
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        expect_shape(
            migraphx::shape{migraphx::shape::float_type,
                            {seq_len, num_dirct, batch_size, hidden_size}},
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        throws_shape(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size + 1},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        throws_shape(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        throws_shape(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
