
#ifndef MIGRAPHX_GUARD_TEST_ONNX_ONNX_TEST_UTILS_HPP
#define MIGRAPHX_GUARD_TEST_ONNX_ONNX_TEST_UTILS_HPP


#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/common.hpp>
#include <migraphx/env.hpp>

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_CK_WORKAROUNDS);

inline void add_celu_instruction(migraphx::module* mm, const migraphx::shape& s, float alpha)
{
    auto x                 = mm->add_parameter("x", s);
    const auto& input_lens = s.lens();
    const auto& input_type = s.type();
    auto zero_lit =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                            mm->add_literal(migraphx::literal{migraphx::shape{input_type}, {0.}}));
    auto one_lit =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                            mm->add_literal(migraphx::literal{migraphx::shape{input_type}, {1.}}));
    auto alpha_lit = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
        mm->add_literal(migraphx::literal{migraphx::shape{input_type}, {alpha}}));
    auto linear_part = mm->add_instruction(migraphx::make_op("max"), zero_lit, x);
    auto divi        = mm->add_instruction(migraphx::make_op("div"), x, alpha_lit);
    auto expo        = mm->add_instruction(migraphx::make_op("exp"), divi);
    auto sub         = mm->add_instruction(migraphx::make_op("sub"), expo, one_lit);
    auto mul         = mm->add_instruction(migraphx::make_op("mul"), alpha_lit, sub);
    auto exp_part    = mm->add_instruction(migraphx::make_op("min"), zero_lit, mul);
    mm->add_instruction(migraphx::make_op("add"), linear_part, exp_part);
}

inline std::vector<double> make_r_eyelike(size_t num_rows, size_t num_cols, size_t k)
{
    std::vector<double> eyelike_mat(num_rows * num_cols, 0);
    for(size_t i = 0; i < num_rows; ++i)
    {
        if(i + k < num_cols)
            eyelike_mat[(num_cols + 1) * i + k] = 1.;
    }
    return eyelike_mat;
}

inline migraphx::program make_dequantizelinear_axis_prog()
{
    migraphx::program p;
    std::vector<size_t> input_lens{1, 1, 5, 1};
    int axis      = 2;
    auto* mm      = p.get_main_module();
    auto l0       = mm->add_parameter("0", {migraphx::shape::int8_type, input_lens});
    auto l1       = mm->add_parameter("1", {migraphx::shape::float_type, {5}});
    auto l2       = mm->add_parameter("2", {migraphx::shape::int8_type, {5}});
    auto l1_bcast = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", axis}, {"out_lens", input_lens}}), l1);
    auto l2_bcast = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", axis}, {"out_lens", input_lens}}), l2);
    l2_bcast = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
        l2_bcast);
    l0 = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
        l0);
    auto sub = mm->add_instruction(migraphx::make_op("sub"), l0, l2_bcast);

    mm->add_instruction(migraphx::make_op("mul"), sub, l1_bcast);
    return p;
}

inline migraphx::program create_external_data_prog()
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s(migraphx::shape::float_type, {1, 1, 224, 224});
    migraphx::shape s2(migraphx::shape::float_type, {10, 1, 11, 11});
    std::vector<float> weight_data(1210, 1);
    std::vector<float> bias_data(10, 1);
    auto bias = mm->add_literal(migraphx::literal({migraphx::shape::float_type, {10}}, bias_data));
    auto weights = mm->add_literal(migraphx::literal(s2, weight_data));
    auto param   = mm->add_parameter("input", s);
    auto conv    = mm->add_instruction(
        migraphx::make_op("convolution", {{"padding", {0, 0, 0, 0}}}), param, weights);
    auto bias_bcast = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 10, 214, 214}}}), bias);
    mm->add_instruction(migraphx::make_op("add"), conv, bias_bcast);
    return p;
}

inline migraphx::program make_group_norm(const std::vector<int64_t>& input_dims,
                                  const std::vector<int64_t>& scale_dims,
                                  const std::vector<int64_t>& bias_dims,
                                  const std::vector<int64_t>& reshape_dims,
                                  const std::vector<int64_t>& reduce_axes,
                                  const float eps_value               = 1e-5f,
                                  const migraphx::shape::type_t dtype = migraphx::shape::float_type)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x     = mm->add_parameter("x", {dtype, input_dims});
    auto scale = mm->add_parameter("scale", {dtype, scale_dims});
    auto bias  = mm->add_parameter("bias", {dtype, bias_dims});

    auto eps = mm->add_literal(migraphx::literal{dtype, {eps_value}});

    auto x_reshapedd =
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", reshape_dims}}), x);
    auto mean =
        mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", reduce_axes}}), x_reshapedd);
    auto x_sub_mean    = add_common_op(*mm, migraphx::make_op("sub"), {x_reshapedd, mean});
    auto x_sqdiff_mean = add_common_op(*mm, migraphx::make_op("sqdiff"), {x_reshapedd, mean});
    auto var     = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", reduce_axes}}),
                                   x_sqdiff_mean);
    auto var_eps = add_common_op(*mm, migraphx::make_op("add"), {var, eps});
    auto rsqrt   = mm->add_instruction(migraphx::make_op("rsqrt"), {var_eps});
    auto result  = add_common_op(*mm, migraphx::make_op("mul"), {x_sub_mean, rsqrt});
    auto scale_bcast = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", reshape_dims}}), scale);
    auto bias_bcast = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", reshape_dims}}), bias);
    auto scaled = mm->add_instruction(migraphx::make_op("mul"), {result, scale_bcast});
    auto y      = mm->add_instruction(migraphx::make_op("add"), {scaled, bias_bcast});
    mm->add_instruction(migraphx::make_op("reshape", {{"dims", input_dims}}), y);

    return p;
}

inline migraphx::program make_layer_norm(const std::vector<int64_t>& input_shape,
                                  const std::vector<int64_t>& scale_bias_shape,
                                  const std::vector<int64_t>& reduce_axes,
                                  size_t skipped_axis,
                                  bool skip_bias                      = false,
                                  const float eps_value               = 1e-5f,
                                  const migraphx::shape::type_t dtype = migraphx::shape::float_type)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto x     = mm->add_parameter("x", {dtype, input_shape});
    auto scale = mm->add_parameter("scale", {dtype, scale_bias_shape});
    migraphx::instruction_ref bias;
    if(not skip_bias)
    {
        bias = mm->add_parameter("bias", {dtype, scale_bias_shape});
    }

    auto eps = mm->add_literal(migraphx::literal{dtype, {eps_value}});

    auto mean = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", reduce_axes}}), x);
    auto x_sub_mean    = add_common_op(*mm, migraphx::make_op("sub"), {x, mean});
    auto x_sqdiff_mean = add_common_op(*mm, migraphx::make_op("sqdiff"), {x, mean});
    auto var     = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", reduce_axes}}),
                                   x_sqdiff_mean);
    auto var_eps = add_common_op(*mm, migraphx::make_op("add"), {var, eps});
    auto rsqrt   = mm->add_instruction(migraphx::make_op("rsqrt"), {var_eps});
    auto result  = add_common_op(*mm, migraphx::make_op("mul"), {x_sub_mean, rsqrt});
    migraphx::instruction_ref scale_bcast = scale;
    migraphx::instruction_ref bias_bcast  = bias;
    if(skipped_axis > 0)
    {
        scale_bcast = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", skipped_axis}, {"out_lens", input_shape}}),
            scale);
        if(not skip_bias)
        {
            bias_bcast = mm->add_instruction(
                migraphx::make_op("broadcast", {{"axis", skipped_axis}, {"out_lens", input_shape}}),
                bias);
        }
    }
    auto scaled = mm->add_instruction(migraphx::make_op("mul"), {result, scale_bcast});
    if(not skip_bias)
    {
        mm->add_instruction(migraphx::make_op("add"), {scaled, bias_bcast});
    }

    return p;
}

inline void mvn_n_rank_test(std::vector<int64_t> axes,
                     std::vector<size_t> input_shape,
                     const migraphx::program& prog)
{
    using migraphx::make_op;

    migraphx::program p;
    auto* mm = p.get_main_module();

    auto data = mm->add_parameter("data", {migraphx::shape::float_type, std::move(input_shape)});
    auto data_mean         = mm->add_instruction(make_op("reduce_mean", {{"axes", axes}}), data);
    auto data_mean_squared = add_common_op(*mm, make_op("mul"), {data_mean, data_mean});

    auto data_squared = add_common_op(*mm, make_op("mul"), {data, data});
    auto data_squared_mean =
        mm->add_instruction(make_op("reduce_mean", {{"axes", axes}}), data_squared);

    auto mean_sub = add_common_op(*mm, make_op("sub"), {data_squared_mean, data_mean_squared});
    auto std      = add_common_op(*mm, make_op("sqrt"), {mean_sub});

    auto dividend = add_common_op(*mm, make_op("sub"), {data, data_mean});
    auto epsilon  = mm->add_literal({migraphx::shape::float_type, {1e-9}});
    auto divisor  = add_common_op(*mm, make_op("add"), {std, epsilon});
    add_common_op(*mm, make_op("div"), {dividend, divisor});

    EXPECT(p == prog);
}

inline migraphx::instruction_ref insert_quantizelinear_clip(migraphx::module& m,
                                                     const migraphx::instruction_ref ins,
                                                     const migraphx::instruction_ref round,
                                                     const migraphx::shape s,
                                                     const int64_t min_quant,
                                                     const int64_t max_quant)
{
    migraphx::instruction_ref min_arg;
    migraphx::instruction_ref max_arg;
    if(migraphx::enabled(MIGRAPHX_ENABLE_CK_WORKAROUNDS{}))
    {
        std::vector<int> min_data(s.elements(), min_quant);
        std::vector<int> max_data(s.elements(), max_quant);
        min_arg = m.add_literal(migraphx::literal(s, min_data));
        max_arg = m.add_literal(migraphx::literal(s, max_data));
    }
    else
    {
        min_arg = m.add_literal(migraphx::literal{migraphx::shape{s.type()}, {min_quant}});
        max_arg = m.add_literal(migraphx::literal{migraphx::shape{s.type()}, {max_quant}});
    }

    return migraphx::insert_common_op(m, ins, migraphx::make_op("clip"), {round, min_arg, max_arg});
}

inline migraphx::program make_quantizelinear_axis_prog()
{
    migraphx::program p;
    std::vector<size_t> input_lens{1, 1, 5, 1};
    int axis = 2;
    auto* mm = p.get_main_module();

    auto l0       = mm->add_parameter("0", {migraphx::shape::float_type, input_lens});
    auto l1       = mm->add_parameter("1", {migraphx::shape::float_type, {5}});
    auto l2       = mm->add_parameter("2", {migraphx::shape::int8_type, {5}});
    auto l1_bcast = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", axis}, {"out_lens", input_lens}}), l1);

    auto div      = mm->add_instruction(migraphx::make_op("div"), l0, l1_bcast);
    auto round    = mm->add_instruction(migraphx::make_op("nearbyint"), div);
    auto l2_bcast = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", axis}, {"out_lens", input_lens}}), l2);
    l2_bcast = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
        l2_bcast);
    auto add  = mm->add_instruction(migraphx::make_op("add"), round, l2_bcast);
    auto s    = round->get_shape();
    auto clip = insert_quantizelinear_clip(*mm, div, add, s, -128, 127);
    mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::int8_type)}}),
        clip);
    return p;
}

inline auto create_upsample_linear_prog()
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape ss{migraphx::shape::float_type, {4}};
    std::vector<float> ds = {1, 1, 2, 2};
    mm->add_literal(migraphx::literal(ss, ds));

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    auto x = mm->add_parameter("X", sx);
    migraphx::shape s_ind{migraphx::shape::int32_type, {16, 1, 4, 4}};
    std::vector<int> d_ind = {
        0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 2, 2, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 2,
        2, 2, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 2, 2, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
        0, 1, 2, 2, 2, 3, 0, 0, 0, 1, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3, 0, 0, 0, 1, 2, 2, 2,
        3, 2, 2, 2, 3, 2, 2, 2, 3, 0, 0, 0, 1, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3, 0, 0, 0, 1,
        2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 2, 3, 3, 3, 0,
        1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 2, 3, 3, 3, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 2, 3,
        3, 3, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 2, 3, 3, 3, 0, 1, 1, 1, 2, 3, 3, 3, 2, 3, 3,
        3, 2, 3, 3, 3, 0, 1, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 0, 1, 1, 1, 2, 3, 3, 3,
        2, 3, 3, 3, 2, 3, 3, 3, 0, 1, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3};
    auto l_ind = mm->add_literal(migraphx::literal(s_ind, d_ind));

    migraphx::shape s8{migraphx::shape::float_type, {8, 1, 4, 4}};
    std::vector<float> d8 = {
        0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0,
        0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0,
        0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0,
        0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0,
        0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0,
        0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0,
        0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0,
        0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0,
        0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0,
        0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0,
        0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0};
    auto l8 = mm->add_literal(migraphx::literal(s8, d8));

    migraphx::shape s4{migraphx::shape::float_type, {4, 1, 4, 4}};
    std::vector<float> d4 = {
        0,        0,        0,        0,        1.0f / 3, 1.0f / 3, 1.0f / 3, 1.0f / 3,
        2.0f / 3, 2.0f / 3, 2.0f / 3, 2.0f / 3, 0,        0,        0,        0,
        0,        0,        0,        0,        1.0f / 3, 1.0f / 3, 1.0f / 3, 1.0f / 3,
        2.0f / 3, 2.0f / 3, 2.0f / 3, 2.0f / 3, 0,        0,        0,        0,
        0,        0,        0,        0,        1.0f / 3, 1.0f / 3, 1.0f / 3, 1.0f / 3,
        2.0f / 3, 2.0f / 3, 2.0f / 3, 2.0f / 3, 0,        0,        0,        0,
        0,        0,        0,        0,        1.0f / 3, 1.0f / 3, 1.0f / 3, 1.0f / 3,
        2.0f / 3, 2.0f / 3, 2.0f / 3, 2.0f / 3, 0,        0,        0,        0};
    auto l4 = mm->add_literal(migraphx::literal(s4, d4));

    migraphx::shape s2{migraphx::shape::float_type, {2, 1, 4, 4}};
    std::vector<float> d2(32, 0);
    auto l2 = mm->add_literal(migraphx::literal(s2, d2));

    migraphx::shape s1{migraphx::shape::float_type, {1, 1, 4, 4}};
    std::vector<float> d1(16, 0.0f);
    auto l1 = mm->add_literal(migraphx::literal(s1, d1));

    mm->add_instruction(migraphx::make_op("undefined"));
    auto rsp   = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4}}}), x);
    auto data  = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), rsp, l_ind);
    auto slc80 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {8}}}), data);
    auto slc81 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {8}}, {"ends", {16}}}), data);
    auto diff8 = mm->add_instruction(migraphx::make_op("sub"), slc81, slc80);
    auto mul8  = mm->add_instruction(migraphx::make_op("mul"), diff8, l8);
    auto add8  = mm->add_instruction(migraphx::make_op("add"), mul8, slc80);
    auto slc40 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {4}}}), add8);
    auto slc41 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {4}}, {"ends", {8}}}), add8);
    auto diff4 = mm->add_instruction(migraphx::make_op("sub"), slc41, slc40);
    auto mul4  = mm->add_instruction(migraphx::make_op("mul"), diff4, l4);
    auto add4  = mm->add_instruction(migraphx::make_op("add"), mul4, slc40);
    auto slc20 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {2}}}), add4);
    auto slc21 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {4}}}), add4);
    auto diff2 = mm->add_instruction(migraphx::make_op("sub"), slc21, slc20);
    auto mul2  = mm->add_instruction(migraphx::make_op("mul"), diff2, l2);
    auto add2  = mm->add_instruction(migraphx::make_op("add"), mul2, slc20);
    auto slc10 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), add2);
    auto slc11 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), add2);
    auto diff1 = mm->add_instruction(migraphx::make_op("sub"), slc11, slc10);
    auto mul1  = mm->add_instruction(migraphx::make_op("mul"), diff1, l1);
    auto add1  = mm->add_instruction(migraphx::make_op("add"), mul1, slc10);
    mm->add_return({add1});

    return p;
}

// the ScatterElements op has 3 reduction modes, which map to separate reference ops
inline migraphx::program create_scatter_program(const std::string& scatter_mode, int axis)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0 = mm->add_parameter("data", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto l1 =
        mm->add_parameter("indices", migraphx::shape{migraphx::shape::int32_type, {2, 3, 4, 5}});
    auto l2 =
        mm->add_parameter("update", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto r = mm->add_instruction(migraphx::make_op(scatter_mode, {{"axis", axis}}), l0, l1, l2);
    mm->add_return({r});
    return p;
}


#endif
