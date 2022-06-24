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
#include "migraphx/instruction_ref.hpp"
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/lowering.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/allocation_model.hpp>
#include <migraphx/apply_alpha_beta.hpp>
#include <migraphx/adjust_allocation.hpp>
#include <migraphx/gpu/pack_int8_args.hpp>
#include <migraphx/gpu/rocblas.hpp>
#include <migraphx/auto_contiguous.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/replace_allocate.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/make_op.hpp>
#include <test.hpp>

void run_passes(migraphx::module& m)
{
    auto ctx = migraphx::gpu::context{};
    migraphx::run_passes(m,
                         {migraphx::auto_contiguous{},
                          migraphx::gpu::lowering{&ctx, false},
                          migraphx::dead_code_elimination{},
                          migraphx::replace_allocate{migraphx::gpu::gpu_allocation_model{}},
                          migraphx::dead_code_elimination{},
                          migraphx::gpu::pack_int8_args{},
                          migraphx::dead_code_elimination{}});
}

bool get_int8_x4_format()
{
    bool int8_x4_format = true;
#if ROCBLAS_VERSION_MAJOR >= 2 && ROCBLAS_VERSION_MINOR >= 38
    auto ctx = migraphx::gpu::context{};
    rocblas_gemm_flags flag;
    rocblas_query_int8_layout_flag(ctx.get_stream().get_rocblas(), &flag);
    int8_x4_format = (flag == rocblas_gemm_flags_pack_int8x4);
#endif
    return int8_x4_format;
}

TEST_CASE(quant_dot)
{
    auto create_module = [] {
        migraphx::module m("test");
        migraphx::shape m1_shape{migraphx::shape::int8_type, {5, 8}};
        migraphx::shape m2_shape{migraphx::shape::int8_type, {8, 7}};
        migraphx::shape m3_shape{migraphx::shape::int32_type, {5, 7}};

        auto l1 = m.add_parameter("a", m1_shape);
        auto l2 = m.add_parameter("b", m2_shape);
        auto l3 = m.add_parameter("c", m3_shape);
        auto r =
            migraphx::add_apply_alpha_beta(m, {l1, l2, l3}, migraphx::make_op("quant_dot"), 1, 1);
        m.add_return({r});
        return m;
    };

    auto create_optimized_int8_x4 = [](bool int8_x4) {
        migraphx::module m("test");
        migraphx::shape m1_shape{migraphx::shape::int8_type, {5, 8}};
        migraphx::shape m2_shape{migraphx::shape::int8_type, {8, 7}};
        migraphx::shape m3_shape{migraphx::shape::int32_type, {5, 7}};

        auto l1         = m.add_parameter("a", m1_shape);
        auto l2         = m.add_parameter("b", m2_shape);
        auto l3         = m.add_parameter("c", m3_shape);
        auto beta       = m.add_literal(1);
        auto output     = m.add_parameter("test:#output_0", m3_shape);
        auto gemm_alloc = m.add_instruction(
            migraphx::make_op("hip::allocate", {{"shape", migraphx::to_value(m3_shape)}}));

        auto packa = l2;
        if(int8_x4)
        {
            auto alloc = m.add_instruction(
                migraphx::make_op("hip::allocate", {{"shape", migraphx::to_value(m2_shape)}}));
            packa = m.add_instruction(migraphx::make_op("gpu::int8_gemm_pack_a"), l2, alloc);
        }
        auto gemm =
            m.add_instruction(migraphx::make_op("gpu::quant_gemm", {{"int8_x4_format", int8_x4}}),
                              l1,
                              packa,
                              gemm_alloc);

        auto beta_broadcast = m.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", m3_shape.lens()}}), beta);
        auto beta_alloc = m.add_instruction(
            migraphx::make_op("hip::allocate", {{"shape", migraphx::to_value(m3_shape)}}));
        auto beta_contiguous =
            m.add_instruction(migraphx::make_op("gpu::contiguous"), beta_broadcast, beta_alloc);
        auto mul_alloc = m.add_instruction(
            migraphx::make_op("hip::allocate", {{"shape", migraphx::to_value(m3_shape)}}));
        auto m3_beta =
            m.add_instruction(migraphx::make_op("gpu::mul"), l3, beta_contiguous, mul_alloc);
        auto gemm_add = m.add_instruction(migraphx::make_op("gpu::add"), gemm, m3_beta, output);
        m.add_return({gemm_add});

        return m;
    };

    auto m1 = create_module();
    run_passes(m1);

    bool flag = get_int8_x4_format();
    auto m2   = create_optimized_int8_x4(flag);
    EXPECT(m1 == m2);
}

TEST_CASE(quant_dot_trans)
{
    auto create_module = [] {
        migraphx::module m("test");
        migraphx::shape s1{migraphx::shape::int8_type, {3, 2, 8, 5}};
        migraphx::shape s2{migraphx::shape::int8_type, {3, 2, 7, 8}};

        auto l1 = m.add_parameter("a", s1);
        auto tl1 =
            m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), l1);
        auto l2 = m.add_parameter("b", s2);
        auto tl2 =
            m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), l2);
        auto r = migraphx::add_apply_alpha_beta(m, {tl1, tl2}, migraphx::make_op("quant_dot"), 3);
        m.add_return({r});
        return m;
    };

    auto create_optimized_int8_x4 = [](bool int8_x4) {
        migraphx::module m("test");
        migraphx::shape s1{migraphx::shape::int8_type, {3, 2, 8, 5}};
        migraphx::shape s2{migraphx::shape::int8_type, {3, 2, 7, 8}};
        migraphx::shape s3{migraphx::shape::int32_type, {3, 2, 5, 7}};

        auto l1     = m.add_parameter("a", s1);
        auto l2     = m.add_parameter("b", s2);
        auto alpha  = m.add_literal(3);
        auto output = m.add_parameter("test:#output_0", s3);

        auto tl1 =
            m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), l1);
        migraphx::shape ts1{migraphx::shape::int8_type, {3, 2, 5, 8}};
        auto alloca = m.add_instruction(
            migraphx::make_op("hip::allocate", {{"shape", migraphx::to_value(ts1)}}));
        auto conta = m.add_instruction(migraphx::make_op("gpu::contiguous"), tl1, alloca);

        auto tl2 =
            m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), l2);
        migraphx::shape ts2{migraphx::shape::int8_type, {3, 2, 8, 7}};
        auto allocb = m.add_instruction(
            migraphx::make_op("hip::allocate", {{"shape", migraphx::to_value(ts2)}}));
        auto contb = m.add_instruction(migraphx::make_op("gpu::contiguous"), tl2, allocb);

        auto alpha_broadcast = m.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", conta->get_shape().lens()}}), alpha);
        auto alpha_alloc = m.add_instruction(migraphx::make_op(
            "hip::allocate",
            {{"shape",
              migraphx::to_value(migraphx::shape(migraphx::shape::int32_type, {3, 2, 5, 8}))}}));
        auto alpha_contiguous =
            m.add_instruction(migraphx::make_op("gpu::contiguous"), alpha_broadcast, alpha_alloc);
        // alpha = int32 and tl1 = int8, convert tl1 to int32 for multiplication and then convert
        // back result to int8
        auto tl1_convert_alloc = m.add_instruction(migraphx::make_op(
            "hip::allocate", {{"shape", migraphx::to_value(alpha_contiguous->get_shape())}}));
        auto tl1_convert       = m.add_instruction(
            migraphx::make_op("gpu::convert", {{"target_type", alpha->get_shape().type()}}),
            conta,
            tl1_convert_alloc);
        auto mul_alloc       = m.add_instruction(migraphx::make_op(
            "hip::allocate", {{"shape", migraphx::to_value(tl1_convert->get_shape())}}));
        auto tl1_alpha_int32 = m.add_instruction(
            migraphx::make_op("gpu::mul"), alpha_contiguous, tl1_convert, mul_alloc);
        // convert mul_res to int8
        auto tl1_alpha_int8_alloc = m.add_instruction(migraphx::make_op(
            "hip::allocate", {{"shape", migraphx::to_value(conta->get_shape())}}));
        auto tl1_alpha_int8       = m.add_instruction(
            migraphx::make_op("gpu::convert", {{"target_type", conta->get_shape().type()}}),
            tl1_alpha_int32,
            tl1_alpha_int8_alloc);

        auto packb = contb;
        if(int8_x4)
        {
            auto allocpb = m.add_instruction(
                migraphx::make_op("hip::allocate", {{"shape", migraphx::to_value(ts2)}}));
            packb = m.add_instruction(migraphx::make_op("gpu::int8_gemm_pack_a"), contb, allocpb);
        }

        auto gemm =
            m.add_instruction(migraphx::make_op("gpu::quant_gemm", {{"int8_x4_format", int8_x4}}),
                              tl1_alpha_int8,
                              packb,
                              output);
        m.add_return({gemm});

        return m;
    };

    auto m1   = create_module();
    bool flag = get_int8_x4_format();
    auto m2   = create_optimized_int8_x4(flag);

    run_passes(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(quant_dot_pad)
{
    auto create_module = [] {
        migraphx::module m("test");
        migraphx::shape s1{migraphx::shape::int8_type, {5, 6}};
        migraphx::shape s2{migraphx::shape::int8_type, {6, 7}};
        migraphx::shape s3{migraphx::shape::int32_type, {5, 7}};

        auto l1 = m.add_parameter("a", s1);
        auto l2 = m.add_parameter("b", s2);
        auto l3 = m.add_parameter("c", s3);
        auto r =
            migraphx::add_apply_alpha_beta(m, {l1, l2, l3}, migraphx::make_op("quant_dot"), 1, 1);
        m.add_return({r});
        return m;
    };

    auto create_optimized_int8_x4 = [](bool int8_x4) {
        migraphx::module m("test");
        migraphx::shape s1{migraphx::shape::int8_type, {5, 6}};
        migraphx::shape ps1{migraphx::shape::int8_type, {5, 8}};
        migraphx::shape s2{migraphx::shape::int8_type, {6, 7}};
        migraphx::shape ps2{migraphx::shape::int8_type, {8, 7}};
        migraphx::shape s3{migraphx::shape::int32_type, {5, 7}};

        auto l1     = m.add_parameter("a", s1);
        auto l2     = m.add_parameter("b", s2);
        auto l3     = m.add_parameter("c", s3);
        auto beta   = m.add_literal(1);
        auto output = m.add_parameter("test:#output_0", s3);

        auto pl1   = l1;
        auto packa = l2;
        migraphx::instruction_ref pl2{};
        if(int8_x4)
        {
            auto po1 = m.insert_instruction(
                l1, migraphx::make_op("hip::allocate", {{"shape", migraphx::to_value(ps1)}}));
            pl1 = m.add_instruction(
                migraphx::make_op("gpu::pad", {{"mode", 0}, {"pads", {0, 2, 0, 0}}, {"value", 0}}),
                l1,
                po1);

            auto po2 = m.insert_instruction(
                l2, migraphx::make_op("hip::allocate", {{"shape", migraphx::to_value(ps2)}}));
            pl2 = m.insert_instruction(
                std::next(l2),
                migraphx::make_op("gpu::pad", {{"mode", 0}, {"pads", {2, 0, 0, 0}}, {"value", 0}}),
                l2,
                po2);
        }

        auto gemm_alloc = m.add_instruction(
            migraphx::make_op("hip::allocate", {{"shape", migraphx::to_value(s3)}}));

        if(int8_x4)
        {
            auto alloc = m.add_instruction(
                migraphx::make_op("hip::allocate", {{"shape", migraphx::to_value(ps2)}}));
            packa = m.add_instruction(migraphx::make_op("gpu::int8_gemm_pack_a"), pl2, alloc);
        }

        auto gemm =
            m.add_instruction(migraphx::make_op("gpu::quant_gemm", {{"int8_x4_format", int8_x4}}),
                              pl1,
                              packa,
                              gemm_alloc);

        auto beta_broadcast =
            m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s3.lens()}}), beta);
        auto beta_alloc = m.add_instruction(
            migraphx::make_op("hip::allocate", {{"shape", migraphx::to_value(s3)}}));
        auto beta_contiguous =
            m.add_instruction(migraphx::make_op("gpu::contiguous"), beta_broadcast, beta_alloc);
        auto mul_alloc = m.add_instruction(
            migraphx::make_op("hip::allocate", {{"shape", migraphx::to_value(s3)}}));
        auto m3_beta =
            m.add_instruction(migraphx::make_op("gpu::mul"), l3, beta_contiguous, mul_alloc);
        auto gemm_add = m.add_instruction(migraphx::make_op("gpu::add"), gemm, m3_beta, output);
        m.add_return({gemm_add});
        return m;
    };

    auto m1   = create_module();
    bool flag = get_int8_x4_format();
    auto m2   = create_optimized_int8_x4(flag);

    run_passes(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(quant_dot_trans_pad)
{
    auto create_module = [] {
        migraphx::module m("test");
        migraphx::shape s1{migraphx::shape::int8_type, {3, 2, 9, 5}};
        migraphx::shape s2{migraphx::shape::int8_type, {3, 2, 7, 9}};

        auto l1 = m.add_parameter("a", s1);
        auto tl1 =
            m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), l1);
        auto l2 = m.add_parameter("b", s2);
        auto tl2 =
            m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), l2);
        auto r = migraphx::add_apply_alpha_beta(m, {tl1, tl2}, migraphx::make_op("quant_dot"), 3);
        m.add_return({r});
        return m;
    };

    auto create_optimized_int8_x4 = [](bool int8_x4) {
        migraphx::module m("test");
        migraphx::shape s1{migraphx::shape::int8_type, {3, 2, 9, 5}};
        migraphx::shape ps1{migraphx::shape::int8_type, {3, 2, 5, 12}};
        migraphx::shape s2{migraphx::shape::int8_type, {3, 2, 7, 9}};
        migraphx::shape ps2{migraphx::shape::int8_type, {3, 2, 12, 7}};
        migraphx::shape s3{migraphx::shape::int32_type, {3, 2, 5, 7}};

        auto l1     = m.add_parameter("a", s1);
        auto l2     = m.add_parameter("b", s2);
        auto alpha  = m.add_literal(3);
        auto output = m.add_parameter("test:#output_0", s3);

        auto tl1 =
            m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), l1);
        migraphx::shape ts1{migraphx::shape::int8_type, {3, 2, 5, 9}};
        auto ta = m.add_instruction(
            migraphx::make_op("hip::allocate", {{"shape", migraphx::to_value(ts1)}}));
        auto conta = m.add_instruction(migraphx::make_op("gpu::contiguous"), tl1, ta);

        auto tl2 =
            m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), l2);
        migraphx::shape ts2{migraphx::shape::int8_type, {3, 2, 9, 7}};
        auto tb = m.add_instruction(
            migraphx::make_op("hip::allocate", {{"shape", migraphx::to_value(ts2)}}));

        migraphx::instruction_ref ptb{};
        if(int8_x4)
        {
            ptb = m.add_instruction(
                migraphx::make_op("hip::allocate", {{"shape", migraphx::to_value(ps2)}}));
        }
        auto contb = m.add_instruction(migraphx::make_op("gpu::contiguous"), tl2, tb);
        auto pb    = contb;
        if(int8_x4)
        {
            pb = m.add_instruction(
                migraphx::make_op("gpu::pad", {{"mode", 0}, {"pads", {0, 0, 3, 0, 0, 0, 0, 0}}}),
                contb,
                ptb);
        }

        auto alpha_broadcast = m.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", conta->get_shape().lens()}}), alpha);
        auto alpha_alloc = m.add_instruction(
            migraphx::make_op("hip::allocate",
                              {{"shape",
                                migraphx::to_value(migraphx::shape(migraphx::shape::int32_type,
                                                                   conta->get_shape().lens()))}}));
        auto alpha_contiguous =
            m.add_instruction(migraphx::make_op("gpu::contiguous"), alpha_broadcast, alpha_alloc);

        // alpha = int32 and tl1 = int8, convert tl1 to int32 for multiplication and then convert
        // back result to int8
        auto tl1_convert_alloc = m.add_instruction(migraphx::make_op(
            "hip::allocate", {{"shape", migraphx::to_value(alpha_contiguous->get_shape())}}));
        auto tl1_convert       = m.add_instruction(
            migraphx::make_op("gpu::convert", {{"target_type", alpha->get_shape().type()}}),
            conta,
            tl1_convert_alloc);
        auto mul_alloc       = m.add_instruction(migraphx::make_op(
            "hip::allocate", {{"shape", migraphx::to_value(tl1_convert->get_shape())}}));
        auto tl1_alpha_int32 = m.add_instruction(
            migraphx::make_op("gpu::mul"), alpha_contiguous, tl1_convert, mul_alloc);
        // convert mul_res to int8
        auto tl1_alpha_int8_alloc = m.add_instruction(migraphx::make_op(
            "hip::allocate", {{"shape", migraphx::to_value(conta->get_shape())}}));

        migraphx::instruction_ref pta{};
        if(int8_x4)
        {
            pta = m.add_instruction(
                migraphx::make_op("hip::allocate", {{"shape", migraphx::to_value(ps1)}}));
        }

        auto tl1_alpha_int8 = m.add_instruction(
            migraphx::make_op("gpu::convert", {{"target_type", conta->get_shape().type()}}),
            tl1_alpha_int32,
            tl1_alpha_int8_alloc);

        auto pa = tl1_alpha_int8;
        if(int8_x4)
        {
            pa = m.add_instruction(
                migraphx::make_op("gpu::pad", {{"mode", 0}, {"pads", {0, 0, 0, 3, 0, 0, 0, 0}}}),
                tl1_alpha_int8,
                pta);
        }

        auto packb = pb;
        if(int8_x4)
        {
            auto allocpb = m.add_instruction(
                migraphx::make_op("hip::allocate", {{"shape", migraphx::to_value(ps2)}}));
            packb = m.add_instruction(migraphx::make_op("gpu::int8_gemm_pack_a"), pb, allocpb);
        }

        auto gemm = m.add_instruction(
            migraphx::make_op("gpu::quant_gemm", {{"int8_x4_format", int8_x4}}), pa, packb, output);
        m.add_return({gemm});

        return m;
    };

    auto m1   = create_module();
    bool flag = get_int8_x4_format();
    auto m2   = create_optimized_int8_x4(flag);

    run_passes(m1);

    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
