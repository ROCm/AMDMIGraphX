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
 *
 */
#include <migraphx/shape_transform_descriptor.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <algorithm>
#include <test.hpp>

using migraphx::make_op;
using migraphx::shape_transform_descriptor;
using all_lens   = std::vector<std::vector<std::size_t>>;
using final_lens = std::vector<std::size_t>;
using all_axes   = std::vector<std::vector<std::vector<std::size_t>>>;
using d_axes     = std::vector<std::vector<std::size_t>>;
using ops        = std::vector<migraphx::operation>;
using dimension  = shape_transform_descriptor::dimension;
using sub        = dimension::sub;
using axes_map   = std::vector<std::vector<std::size_t>>;

static all_lens get_all_lens(const shape_transform_descriptor& d)
{
    all_lens result;
    std::transform(
        d.dimensions.begin(), d.dimensions.end(), std::back_inserter(result), [](const auto& dim) {
            std::vector<std::size_t> sub_lens;
            std::transform(dim.subdimensions.begin(),
                           dim.subdimensions.end(),
                           std::back_inserter(sub_lens),
                           [](const auto& x) { return x.len; });
            return sub_lens;
        });
    return result;
}

static final_lens get_final_lens(const shape_transform_descriptor& d)
{
    final_lens result;
    std::transform(d.dimensions.begin(),
                   d.dimensions.end(),
                   std::back_inserter(result),
                   [](const auto& x) { return x.len(); });
    return result;
}

static all_axes get_all_axes(const shape_transform_descriptor& d)
{
    all_axes result;
    std::transform(
        d.dimensions.begin(), d.dimensions.end(), std::back_inserter(result), [](const auto& dim) {
            std::vector<std::vector<std::size_t>> sub_axis;
            std::transform(dim.subdimensions.begin(),
                           dim.subdimensions.end(),
                           std::back_inserter(sub_axis),
                           [](const auto& x) { return x.axis; });
            return sub_axis;
        });
    return result;
}

static std::vector<int64_t> run_shape_transforms(const std::vector<std::size_t>& dims,
                                                 const std::vector<migraphx::operation>& ops)
{
    migraphx::shape s{migraphx::shape::int64_type, dims};
    std::vector<int64_t> data(s.elements());
    std::iota(data.begin(), data.end(), 0);

    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto start = mm->add_literal(s, data);
    for(const auto& op : ops)
        start = mm->add_instruction(op, start);
    mm->add_return({start});

    auto result = p.eval({}).at(0);
    return result.to_vector<int64_t>();
}

static std::vector<migraphx::operation>
check_optimize_shape_transforms(const std::vector<std::size_t>& dims,
                                const std::vector<migraphx::operation>& ops)
{
    auto result = migraphx::optimize_shape_transforms(dims, ops);
    CHECK(run_shape_transforms(dims, ops) == run_shape_transforms(dims, result));
    CHECK(result == migraphx::optimize_shape_transforms(dims, result));
    return result;
}

template <class... Ts>
static shape_transform_descriptor make_descriptor(const std::vector<std::size_t>& dims,
                                                  const Ts&... xs)
{
    auto desc = shape_transform_descriptor{dims};
    CHECK(desc.apply({xs...}));
    return desc;
}

template <class... Ts>
static shape_transform_descriptor make_simple_descriptor(const std::vector<std::size_t>& dims,
                                                         const Ts&... xs)
{
    auto desc = make_descriptor(dims, xs...);
    desc.simplify();
    return desc;
}

TEST_CASE(dimension_len)
{
    dimension dim;
    dim.subdimensions = std::vector<sub>{sub{4, {1}}, sub{5, {2}}};
    EXPECT(dim.len() == 20);
}

TEST_CASE(record_reshape)
{
    auto desc = make_descriptor({256, 3, 16, 16}, make_op("reshape", {{"dims", {16, 16, 48, 16}}}));
    EXPECT(get_final_lens(desc) == final_lens{16, 16, 48, 16});
    EXPECT(get_all_lens(desc) == all_lens{{16}, {16}, {3, 16}, {16}});
    EXPECT(get_all_axes(desc) ==
           all_axes{d_axes{{0, 0}}, d_axes{{0, 1}}, d_axes{{1}, {2}}, d_axes{{3}}});
}

TEST_CASE(record_reshape_1s)
{
    auto desc = make_descriptor({3, 4, 4}, make_op("reshape", {{"dims", {3, 1, 4, 1, 4}}}));
    EXPECT(get_final_lens(desc) == final_lens{3, 1, 4, 1, 4});
    EXPECT(get_all_lens(desc) == all_lens{{3}, {1}, {4}, {1}, {4}});
    EXPECT(get_all_axes(desc) ==
           all_axes{d_axes{{0}}, d_axes{{1, 0}}, d_axes{{1, 1}}, d_axes{{2, 0}}, d_axes{{2, 1}}});
}

TEST_CASE(record_reshape_trailing_1s)
{
    auto desc = make_descriptor({3, 4, 4}, make_op("reshape", {{"dims", {3, 4, 4, 1, 1}}}));
    EXPECT(get_final_lens(desc) == final_lens{3, 4, 4, 1, 1});
    EXPECT(get_all_lens(desc) == all_lens{{3}, {4}, {4}, {1}, {1}});
    EXPECT(get_all_axes(desc) ==
           all_axes{d_axes{{0}}, d_axes{{1}}, d_axes{{2, 0}}, d_axes{{2, 1}}, d_axes{{2, 2}}});
}

TEST_CASE(record_reshape_merge)
{
    auto desc = make_descriptor({3, 4, 5}, make_op("reshape", {{"dims", {3, 20}}}));
    EXPECT(get_final_lens(desc) == final_lens{3, 20});
    EXPECT(get_all_lens(desc) == all_lens{{3}, {4, 5}});
    EXPECT(get_all_axes(desc) == all_axes{d_axes{{0}}, d_axes{{1}, {2}}});
}

TEST_CASE(record_reshape_split)
{
    auto desc = make_descriptor({3, 20}, make_op("reshape", {{"dims", {3, 4, 5}}}));
    EXPECT(get_final_lens(desc) == final_lens{3, 4, 5});
    EXPECT(get_all_lens(desc) == all_lens{{3}, {4}, {5}});
    EXPECT(get_all_axes(desc) == all_axes{d_axes{{0}}, d_axes{{1, 0}}, d_axes{{1, 1}}});
}

TEST_CASE(record_reshape_merge_split)
{
    auto desc = make_descriptor({3, 10, 16}, make_op("reshape", {{"dims", {3, 40, 2, 2}}}));
    EXPECT(get_final_lens(desc) == final_lens{3, 40, 2, 2});
    EXPECT(get_all_lens(desc) == all_lens{{3}, {10, 4}, {2}, {2}});
    EXPECT(get_all_axes(desc) ==
           all_axes{d_axes{{0}}, d_axes{{1}, {2, 0}}, d_axes{{2, 1}}, d_axes{{2, 2}}});
}

TEST_CASE(record_squeeze_trailing_1s)
{
    auto desc = make_descriptor({3, 4, 4, 1, 1}, make_op("reshape", {{"dims", {3, 4, 4}}}));
    EXPECT(get_final_lens(desc) == final_lens{3, 4, 4});
    EXPECT(get_all_lens(desc) == all_lens{{3}, {4}, {4}});
    EXPECT(get_all_axes(desc) == all_axes{d_axes{{0}}, d_axes{{1}}, d_axes{{2}}});
}

TEST_CASE(record_reshape_squeeze_trailing_1s)
{
    auto desc = make_descriptor({3, 4, 4},
                                make_op("reshape", {{"dims", {3, 4, 4, 1, 1}}}),
                                make_op("reshape", {{"dims", {3, 4, 4}}}));
    EXPECT(get_final_lens(desc) == final_lens{3, 4, 4});
    EXPECT(get_all_lens(desc) == all_lens{{3}, {4}, {4}});
    EXPECT(get_all_axes(desc) == all_axes{d_axes{{0}}, d_axes{{1}}, d_axes{{2, 0}}});
}

TEST_CASE(record_reshape_non_divisible_fail)
{
    auto desc = shape_transform_descriptor{{2, 3, 5}};
    EXPECT(not desc.apply({make_op("reshape", {{"dims", {10, 3}}})}));
}

TEST_CASE(record_transpose)
{
    auto desc =
        make_descriptor({256, 3, 16, 16}, make_op("transpose", {{"permutation", {0, 2, 3, 1}}}));
    EXPECT(get_final_lens(desc) == final_lens{256, 16, 16, 3});
    EXPECT(get_all_lens(desc) == all_lens{{256}, {16}, {16}, {3}});
    EXPECT(get_all_axes(desc) == all_axes{d_axes{{0}}, d_axes{{2}}, d_axes{{3}}, d_axes{{1}}});
}

TEST_CASE(record_multibroadcast)
{
    auto desc =
        make_descriptor({1, 3, 1, 1}, make_op("multibroadcast", {{"out_lens", {256, 3, 16, 16}}}));
    EXPECT(get_final_lens(desc) == final_lens{256, 3, 16, 16});
    EXPECT(get_all_lens(desc) == all_lens{{256}, {3}, {16}, {16}});
    EXPECT(get_all_axes(desc) == all_axes{d_axes{{}}, d_axes{{1}}, d_axes{{}}, d_axes{{}}});
}

TEST_CASE(record_broadcast1)
{
    auto desc =
        make_descriptor({3}, make_op("broadcast", {{"axis", 1}, {"out_lens", {256, 3, 16, 16}}}));
    EXPECT(get_final_lens(desc) == final_lens{256, 3, 16, 16});
    EXPECT(get_all_lens(desc) == all_lens{{256}, {3}, {16}, {16}});
    EXPECT(get_all_axes(desc) == all_axes{d_axes{{}}, d_axes{{0}}, d_axes{{}}, d_axes{{}}});
}

TEST_CASE(record_broadcast2)
{
    auto desc = make_descriptor(
        {32, 10}, make_op("broadcast", {{"axis", 1}, {"out_lens", {256, 32, 10, 16, 16}}}));
    EXPECT(get_final_lens(desc) == final_lens{256, 32, 10, 16, 16});
    EXPECT(get_all_lens(desc) == all_lens{{256}, {32}, {10}, {16}, {16}});
    EXPECT(get_all_axes(desc) ==
           all_axes{d_axes{{}}, d_axes{{0}}, d_axes{{1}}, d_axes{{}}, d_axes{{}}});
}

TEST_CASE(simplify_dimension_merge_adjacent)
{
    auto d = dimension{{sub{2, {0, 0}}, sub{3, {0, 1}}}};
    d.simplify();
    EXPECT(d == dimension{{sub{6, {0, 1}}}});
}

TEST_CASE(simplify_dimension_no_merge_adjacent1)
{
    auto d = dimension{{sub{2, {0, 1}}, sub{3, {0, 0}}}};
    d.simplify();
    EXPECT(d == dimension{{sub{2, {0, 1}}, sub{3, {0, 0}}}});
}

TEST_CASE(simplify_dimension_no_merge_adjacent2)
{
    auto d = dimension{{sub{2, {0, 0, 0}}, sub{3, {0, 1, 1}}}};
    d.simplify();
    EXPECT(d == dimension{{sub{2, {0, 0, 0}}, sub{3, {0, 1, 1}}}});
}

TEST_CASE(simplify_dimension_remove_1_dim)
{
    auto d = dimension{{sub{2, {0, 1}}, sub{1, {1}}, sub{3, {0, 0}}}};
    d.simplify();
    EXPECT(d == dimension{{sub{2, {0, 1}}, sub{3, {0, 0}}}});
}

TEST_CASE(optimize_transpose_transpose)
{
    EXPECT(check_optimize_shape_transforms({3, 5, 2},
                                           {
                                               make_op("transpose", {{"permutation", {0, 2, 1}}}),
                                               make_op("transpose", {{"permutation", {1, 0, 2}}}),
                                           }) ==
           ops{
               make_op("transpose", {{"permutation", {2, 0, 1}}}),
           });
}

TEST_CASE(optimize_reshape_reshape1)
{
    EXPECT(check_optimize_shape_transforms({3, 5, 2},
                                           {
                                               make_op("reshape", {{"dims", {30}}}),
                                               make_op("reshape", {{"dims", {3, 10}}}),
                                           }) == ops{
                                                     make_op("reshape", {{"dims", {3, 10}}}),
                                                 });
}

TEST_CASE(optimize_reshape_reshape2)
{
    EXPECT(check_optimize_shape_transforms({15, 4},
                                           {
                                               make_op("reshape", {{"dims", {3, 5, 2, 2}}}),
                                               make_op("reshape", {{"dims", {15, 2, 2}}}),
                                           }) == ops{
                                                     make_op("reshape", {{"dims", {15, 2, 2}}}),
                                                 });
}

TEST_CASE(optimize_reshape_transpose_reshape_to_none)
{
    EXPECT(check_optimize_shape_transforms(
               {6, 5, 2},
               {
                   make_op("reshape", {{"dims", {6, 5, 2, 1, 1}}}),
                   make_op("transpose", {{"permutation", {0, 1, 2, 4, 3}}}),
                   make_op("reshape", {{"dims", {6, 5, 2}}}),
               }) == ops{});
}

TEST_CASE(optimize_reshape_transpose_reshape_to_same)
{
    EXPECT(check_optimize_shape_transforms(
               {1, 112, 7, 7},
               {
                   make_op("reshape", {{"dims", {1, 4, 28, 7, 7}}}),
                   make_op("transpose", {{"permutation", {0, 2, 1, 3, 4}}}),
                   make_op("reshape", {{"dims", {1, 112, 7, 7}}}),
               }) == ops{
                         make_op("reshape", {{"dims", {1, 4, 28, 7, 7}}}),
                         make_op("transpose", {{"permutation", {0, 2, 1, 3, 4}}}),
                         make_op("reshape", {{"dims", {1, 112, 7, 7}}}),
                     });
}

TEST_CASE(optimize_reshape_transpose_reshape_to_transpose)
{
    EXPECT(check_optimize_shape_transforms(
               {6, 5, 2},
               {
                   make_op("reshape", {{"dims", {2, 3, 5, 2}}}),
                   make_op("transpose", {{"permutation", {0, 1, 3, 2}}}),
                   make_op("reshape", {{"dims", {6, 2, 5}}}),
               }) == ops{
                         make_op("transpose", {{"permutation", {0, 2, 1}}}),
                     });
}

TEST_CASE(optimize_reshape_transpose_reshape_to_reshape)
{
    EXPECT(
        check_optimize_shape_transforms({6, 5, 2},
                                        {
                                            make_op("reshape", {{"dims", {6, 5, 2, 1}}}),
                                            make_op("transpose", {{"permutation", {0, 1, 3, 2}}}),
                                            make_op("reshape", {{"dims", {6, 10}}}),
                                        }) == ops{
                                                  make_op("reshape", {{"dims", {6, 10}}}),
                                              });
}

TEST_CASE(optimize_multibroadcast_transpose_reshape)
{
    EXPECT(check_optimize_shape_transforms(
               {1, 5, 2},
               {
                   make_op("multibroadcast", {{"out_lens", {20, 5, 2}}}),
                   make_op("transpose", {{"permutation", {0, 2, 1}}}),
                   make_op("reshape", {{"dims", {20, 10}}}),
               }) == ops{
                         make_op("transpose", {{"permutation", {0, 2, 1}}}),
                         make_op("reshape", {{"dims", {1, 10}}}),
                         make_op("multibroadcast", {{"out_lens", {20, 10}}}),
                     });
}

TEST_CASE(optimize_resize1)
{
    EXPECT(check_optimize_shape_transforms(
               {3, 4, 4},
               {
                   make_op("reshape", {{"dims", {3, 1, 4, 1, 4}}}),
                   make_op("multibroadcast", {{"out_lens", {3, 2, 4, 2, 4}}}),
                   make_op("reshape", {{"dims", {3, 8, 8}}}),
               }) == ops{
                         make_op("unsqueeze", {{"axes", {1, 3}}}),
                         make_op("multibroadcast", {{"out_lens", {3, 2, 4, 2, 4}}}),
                         make_op("reshape", {{"dims", {3, 8, 8}}}),
                     });
}

TEST_CASE(optimize_resize2)
{
    EXPECT(check_optimize_shape_transforms(
               {1, 1, 2, 2},
               {
                   make_op("reshape", {{"dims", {1, 1, 2, 1, 2, 1}}}),
                   make_op("multibroadcast", {{"out_lens", {1, 2, 2, 2, 2, 3}}}),
                   make_op("reshape", {{"dims", {1, 2, 4, 6}}}),
               }) == ops{
                         make_op("unsqueeze", {{"axes", {3, 5}}}),
                         make_op("multibroadcast", {{"out_lens", {1, 1, 2, 2, 2, 3}}}),
                         make_op("reshape", {{"dims", {1, 1, 4, 6}}}),
                         make_op("multibroadcast", {{"out_lens", {1, 2, 4, 6}}}),
                     });
}

TEST_CASE(optimize_reshape_2_squeeze)
{
    EXPECT(check_optimize_shape_transforms({3, 1, 5, 1, 2, 1, 1},
                                           {
                                               make_op("reshape", {{"dims", {3, 5, 2}}}),
                                           }) == ops{
                                                     make_op("squeeze", {{"axes", {1, 3, 5, 6}}}),
                                                 });
}

TEST_CASE(optimize_reshape_2_unsqueeze)
{
    EXPECT(
        check_optimize_shape_transforms({3, 5, 2},
                                        {
                                            make_op("reshape", {{"dims", {3, 1, 5, 1, 2, 1, 1}}}),
                                        }) == ops{
                                                  make_op("unsqueeze", {{"axes", {1, 3, 5, 6}}}),
                                              });
}

TEST_CASE(optimize_unsqueeze_multibroadcast)
{
    EXPECT(check_optimize_shape_transforms(
               {32, 10},
               {
                   make_op("unsqueeze", {{"axes", {0, 3, 4}}}),
                   make_op("multibroadcast", {{"out_lens", {4, 32, 10, 16, 16}}}),
               }) == ops{
                         make_op("broadcast", {{"axis", 1}, {"out_lens", {4, 32, 10, 16, 16}}}),
                     });
}

TEST_CASE(optimize_multibroadcast_reshape)
{
    EXPECT(check_optimize_shape_transforms({1, 4, 1},
                                           {
                                               make_op("multibroadcast", {{"out_lens", {2, 4, 6}}}),
                                               make_op("reshape", {{"dims", {2, 2, 2, 6}}}),
                                           }) ==
           ops{
               make_op("reshape", {{"dims", {1, 2, 2, 1}}}),
               make_op("multibroadcast", {{"out_lens", {2, 2, 2, 6}}}),
           });
}

TEST_CASE(optimize_squeeze_broadcast1)
{
    EXPECT(check_optimize_shape_transforms(
               {256, 1, 1},
               {
                   make_op("squeeze"),
                   make_op("broadcast", {{"axis", 0}, {"out_lens", {256, 64, 1, 1}}}),
               }) == ops{
                         make_op("unsqueeze", {{"axes", {3}}}),
                         make_op("multibroadcast", {{"out_lens", {256, 64, 1, 1}}}),
                     });
}

TEST_CASE(optimize_squeeze_broadcast2)
{
    EXPECT(check_optimize_shape_transforms(
               {1, 128, 1},
               {
                   make_op("squeeze", {{"axes", {0}}}),
                   make_op("multibroadcast", {{"out_lens", {128, 768}}}),
               }) == ops{
                         make_op("squeeze", {{"axes", {0}}}),
                         make_op("multibroadcast", {{"out_lens", {128, 768}}}),
                     });
}

TEST_CASE(optimize_squeeze_unsqueeze_broadcast_to_broadcast)
{
    EXPECT(check_optimize_shape_transforms(
               {256},
               {
                   make_op("unsqueeze", {{"axes", {0}}}),
                   make_op("squeeze"),
                   make_op("broadcast", {{"axis", 0}, {"out_lens", {256, 64, 1, 1}}}),
               }) == ops{
                         make_op("broadcast", {{"axis", 0}, {"out_lens", {256, 64, 1, 1}}}),
                     });
}

TEST_CASE(optimize_transpose_reshape_to_transpose)
{
    EXPECT(check_optimize_shape_transforms(
               {3, 3, 3, 1},
               {
                   make_op("transpose", {{"permutation", {3, 2, 0, 1}}}),
                   make_op("reshape", {{"dims", {3, 1, 3, 3}}}),
               }) == ops{
                         make_op("transpose", {{"permutation", {2, 3, 0, 1}}}),
                     });
}

TEST_CASE(optimize_scalar_broadcast_unsqueeze)
{
    EXPECT(check_optimize_shape_transforms({1},
                                           {
                                               make_op("multibroadcast", {{"out_lens", {2}}}),
                                               make_op("unsqueeze", {{"axes", {1}}}),
                                           }) ==
           ops{
               make_op("multibroadcast", {{"out_lens", {2, 1}}}),
           });
}

TEST_CASE(optimize_broadcast_reshape_transpose)
{
    EXPECT(check_optimize_shape_transforms(
               {2, 16, 1},
               {
                   make_op("multibroadcast", {{"out_lens", {2, 16, 10240}}}),
                   make_op("reshape", {{"dims", {2, 160, 32, 32}}}),
                   make_op("transpose", {{"permutation", {0, 2, 3, 1}}}),
               }) == ops{
                         make_op("unsqueeze", {{"axes", {3, 4}}}),
                         make_op("transpose", {{"permutation", {0, 3, 4, 1, 2}}}),
                         make_op("multibroadcast", {{"out_lens", {2, 1, 1, 16, 10}}}),
                         make_op("reshape", {{"dims", {2, 1, 1, 160}}}),
                         make_op("multibroadcast", {{"out_lens", {2, 32, 32, 160}}}),
                     });
}

TEST_CASE(optimize_multibroadcast_transpose)
{
    EXPECT(check_optimize_shape_transforms(
               {320, 1, 1},
               {
                   make_op("multibroadcast", {{"out_lens", {2, 320, 64, 64}}}),
                   make_op("transpose", {{"permutation", {0, 2, 3, 1}}}),
               }) == ops{
                         make_op("unsqueeze", {{"axes", {0}}}),
                         make_op("transpose", {{"permutation", {0, 2, 3, 1}}}),
                         make_op("multibroadcast", {{"out_lens", {2, 64, 64, 320}}}),
                     });
}

TEST_CASE(optimize_unsqueeze_transpose_squeeze_multibroadcast)
{
    EXPECT(check_optimize_shape_transforms(
               {320, 1, 1},
               {
                   make_op("unsqueeze", {{"axes", {0}}}),
                   make_op("transpose", {{"permutation", {0, 2, 1, 3}}}),
                   make_op("squeeze", {{"axes", {0, 1}}}),
                   make_op("multibroadcast", {{"out_lens", {320, 320}}}),
               }) == ops{
                         make_op("multibroadcast", {{"out_lens", {320, 1, 320}}}),
                         make_op("squeeze", {{"axes", {1}}}),
                     });
}

TEST_CASE(optimize_squeeze_multibroadcast_transpose)
{
    EXPECT(check_optimize_shape_transforms(
               {16, 1, 16},
               {
                   make_op("squeeze", {{"axes", {1}}}),
                   make_op("multibroadcast", {{"out_lens", {4, 16, 16}}}),
                   make_op("transpose", {{"permutation", {1, 0, 2}}}),
               }) == ops{
                         make_op("multibroadcast", {{"out_lens", {16, 4, 16}}}),
                     });
}

TEST_CASE(optimize_squeeze_1x1)
{
    EXPECT(check_optimize_shape_transforms({1, 1},
                                           {
                                               make_op("squeeze", {{"axes", {0}}}),
                                           }) == ops{
                                                     make_op("squeeze", {{"axes", {0}}}),
                                                 });
}

TEST_CASE(optimize_broadcast_squeeze_reshape)
{
    EXPECT(check_optimize_shape_transforms(
               {2, 32, 1, 1, 1},
               {
                   make_op("multibroadcast", {{"out_lens", {2, 32, 40960, 1, 1}}}),
                   make_op("squeeze", {{"axes", {3, 4}}}),
                   make_op("reshape", {{"dims", {2, 32, 10, 64, 64}}}),
               }) == ops{
                         make_op("multibroadcast", {{"out_lens", {2, 32, 10, 64, 64}}}),
                     });
}

TEST_CASE(common_dims_reshape_less)
{
    auto desc =
        make_simple_descriptor({2, 32, 40, 8}, make_op("reshape", {{"dims", {2, 1280, 8}}}));
    EXPECT(desc.common_dims() == final_lens{2, 32, 40, 8});
    EXPECT(desc.common_axes_map_from_src() == axes_map{{0}, {1}, {2}, {3}});
    EXPECT(desc.common_axes_map_from_dst() == axes_map{{0}, {1, 2}, {3}});
    EXPECT(desc.to_common_from_src().generate() == ops{});
    EXPECT(desc.to_common_from_dst().generate() ==
           ops{make_op("reshape", {{"dims", {2, 32, 40, 8}}})});
    EXPECT(desc.to_dst_from_common().generate() ==
           ops{make_op("reshape", {{"dims", {2, 1280, 8}}})});
    EXPECT(desc.to_src_from_common().generate() == ops{});
}

TEST_CASE(common_dims_reshape1)
{
    auto desc =
        make_simple_descriptor({2, 32, 2560}, make_op("reshape", {{"dims", {2, 1280, 8, 8}}}));
    EXPECT(desc.common_dims() == final_lens{2, 32, 40, 8, 8});
    EXPECT(desc.common_axes_map_from_src() == axes_map{{{0}, {1}, {2, 3, 4}}});
    EXPECT(desc.common_axes_map_from_dst() == axes_map{{0}, {1, 2}, {3}, {4}});
    EXPECT(desc.to_common_from_src().generate() ==
           ops{make_op("reshape", {{"dims", {2, 32, 40, 8, 8}}})});
    EXPECT(desc.to_common_from_dst().generate() ==
           ops{make_op("reshape", {{"dims", {2, 32, 40, 8, 8}}})});
    EXPECT(desc.to_dst_from_common().generate() ==
           ops{make_op("reshape", {{"dims", {2, 1280, 8, 8}}})});
    EXPECT(desc.to_src_from_common().generate() ==
           ops{make_op("reshape", {{"dims", {2, 32, 2560}}})});
}

TEST_CASE(common_dims_reshape2)
{
    auto desc =
        make_simple_descriptor({2, 1280, 8, 8}, make_op("reshape", {{"dims", {2, 32, 2560}}}));
    EXPECT(desc.common_dims() == final_lens{2, 32, 40, 8, 8});
    EXPECT(desc.common_axes_map_from_src() == axes_map{{0}, {1, 2}, {3}, {4}});
    EXPECT(desc.common_axes_map_from_dst() == axes_map{{{0}, {1}, {2, 3, 4}}});
    EXPECT(desc.to_common_from_src().generate() ==
           ops{make_op("reshape", {{"dims", {2, 32, 40, 8, 8}}})});
    EXPECT(desc.to_common_from_dst().generate() ==
           ops{make_op("reshape", {{"dims", {2, 32, 40, 8, 8}}})});
    EXPECT(desc.to_dst_from_common().generate() ==
           ops{make_op("reshape", {{"dims", {2, 32, 2560}}})});
    EXPECT(desc.to_src_from_common().generate() ==
           ops{make_op("reshape", {{"dims", {2, 1280, 8, 8}}})});
}

TEST_CASE(common_dims_reshape3)
{
    auto desc =
        make_simple_descriptor({2, 32, 4096}, make_op("reshape", {{"dims", {4, 16, 64, 64}}}));

    EXPECT(desc.common_dims() == final_lens{2, 2, 16, 64, 64});
    EXPECT(desc.common_dims({2, 1, 4096}) == final_lens{2, 1, 1, 64, 64});
    EXPECT(desc.common_dims({2, 32, 1}) == final_lens{2, 2, 16, 1, 1});

    EXPECT(desc.common_axes_map_from_src() == axes_map{{0}, {1, 2}, {3, 4}});
    EXPECT(desc.common_axes_map_from_dst() == axes_map{{0, 1}, {2}, {3}, {4}});

    EXPECT(desc.to_common_from_src().generate() ==
           ops{make_op("reshape", {{"dims", {2, 2, 16, 64, 64}}})});
    EXPECT(desc.to_common_from_src().generate({2, 32, 1}) ==
           ops{make_op("reshape", {{"dims", {2, 2, 16, 1, 1}}})});
    EXPECT(desc.to_common_from_src().generate({2, 1, 4096}) ==
           ops{make_op("reshape", {{"dims", {2, 1, 1, 64, 64}}})});

    EXPECT(desc.to_common_from_dst().generate() ==
           ops{make_op("reshape", {{"dims", {2, 2, 16, 64, 64}}})});
    EXPECT(desc.to_common_from_dst().generate({4, 16, 1, 1}) ==
           ops{make_op("reshape", {{"dims", {2, 2, 16, 1, 1}}})});
    EXPECT(desc.to_common_from_dst().generate({4, 1, 64, 64}) ==
           ops{make_op("reshape", {{"dims", {2, 2, 1, 64, 64}}})});

    EXPECT(desc.to_dst_from_common().generate() ==
           ops{make_op("reshape", {{"dims", {4, 16, 64, 64}}})});
    EXPECT(desc.to_dst_from_common().generate({2, 2, 1, 64, 64}) ==
           ops{make_op("reshape", {{"dims", {4, 1, 64, 64}}})});
    EXPECT(desc.to_dst_from_common().generate({2, 2, 16, 1, 1}) ==
           ops{make_op("reshape", {{"dims", {4, 16, 1, 1}}})});
    EXPECT(desc.to_dst_from_common().generate({2, 1, 16, 64, 64}) ==
           ops{make_op("squeeze", {{"axes", {1}}})});

    EXPECT(desc.to_src_from_common().generate() ==
           ops{make_op("reshape", {{"dims", {2, 32, 4096}}})});
    EXPECT(desc.to_src_from_common().generate({2, 2, 1, 64, 64}) ==
           ops{make_op("reshape", {{"dims", {2, 2, 4096}}})});
    EXPECT(desc.to_src_from_common().generate({2, 2, 16, 1, 1}) ==
           ops{make_op("reshape", {{"dims", {2, 32, 1}}})});
    EXPECT(desc.to_src_from_common().generate({2, 1, 16, 64, 64}) ==
           ops{make_op("reshape", {{"dims", {2, 16, 4096}}})});
}

TEST_CASE(common_dims_reshape4)
{
    auto desc =
        make_simple_descriptor({4, 16, 64, 64}, make_op("reshape", {{"dims", {2, 32, 4096}}}));

    EXPECT(desc.common_dims() == final_lens{2, 2, 16, 64, 64});
    EXPECT(desc.common_dims({4, 16, 1, 1}) == final_lens{2, 2, 16, 1, 1});
    EXPECT(desc.common_dims({4, 1, 64, 64}) == final_lens{2, 2, 1, 64, 64});

    EXPECT(desc.common_axes_map_from_src() == axes_map{{0, 1}, {2}, {3}, {4}});
    EXPECT(desc.common_axes_map_from_dst() == axes_map{{0}, {1, 2}, {3, 4}});

    EXPECT(desc.to_common_from_dst().generate() ==
           ops{make_op("reshape", {{"dims", {2, 2, 16, 64, 64}}})});
    EXPECT(desc.to_common_from_dst().generate({2, 32, 1}) ==
           ops{make_op("reshape", {{"dims", {2, 2, 16, 1, 1}}})});
    EXPECT(desc.to_common_from_dst().generate({2, 1, 4096}) ==
           ops{make_op("reshape", {{"dims", {2, 1, 1, 64, 64}}})});

    EXPECT(desc.to_common_from_src().generate() ==
           ops{make_op("reshape", {{"dims", {2, 2, 16, 64, 64}}})});
    EXPECT(desc.to_common_from_src().generate({4, 16, 1, 1}) ==
           ops{make_op("reshape", {{"dims", {2, 2, 16, 1, 1}}})});
    EXPECT(desc.to_common_from_src().generate({4, 1, 64, 64}) ==
           ops{make_op("reshape", {{"dims", {2, 2, 1, 64, 64}}})});

    EXPECT(desc.to_dst_from_common().generate() ==
           ops{make_op("reshape", {{"dims", {2, 32, 4096}}})});
    EXPECT(desc.to_dst_from_common().generate({2, 2, 1, 64, 64}) ==
           ops{make_op("reshape", {{"dims", {2, 2, 4096}}})});
    EXPECT(desc.to_dst_from_common().generate({2, 2, 16, 1, 1}) ==
           ops{make_op("reshape", {{"dims", {2, 32, 1}}})});
    EXPECT(desc.to_dst_from_common().generate({2, 1, 16, 64, 64}) ==
           ops{make_op("reshape", {{"dims", {2, 16, 4096}}})});

    EXPECT(desc.to_src_from_common().generate() ==
           ops{make_op("reshape", {{"dims", {4, 16, 64, 64}}})});
    EXPECT(desc.to_src_from_common().generate({2, 2, 1, 64, 64}) ==
           ops{make_op("reshape", {{"dims", {4, 1, 64, 64}}})});
    EXPECT(desc.to_src_from_common().generate({2, 2, 16, 1, 1}) ==
           ops{make_op("reshape", {{"dims", {4, 16, 1, 1}}})});
    EXPECT(desc.to_src_from_common().generate({2, 1, 16, 64, 64}) ==
           ops{make_op("squeeze", {{"axes", {1}}})});
}

TEST_CASE(common_dims_transpose_reshape)
{
    auto desc = make_simple_descriptor({2, 16, 64, 64},
                                       make_op("transpose", {{"permutation", {0, 2, 3, 1}}}),
                                       make_op("reshape", {{"dims", {2, 32, 2048}}}));
    EXPECT(desc.common_dims() == final_lens{2, 32, 2, 64, 16});

    EXPECT(desc.common_axes_map_from_src() == axes_map{{0}, {4}, {1, 2}, {3}});
    EXPECT(desc.common_axes_map_from_dst() == axes_map{{0}, {1}, {2, 3, 4}});

    EXPECT(desc.to_common_from_dst().generate() ==
           ops{make_op("reshape", {{"dims", {2, 32, 2, 64, 16}}})});
    EXPECT(desc.to_common_from_dst().generate({2, 32, 1}) ==
           ops{make_op("unsqueeze", {{"axes", {3, 4}}})});
    EXPECT(desc.to_common_from_dst().generate({2, 1, 2048}) ==
           ops{make_op("reshape", {{"dims", {2, 1, 2, 64, 16}}})});

    EXPECT(desc.to_common_from_src().generate() ==
           ops{make_op("reshape", {{"dims", {2, 16, 32, 2, 64}}}),
               make_op("transpose", {{"permutation", {0, 2, 3, 4, 1}}})});
    EXPECT(desc.to_common_from_src().generate({2, 16, 1, 1}) ==
           ops{make_op("unsqueeze", {{"axes", {3}}}),
               make_op("transpose", {{"permutation", {0, 2, 3, 4, 1}}})});
    EXPECT(desc.to_common_from_src().generate({2, 1, 64, 64}) ==
           ops{make_op("reshape", {{"dims", {2, 1, 32, 2, 64}}}),
               make_op("transpose", {{"permutation", {0, 2, 3, 4, 1}}})});

    EXPECT(desc.to_dst_from_common().generate() ==
           ops{make_op("reshape", {{"dims", {2, 32, 2048}}})});
    EXPECT(desc.to_dst_from_common().generate({2, 1, 2, 64, 16}) ==
           ops{make_op("reshape", {{"dims", {2, 1, 2048}}})});
    EXPECT(desc.to_dst_from_common().generate({2, 1, 1, 1, 16}) ==
           ops{make_op("squeeze", {{"axes", {2, 3}}})});
    EXPECT(desc.to_dst_from_common().generate({2, 32, 2, 64, 1}) ==
           ops{make_op("reshape", {{"dims", {2, 32, 128}}})});

    // 2, 16, 32, 2, 64
    EXPECT(desc.to_src_from_common().generate() ==
           ops{make_op("transpose", {{"permutation", {0, 4, 1, 2, 3}}}),
               make_op("reshape", {{"dims", {2, 16, 64, 64}}})});
    // 2, 16, 1, 2, 64 => 2, 16, 2, 64
    EXPECT(desc.to_src_from_common().generate({2, 1, 2, 64, 16}) ==
           ops{make_op("transpose", {{"permutation", {0, 4, 1, 2, 3}}}),
               make_op("squeeze", {{"axes", {2}}})});
    // 2, 16, 1, 1, 1 => 2, 16, 1, 1
    EXPECT(desc.to_src_from_common().generate({2, 1, 1, 1, 16}) ==
           ops{make_op("transpose", {{"permutation", {0, 4, 1, 2, 3}}}),
               make_op("squeeze", {{"axes", {2}}})});
    // 2, 1, 32, 2, 64 => 2, 1, 64, 64
    EXPECT(desc.to_src_from_common().generate({2, 32, 2, 64, 1}) ==
           ops{make_op("transpose", {{"permutation", {0, 4, 1, 2, 3}}}),
               make_op("reshape", {{"dims", {2, 1, 64, 64}}})});
}

TEST_CASE(common_dims_broadcast_reshape)
{
    auto desc = make_simple_descriptor({2, 32, 1},
                                       make_op("multibroadcast", {{"out_lens", {2, 32, 4096}}}),
                                       make_op("reshape", {{"dims", {4, 16, 64, 64}}}));

    EXPECT(desc.common_dims() == final_lens{2, 2, 16, 64, 64});
    EXPECT(desc.common_dims({2, 1, 1}) == final_lens{2, 1, 1, 64, 64});
    EXPECT(desc.common_dims({2, 1, 4096}) == final_lens{2, 1, 1, 64, 64});
    EXPECT(desc.common_dims({2, 32, 4096}) == final_lens{2, 2, 16, 64, 64});

    EXPECT(desc.common_axes_map_from_src() == axes_map{{0}, {1, 2}, {3, 4}});
    EXPECT(desc.common_axes_map_from_dst() == axes_map{{0, 1}, {2}, {3}, {4}});

    EXPECT(desc.to_common_from_src().generate() ==
           ops{make_op("reshape", {{"dims", {2, 2, 16, 1, 1}}}),
               make_op("multibroadcast", {{"out_lens", {2, 2, 16, 64, 64}}})});
    EXPECT(desc.to_common_from_src().generate({2, 1, 1}) ==
           ops{make_op("unsqueeze", {{"axes", {2, 4}}}),
               make_op("multibroadcast", {{"out_lens", {2, 1, 1, 64, 64}}})});

    CHECK(desc.to_common_from_dst().generate() ==
          ops{make_op("reshape", {{"dims", {2, 2, 16, 64, 64}}})});
    EXPECT(desc.to_common_from_dst().generate({4, 16, 1, 1}) ==
           ops{make_op("reshape", {{"dims", {2, 2, 16, 1, 1}}})});
    EXPECT(desc.to_common_from_dst().generate({4, 1, 64, 64}) ==
           ops{make_op("reshape", {{"dims", {2, 2, 1, 64, 64}}})});

    EXPECT(desc.to_dst_from_common().generate() ==
           ops{make_op("reshape", {{"dims", {4, 16, 64, 64}}})});
    EXPECT(desc.to_dst_from_common().generate({2, 2, 1, 64, 64}) ==
           ops{make_op("reshape", {{"dims", {4, 1, 64, 64}}})});
    EXPECT(desc.to_dst_from_common().generate({2, 2, 16, 1, 1}) ==
           ops{make_op("reshape", {{"dims", {4, 16, 1, 1}}})});
    EXPECT(desc.to_dst_from_common().generate({2, 1, 16, 64, 64}) ==
           ops{make_op("squeeze", {{"axes", {1}}})});

    EXPECT(desc.to_src_from_common().generate() ==
           ops{make_op("reshape", {{"dims", {2, 32, 4096}}})});
    EXPECT(desc.to_src_from_common().generate({2, 2, 1, 64, 64}) ==
           ops{make_op("reshape", {{"dims", {2, 2, 4096}}})});
    EXPECT(desc.to_src_from_common().generate({2, 2, 16, 1, 1}) ==
           ops{make_op("reshape", {{"dims", {2, 32, 1}}})});
    EXPECT(desc.to_src_from_common().generate({2, 1, 16, 64, 64}) ==
           ops{make_op("reshape", {{"dims", {2, 16, 4096}}})});
}

TEST_CASE(common_dims_resize)
{
    auto desc =
        make_simple_descriptor({4, 16, 32, 32},
                               make_op("reshape", {{"dims", {4, 16, 32, 1, 32, 1}}}),
                               make_op("multibroadcast", {{"out_lens", {4, 16, 32, 2, 32, 2}}}),
                               make_op("reshape", {{"dims", {4, 16, 64, 64}}}));

    EXPECT(desc.common_dims() == final_lens{4, 16, 32, 2, 32, 2});
    EXPECT(desc.common_dims({4, 16, 1, 1}) == final_lens{4, 16, 1, 2, 1, 2});
    EXPECT(desc.common_dims({4, 1, 32, 32}) == final_lens{4, 1, 32, 2, 32, 2});

    EXPECT(desc.common_axes_map_from_src() == axes_map{{0}, {1}, {2}, {4}});
    EXPECT(desc.common_axes_map_from_dst() == axes_map{{0}, {1}, {2, 3}, {4, 5}});

    EXPECT(desc.to_common_from_src().generate() ==
           ops{make_op("unsqueeze", {{"axes", {3, 5}}}),
               make_op("multibroadcast", {{"out_lens", {4, 16, 32, 2, 32, 2}}})});
    EXPECT(desc.to_common_from_src().generate({4, 16, 1, 1}) ==
           ops{make_op("unsqueeze", {{"axes", {3, 5}}}),
               make_op("multibroadcast", {{"out_lens", {4, 16, 1, 2, 1, 2}}})});
    EXPECT(desc.to_common_from_src().generate({4, 1, 32, 32}) ==
           ops{make_op("unsqueeze", {{"axes", {3, 5}}}),
               make_op("multibroadcast", {{"out_lens", {4, 1, 32, 2, 32, 2}}})});

    EXPECT(desc.to_common_from_dst().generate() ==
           ops{make_op("reshape", {{"dims", {4, 16, 32, 2, 32, 2}}})});
    EXPECT(desc.to_common_from_dst().generate({4, 16, 1, 1}) ==
           ops{make_op("unsqueeze", {{"axes", {3, 5}}})});
    EXPECT(desc.to_common_from_dst().generate({4, 1, 64, 64}) ==
           ops{make_op("reshape", {{"dims", {4, 1, 32, 2, 32, 2}}})});

    EXPECT(desc.to_dst_from_common().generate() ==
           ops{make_op("reshape", {{"dims", {4, 16, 64, 64}}})});
    EXPECT(desc.to_dst_from_common().generate({4, 16, 1, 2, 1, 2}) ==
           ops{make_op("squeeze", {{"axes", {2, 4}}})});
    EXPECT(desc.to_dst_from_common().generate({4, 1, 32, 2, 32, 2}) ==
           ops{make_op("reshape", {{"dims", {4, 1, 64, 64}}})});
    EXPECT(desc.to_dst_from_common().generate({4, 16, 32, 1, 32, 1}) ==
           ops{make_op("squeeze", {{"axes", {3, 5}}})});

    EXPECT(desc.to_src_from_common().generate() == ops{make_op("squeeze", {{"axes", {3, 5}}})});
    EXPECT(desc.to_src_from_common().generate({4, 16, 1, 1, 1, 1}) ==
           ops{make_op("squeeze", {{"axes", {3, 4}}})});
    EXPECT(desc.to_src_from_common().generate({4, 1, 32, 1, 32, 1}) ==
           ops{make_op("squeeze", {{"axes", {3, 5}}})});
    EXPECT(desc.to_src_from_common().generate({4, 16, 32, 1, 32, 1}) ==
           ops{make_op("squeeze", {{"axes", {3, 5}}})});
}

TEST_CASE(common_dims_squeeze_1x1)
{
    auto desc = make_simple_descriptor({1, 1}, make_op("squeeze", {{"axes", {0}}}));
    desc.simplify();
    EXPECT(desc.common_dims() == final_lens{1, 1});
    EXPECT(desc.common_axes_map_from_src() == axes_map{{0}, {1}});
    EXPECT(desc.common_axes_map_from_dst() == axes_map{{0, 1}});
    EXPECT(desc.to_common_from_src().generate() == ops{});
    EXPECT(desc.to_common_from_dst().generate() == ops{make_op("unsqueeze", {{"axes", {1}}})});
    EXPECT(desc.to_dst_from_common().generate() == ops{make_op("squeeze", {{"axes", {0}}})});
    EXPECT(desc.to_src_from_common().generate() == ops{});
}

TEST_CASE(rebase_reshape_broadcast)
{
    auto base_desc =
        make_simple_descriptor({3, 4, 64, 1},
                               make_op("reshape", {{"dims", {12, 8, 8, 1, 1}}}),
                               make_op("multibroadcast", {{"out_lens", {12, 8, 8, 2, 2}}}));

    {
        auto desc = base_desc.rebase({3, 4, 64, 4});
        EXPECT(get_final_lens(desc) == final_lens{12, 8, 8, 2, 2});
        EXPECT(get_all_lens(desc) == all_lens{{3, 4}, {8}, {8}, {2}, {2}});
        EXPECT(desc.generate() == ops{make_op("reshape", {{"dims", {3, 4, 8, 8, 2, 2}}}),
                                      make_op("reshape", {{"dims", {12, 8, 8, 2, 2}}})});
    }

    {
        auto desc = base_desc.rebase({3, 5, 64, 1});
        EXPECT(get_final_lens(desc) == final_lens{15, 8, 8, 2, 2});
        EXPECT(get_all_lens(desc) == all_lens{{3, 5}, {8}, {8}, {2}, {2}});
        EXPECT(desc.generate() == ops{make_op("reshape", {{"dims", {3, 5, 8, 8, 1, 1}}}),
                                      make_op("reshape", {{"dims", {15, 8, 8, 1, 1}}}),
                                      make_op("multibroadcast", {{"out_lens", {15, 8, 8, 2, 2}}})});
    }

    {
        auto desc = base_desc.rebase({3, 4, 1, 1});
        EXPECT(get_final_lens(desc) == final_lens{12, 1, 1, 2, 2});
        EXPECT(get_all_lens(desc) == all_lens{{3, 4}, {1}, {1}, {2}, {2}});
        EXPECT(desc.generate() == ops{make_op("unsqueeze", {{"axes", {3, 5}}}),
                                      make_op("reshape", {{"dims", {12, 1, 1, 1, 1}}}),
                                      make_op("multibroadcast", {{"out_lens", {12, 1, 1, 2, 2}}})});
    }
}

TEST_CASE(rebase_unsqueeze_broadcast)
{
    auto base_desc =
        make_simple_descriptor({1, 3, 1, 1},
                               make_op("unsqueeze", {{"axes", {3, 5}}}),
                               make_op("multibroadcast", {{"out_lens", {1, 3, 256, 2, 256, 2}}}));

    {
        auto desc = base_desc.rebase({1, 3, 512, 512});
        EXPECT(get_final_lens(desc) == final_lens{1, 3, 256, 2, 256, 2});
        EXPECT(get_all_lens(desc) == all_lens{{1}, {3}, {256}, {2}, {256}, {2}});
        EXPECT(desc.generate() == ops{
                                      make_op("reshape", {{"dims", {1, 3, 256, 2, 256, 2}}}),
                                  });
    }
}

TEST_CASE(rebase_unsqueeze_broadcast_transpose)
{
    auto base_desc =
        make_simple_descriptor({1, 1, 1, 3},
                               make_op("unsqueeze", {{"axes", {3, 4}}}),
                               make_op("transpose", {{"permutation", {0, 5, 1, 2, 3, 4}}}),
                               make_op("multibroadcast", {{"out_lens", {1, 3, 256, 2, 256, 2}}}));

    {
        auto desc = base_desc.rebase({1, 512, 512, 3});
        EXPECT(get_final_lens(desc) == final_lens{1, 3, 256, 2, 256, 2});
        EXPECT(get_all_lens(desc) == all_lens{{1}, {3}, {256}, {2}, {256}, {2}});
        EXPECT(desc.generate() == ops{
                                      make_op("reshape", {{"dims", {1, 256, 2, 256, 2, 3}}}),
                                      make_op("transpose", {{"permutation", {0, 5, 1, 2, 3, 4}}}),
                                  });
    }
}

TEST_CASE(rebase_squeeze_broadcast_transpose)
{
    auto base_desc =
        make_simple_descriptor({1, 1, 1, 1, 1, 3},
                               make_op("squeeze", {{"axes", {2, 4}}}),
                               make_op("transpose", {{"permutation", {0, 3, 1, 2}}}),
                               make_op("multibroadcast", {{"out_lens", {1, 3, 512, 512}}}));

    {
        auto desc = base_desc.rebase({1, 256, 2, 256, 2, 3});
        EXPECT(get_final_lens(desc) == final_lens{1, 3, 512, 512});
        EXPECT(get_all_lens(desc) == all_lens{{1}, {3}, {256, 2}, {256, 2}});
        EXPECT(desc.generate() == ops{
                                      make_op("transpose", {{"permutation", {0, 5, 1, 2, 3, 4}}}),
                                      make_op("reshape", {{"dims", {1, 3, 512, 512}}}),
                                  });
    }
}

TEST_CASE(rebase_transpose_reshape_1s)
{
    auto base_desc =
        make_simple_descriptor({1, 3, 256, 2, 256, 2},
                               make_op("transpose", {{"permutation", {0, 2, 5, 3, 4, 1}}}),
                               make_op("reshape", {{"dims", {1, 512, 512, 3}}}));

    {
        auto desc = base_desc.rebase({1, 3, 1, 1, 1, 1});
        EXPECT(get_final_lens(desc) == final_lens{1, 1, 1, 3});
        EXPECT(get_all_lens(desc) == all_lens{{1}, {1, 1}, {1, 1}, {3}});
        EXPECT(desc.generate() == ops{
                                      make_op("transpose", {{"permutation", {0, 2, 5, 3, 4, 1}}}),
                                      make_op("squeeze", {{"axes", {1, 3}}}),
                                  });
    }
}

// Test cases specifically targeting different paths in adjust_axes_for_rebase

TEST_CASE(rebase_adjust_axes_basic_shortage)
{
    // Test shortage matching with broadcast excess
    // Original: {1, 16} broadcast to larger, then need to match with {1, 16}
    auto base_desc = make_simple_descriptor({1, 16},
                                            make_op("multibroadcast", {{"out_lens", {16, 16}}}),
                                            make_op("reshape", {{"dims", {2, 8, 16}}}));

    {
        // Rebase back to the original - should work
        auto desc = base_desc.rebase({1, 16});
        EXPECT(not desc.empty());
        EXPECT(get_final_lens(desc) == final_lens{2, 8, 16});
        EXPECT(get_all_lens(desc) == all_lens{{2}, {8}, {16}});
        auto generated = desc.generate();
        EXPECT(generated == ops{
                                make_op("unsqueeze", {{"axes", {1}}}),
                                make_op("multibroadcast", {{"out_lens", {2, 8, 16}}}),
                            });
    }
}

TEST_CASE(rebase_adjust_axes_multiple_groups)
{
    // Test multiple groups in group algorithms
    // Create transformation with multiple broadcast dimensions
    auto base_desc = make_simple_descriptor({1, 1, 32},
                                            make_op("multibroadcast", {{"out_lens", {4, 8, 32}}}),
                                            make_op("reshape", {{"dims", {2, 2, 2, 4, 32}}}));

    {
        // Multiple axes have shortages that need different handling
        auto desc = base_desc.rebase({1, 1, 32}, true); // Force broadcast mode
        EXPECT(not desc.empty());
        EXPECT(get_final_lens(desc) == final_lens{2, 2, 2, 4, 32});
        EXPECT(get_all_lens(desc) == all_lens{{2}, {2}, {2}, {4}, {32}});
        auto generated = desc.generate();
        EXPECT(generated == ops{
                                make_op("unsqueeze", {{"axes", {1, 3}}}),
                                make_op("multibroadcast", {{"out_lens", {2, 2, 2, 4, 32}}}),
                            });
    }
}

TEST_CASE(rebase_adjust_axes_hidden_swap_simple)
{
    // Simple test for hidden axis logic without complex transformations
    auto base_desc =
        make_simple_descriptor({2, 1, 16}, make_op("multibroadcast", {{"out_lens", {2, 8, 16}}}));

    {
        // The broadcast axis becomes hidden when rebasing
        auto desc = base_desc.rebase({2, 1, 16}, true);
        EXPECT(not desc.empty());
        EXPECT(get_final_lens(desc) == final_lens{2, 8, 16});
        EXPECT(get_all_lens(desc) == all_lens{{2}, {8}, {16}});
        EXPECT(desc.find_broadcasted_axes().size() == 1); // Should have hidden axis at 1
        auto generated = desc.generate();
        EXPECT(generated == ops{
                                make_op("multibroadcast", {{"out_lens", {2, 8, 16}}}),
                            });
    }
}

TEST_CASE(rebase_adjust_axes_axis_movement)
{
    // Test axis movement and insertion logic (subs_to_insert path)
    auto base_desc = make_simple_descriptor({8, 1},
                                            make_op("reshape", {{"dims", {2, 4, 1}}}),
                                            make_op("multibroadcast", {{"out_lens", {2, 4, 16}}}));

    {
        // Rebase to something compatible
        auto desc = base_desc.rebase({8, 1});
        EXPECT(not desc.empty());
        EXPECT(get_final_lens(desc) == final_lens{2, 4, 16});
        EXPECT(get_all_lens(desc) == all_lens{{2}, {4}, {16}});
        auto generated = desc.generate();
        EXPECT(generated == ops{
                                make_op("reshape", {{"dims", {2, 4, 1}}}),
                                make_op("multibroadcast", {{"out_lens", {2, 4, 16}}}),
                            });
    }
}

TEST_CASE(rebase_adjust_axes_group_unique_hidden)
{
    // Test group_unique with hidden axis groups
    auto base_desc = make_simple_descriptor(
        {1, 1, 1, 16}, make_op("multibroadcast", {{"out_lens", {2, 4, 8, 16}}}));

    {
        // Multiple hidden axes that need grouping and sorting
        auto desc = base_desc.rebase({1, 1, 1, 16}, true);
        EXPECT(not desc.empty());
        EXPECT(get_final_lens(desc) == final_lens{2, 4, 8, 16});
        EXPECT(get_all_lens(desc) == all_lens{{2}, {4}, {8}, {16}});
        EXPECT(desc.find_broadcasted_axes().size() == 3); // Three broadcast axes
        auto generated = desc.generate();
        EXPECT(generated == ops{
                                make_op("multibroadcast", {{"out_lens", {2, 4, 8, 16}}}),
                            });
    }
}

TEST_CASE(rebase_no_axis_subdimensions)
{
    // Test the logic for subdimensions with no axis (pure broadcasts)
    auto base_desc = make_simple_descriptor({1},
                                            make_op("multibroadcast", {{"out_lens", {16}}}),
                                            make_op("reshape", {{"dims", {2, 8}}}));

    {
        // The no-axis subdimension should be considered for excess
        auto desc = base_desc.rebase({16});
        EXPECT(get_final_lens(desc) == final_lens{2, 8});
        EXPECT(desc.generate() == ops{
                                      make_op("reshape", {{"dims", {2, 8}}}),
                                  });
    }
}

TEST_CASE(rebase_adjust_axes_insert_and_sort)
{
    // Test subs_to_insert path and moved axes sorting
    auto base_desc = make_simple_descriptor({16, 1},
                                            make_op("reshape", {{"dims", {2, 8, 1}}}),
                                            make_op("multibroadcast", {{"out_lens", {2, 8, 32}}}),
                                            make_op("reshape", {{"dims", {2, 256}}}));

    {
        // Use the original shape for rebase
        auto desc = base_desc.rebase({16, 1});
        EXPECT(not desc.empty());
        EXPECT(get_final_lens(desc) == final_lens{2, 256});
        EXPECT(get_all_lens(desc) == all_lens{{2}, {8, 32}});
        auto generated = desc.generate();
        EXPECT(generated == ops{
                                make_op("reshape", {{"dims", {2, 8, 1}}}),
                                make_op("multibroadcast", {{"out_lens", {2, 8, 32}}}),
                                make_op("reshape", {{"dims", {2, 256}}}),
                            });
    }
}

TEST_CASE(rebase_adjust_axes_group_unique_segments)
{
    // Test multiple segments in group_unique with different hidden axis groups
    auto base_desc = make_simple_descriptor({1, 1, 32},
                                            make_op("multibroadcast", {{"out_lens", {8, 4, 32}}}),
                                            make_op("reshape", {{"dims", {2, 4, 4, 32}}}));

    {
        // Multiple groups for group_unique to process
        auto desc = base_desc.rebase({1, 1, 32}, true);
        EXPECT(not desc.empty());
        EXPECT(get_final_lens(desc) == final_lens{2, 4, 4, 32});
        EXPECT(get_all_lens(desc) == all_lens{{2}, {4}, {4}, {32}});
        auto generated = desc.generate();
        EXPECT(generated == ops{
                                make_op("unsqueeze", {{"axes", {1}}}),
                                make_op("multibroadcast", {{"out_lens", {2, 4, 4, 32}}}),
                            });
    }
}

TEST_CASE(rebase_adjust_axes_many_hidden_groups)
{
    // Test with >3 groups of hidden axes to exercise group_unique sorting thoroughly
    auto base_desc =
        make_simple_descriptor({1, 1, 1, 1, 1, 16},
                               make_op("multibroadcast", {{"out_lens", {2, 3, 4, 5, 6, 16}}}),
                               make_op("reshape", {{"dims", {2, 3, 2, 2, 5, 2, 3, 16}}}));

    {
        // 5 broadcast axes that become hidden, creating multiple groups
        auto desc = base_desc.rebase({1, 1, 1, 1, 1, 16}, true);
        EXPECT(not desc.empty());
        EXPECT(get_final_lens(desc) == final_lens{2, 3, 2, 2, 5, 2, 3, 16});
        EXPECT(get_all_lens(desc) == all_lens{{2}, {3}, {2}, {2}, {5}, {2}, {3}, {16}});
        // Should have 5 hidden axes
        EXPECT(desc.find_broadcasted_axes().size() == 5);
        auto generated = desc.generate();
        EXPECT(generated ==
               ops{
                   make_op("unsqueeze", {{"axes", {3, 6}}}),
                   make_op("multibroadcast", {{"out_lens", {2, 3, 2, 2, 5, 2, 3, 16}}}),
               });
    }
}

TEST_CASE(rebase_adjust_axes_many_moved_groups)
{
    // Test with >3 groups where axes need to be moved and sorted
    // This creates a complex scenario with multiple shortage/excess pairs
    auto base_desc =
        make_simple_descriptor({1, 1, 1, 1, 64},
                               make_op("multibroadcast", {{"out_lens", {8, 4, 2, 6, 64}}}),
                               make_op("reshape", {{"dims", {2, 4, 2, 2, 2, 3, 2, 2, 32}}}));

    {
        // Multiple axes with different shortage/excess patterns - use broadcast mode
        auto desc = base_desc.rebase({1, 1, 1, 1, 64}, true);
        EXPECT(not desc.empty());
        EXPECT(get_final_lens(desc) == final_lens{2, 4, 2, 2, 2, 3, 2, 2, 32});
        EXPECT(get_all_lens(desc) == all_lens{{2}, {4}, {2}, {2}, {2}, {3}, {2}, {2}, {32}});
        auto generated = desc.generate();
        EXPECT(generated ==
               ops{
                   make_op("reshape", {{"dims", {1, 1, 1, 1, 1, 1, 1, 2, 32}}}),
                   make_op("multibroadcast", {{"out_lens", {2, 4, 2, 2, 2, 3, 2, 2, 32}}}),
               });
    }
}

TEST_CASE(rebase_adjust_squeeze_unsqueeze_broadcast)
{
    auto base_desc = make_simple_descriptor(
        {1, 1, 1, 1, 1, 1, 32, 1, 1, 1, 1, 1},
        make_op("squeeze", {{"axes", {1, 2, 3, 4, 5, 7, 8, 9, 10}}}),
        make_op("unsqueeze", {{"axes", {1, 2, 3, 4, 5, 7, 8, 10, 11}}}),
        make_op("multibroadcast", {{"out_lens", {1, 1, 1, 1, 1, 1, 32, 10, 16, 1, 90, 160}}}));

    {
        auto desc = base_desc.rebase({1, 1, 1, 1, 1, 1, 32, 10, 16, 1, 90, 160});
        EXPECT(not desc.empty());
        EXPECT(get_final_lens(desc) == final_lens{1, 1, 1, 1, 1, 1, 32, 10, 16, 1, 90, 160});
        EXPECT(get_all_lens(desc) ==
               all_lens{{1}, {1}, {1}, {1}, {1}, {1}, {32}, {10}, {16}, {1}, {90}, {160}});
        EXPECT(desc.generate() == ops{});
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
