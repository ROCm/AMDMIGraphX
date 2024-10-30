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
 *
 */
#include <migraphx/shape_transform_descriptor.hpp>
#include <migraphx/make_op.hpp>
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

all_lens get_all_lens(const shape_transform_descriptor& d)
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

final_lens get_final_lens(const shape_transform_descriptor& d)
{
    final_lens result;
    std::transform(d.dimensions.begin(),
                   d.dimensions.end(),
                   std::back_inserter(result),
                   [](const auto& x) { return x.len(); });
    return result;
}

all_axes get_all_axes(const shape_transform_descriptor& d)
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

template <class... Ts>
shape_transform_descriptor make_descriptor(const std::vector<std::size_t>& dims, const Ts&... xs)
{
    auto desc = shape_transform_descriptor{dims};
    CHECK(desc.apply({xs...}));
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
           all_axes{d_axes{{0}}, d_axes{{1}}, d_axes{{2}}, d_axes{{}}, d_axes{{}}});
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
    EXPECT(get_all_axes(desc) == all_axes{d_axes{{0}}, d_axes{{1}}, d_axes{{2}}});
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
    EXPECT(migraphx::optimize_shape_transforms(
               {3, 5, 2},
               {
                   make_op("transpose", {{"permutation", {0, 2, 1}}}),
                   make_op("transpose", {{"permutation", {1, 0, 2}}}),
               }) == ops{
                         make_op("transpose", {{"permutation", {2, 0, 1}}}),
                     });
}

TEST_CASE(optimize_reshape_reshape1)
{
    EXPECT(migraphx::optimize_shape_transforms({3, 5, 2},
                                               {
                                                   make_op("reshape", {{"dims", {30}}}),
                                                   make_op("reshape", {{"dims", {3, 10}}}),
                                               }) == ops{
                                                         make_op("reshape", {{"dims", {3, 10}}}),
                                                     });
}

TEST_CASE(optimize_reshape_reshape2)
{
    EXPECT(migraphx::optimize_shape_transforms({15, 4},
                                               {
                                                   make_op("reshape", {{"dims", {3, 5, 2, 2}}}),
                                                   make_op("reshape", {{"dims", {15, 2, 2}}}),
                                               }) == ops{
                                                         make_op("reshape", {{"dims", {15, 2, 2}}}),
                                                     });
}

TEST_CASE(optimize_reshape_transpose_reshape_to_none)
{
    EXPECT(migraphx::optimize_shape_transforms(
               {6, 5, 2},
               {
                   make_op("reshape", {{"dims", {6, 5, 2, 1, 1}}}),
                   make_op("transpose", {{"permutation", {0, 1, 2, 4, 3}}}),
                   make_op("reshape", {{"dims", {6, 5, 2}}}),
               }) == ops{});
}

TEST_CASE(optimize_reshape_transpose_reshape_to_same)
{
    EXPECT(migraphx::optimize_shape_transforms(
               {1, 112, 56, 56},
               {
                   make_op("reshape", {{"dims", {1, 4, 28, 56, 56}}}),
                   make_op("transpose", {{"permutation", {0, 2, 1, 3, 4}}}),
                   make_op("reshape", {{"dims", {1, 112, 56, 56}}}),
               }) == ops{
                         make_op("reshape", {{"dims", {1, 4, 28, 56, 56}}}),
                         make_op("transpose", {{"permutation", {0, 2, 1, 3, 4}}}),
                         make_op("reshape", {{"dims", {1, 112, 56, 56}}}),
                     });
}

TEST_CASE(optimize_reshape_transpose_reshape_to_transpose)
{
    EXPECT(migraphx::optimize_shape_transforms(
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
    EXPECT(migraphx::optimize_shape_transforms(
               {6, 5, 2},
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
    EXPECT(migraphx::optimize_shape_transforms(
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
    EXPECT(migraphx::optimize_shape_transforms(
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
    EXPECT(migraphx::optimize_shape_transforms(
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
    EXPECT(migraphx::optimize_shape_transforms({3, 1, 5, 1, 2, 1, 1},
                                               {
                                                   make_op("reshape", {{"dims", {3, 5, 2}}}),
                                               }) ==
           ops{
               make_op("squeeze", {{"axes", {1, 3, 5, 6}}}),
           });
}

TEST_CASE(optimize_reshape_2_unsqueeze)
{
    EXPECT(migraphx::optimize_shape_transforms(
               {3, 5, 2},
               {
                   make_op("reshape", {{"dims", {3, 1, 5, 1, 2, 1, 1}}}),
               }) == ops{
                         make_op("unsqueeze", {{"axes", {1, 3, 5, 6}}}),
                     });
}

TEST_CASE(optimize_unsqueeze_multibroadcast)
{
    EXPECT(migraphx::optimize_shape_transforms(
               {32, 10},
               {
                   make_op("unsqueeze", {{"axes", {0, 3, 4}}}),
                   make_op("multibroadcast", {{"out_lens", {256, 32, 10, 16, 16}}}),
               }) == ops{
                         make_op("broadcast", {{"axis", 1}, {"out_lens", {256, 32, 10, 16, 16}}}),
                     });
}

TEST_CASE(optimize_multibroadcast_reshape)
{
    EXPECT(migraphx::optimize_shape_transforms(
               {1, 4, 1},
               {
                   make_op("multibroadcast", {{"out_lens", {2, 4, 6}}}),
                   make_op("reshape", {{"dims", {2, 2, 2, 6}}}),
               }) == ops{
                         make_op("reshape", {{"dims", {1, 2, 2, 1}}}),
                         make_op("multibroadcast", {{"out_lens", {2, 2, 2, 6}}}),
                     });
}

TEST_CASE(optimize_squeeze_broadcast)
{
    EXPECT(migraphx::optimize_shape_transforms(
               {256, 1, 1},
               {
                   make_op("squeeze"),
                   make_op("broadcast", {{"axis", 0}, {"out_lens", {256, 64, 1, 1}}}),
               }) == ops{
                         make_op("unsqueeze", {{"axes", {3}}}),
                         make_op("multibroadcast", {{"out_lens", {256, 64, 1, 1}}}),
                     });
}

TEST_CASE(optimize_squeeze_unsqueeze_broadcast_to_broadcast)
{
    EXPECT(migraphx::optimize_shape_transforms(
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
    EXPECT(migraphx::optimize_shape_transforms(
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
    EXPECT(migraphx::optimize_shape_transforms({1},
                                               {
                                                   make_op("multibroadcast", {{"out_lens", {2}}}),
                                                   make_op("unsqueeze", {{"axes", {1}}}),
                                               }) ==
           ops{
               make_op("multibroadcast", {{"out_lens", {2, 1}}}),
           });
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
