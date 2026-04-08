/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/kernels/spatial_tiler.hpp>
#include <migraphx/kernels/test.hpp>

// Helper: create a standard 4D shape from lens
template <migraphx::index_int N,
          migraphx::index_int C,
          migraphx::index_int H,
          migraphx::index_int W>
constexpr auto make_4d_shape()
{
    constexpr auto lens = migraphx::index_ints<N, C, H, W>{};
    return migraphx::make_shape(lens);
}

// ======== output_lens ========

// Tile {4, 4} with NTiles=1 → output_lens = {1, 1, 4, 4}
TEST_CASE(output_lens_ntiles_1)
{
    using tiler = migraphx::
        spatial_tiler<1, migraphx::index_ints<4, 4>, decltype(make_4d_shape<1, 1, 8, 8>())>;
    constexpr auto ol = tiler::output_lens();
    EXPECT(ol.size() == 4);
    EXPECT(ol[0] == 1);
    EXPECT(ol[1] == 1);
    EXPECT(ol[2] == 4);
    EXPECT(ol[3] == 4);
}

// Tile {4, 4} with NTiles=2 → last dim doubled: {1, 1, 4, 8}
TEST_CASE(output_lens_ntiles_2)
{
    using tiler = migraphx::
        spatial_tiler<2, migraphx::index_ints<4, 4>, decltype(make_4d_shape<1, 1, 8, 8>())>;
    constexpr auto ol = tiler::output_lens();
    EXPECT(ol[2] == 4);
    EXPECT(ol[3] == 8);
}

// ======== out_spatial_lens ========

TEST_CASE(out_spatial_lens_basic)
{
    using tiler = migraphx::
        spatial_tiler<1, migraphx::index_ints<4, 4>, decltype(make_4d_shape<2, 3, 16, 16>())>;
    constexpr auto sl = tiler::out_spatial_lens();
    // keep_spatial sets dims 0,1 to 1; keeps H,W
    EXPECT(sl[0] == 1);
    EXPECT(sl[1] == 1);
    EXPECT(sl[2] == 16);
    EXPECT(sl[3] == 16);
}

// ======== tiles_per_dim ========

// 8x8 output, 4x4 tile, NTiles=1 → ceil(8/4)=2 per spatial dim
TEST_CASE(tiles_per_dim_exact)
{
    using tiler = migraphx::
        spatial_tiler<1, migraphx::index_ints<4, 4>, decltype(make_4d_shape<1, 1, 8, 8>())>;
    constexpr auto tpd = tiler::tiles_per_dim();
    EXPECT(tpd[2] == 2);
    EXPECT(tpd[3] == 2);
}

// 10x10 output, 4x4 tile → ceil(10/4)=3 per spatial dim
TEST_CASE(tiles_per_dim_inexact)
{
    using tiler = migraphx::
        spatial_tiler<1, migraphx::index_ints<4, 4>, decltype(make_4d_shape<1, 1, 10, 10>())>;
    constexpr auto tpd = tiler::tiles_per_dim();
    EXPECT(tpd[2] == 3);
    EXPECT(tpd[3] == 3);
}

// NTiles=2 scales last dim: tile output is {4, 8} → ceil(16/4)=4, ceil(16/8)=2
TEST_CASE(tiles_per_dim_ntiles)
{
    using tiler = migraphx::
        spatial_tiler<2, migraphx::index_ints<4, 4>, decltype(make_4d_shape<1, 1, 16, 16>())>;
    constexpr auto tpd = tiler::tiles_per_dim();
    EXPECT(tpd[2] == 4);
    EXPECT(tpd[3] == 2);
}

// ======== tiles_total ========

TEST_CASE(tiles_total_exact)
{
    using tiler = migraphx::
        spatial_tiler<1, migraphx::index_ints<4, 4>, decltype(make_4d_shape<1, 1, 8, 8>())>;
    // tiles_per_dim = {1, 1, 2, 2}, product = 4
    EXPECT(tiler::tiles_total() == 4);
}

// ======== get_padding / left_padding / total_padding ========

// No Padding arg → get_padding returns zeros matching TileLens size
TEST_CASE(get_padding_default)
{
    using tiler = migraphx::
        spatial_tiler<1, migraphx::index_ints<4, 4>, decltype(make_4d_shape<1, 1, 8, 8>())>;
    constexpr auto gp = tiler::get_padding();
    EXPECT(gp.size() == 4);
    EXPECT(gp[0] == 0);
    EXPECT(gp[1] == 0);
    EXPECT(gp[2] == 0);
    EXPECT(gp[3] == 0);
}

// No padding template arg → all zeros
TEST_CASE(padding_default_no_padding)
{
    using tiler = migraphx::
        spatial_tiler<1, migraphx::index_ints<4, 4>, decltype(make_4d_shape<1, 1, 8, 8>())>;
    constexpr auto lp = tiler::left_padding();
    constexpr auto tp = tiler::total_padding();
    EXPECT(lp[0] == 0);
    EXPECT(lp[1] == 0);
    EXPECT(lp[2] == 0);
    EXPECT(lp[3] == 0);
    EXPECT(tp[0] == 0);
    EXPECT(tp[1] == 0);
    EXPECT(tp[2] == 0);
    EXPECT(tp[3] == 0);
}

// Symmetric padding {1, 1, 1, 1} → left={0,0,1,1}, total={0,0,2,2}
TEST_CASE(padding_symmetric)
{
    using tiler       = migraphx::spatial_tiler<1,
                                                migraphx::index_ints<4, 4>,
                                                decltype(make_4d_shape<1, 1, 8, 8>()),
                                                migraphx::index_ints<1, 1, 1, 1>>;
    constexpr auto lp = tiler::left_padding();
    EXPECT(lp[0] == 0);
    EXPECT(lp[1] == 0);
    EXPECT(lp[2] == 1);
    EXPECT(lp[3] == 1);

    constexpr auto tp = tiler::total_padding();
    EXPECT(tp[0] == 0);
    EXPECT(tp[1] == 0);
    EXPECT(tp[2] == 2);
    EXPECT(tp[3] == 2);
}

// Asymmetric padding {1, 2, 3, 4} → left={0,0,1,2}, total={0,0,1+3,2+4}={0,0,4,6}
TEST_CASE(padding_asymmetric)
{
    using tiler       = migraphx::spatial_tiler<1,
                                                migraphx::index_ints<4, 4>,
                                                decltype(make_4d_shape<1, 1, 8, 8>()),
                                                migraphx::index_ints<1, 2, 3, 4>>;
    constexpr auto lp = tiler::left_padding();
    EXPECT(lp[2] == 1);
    EXPECT(lp[3] == 2);

    constexpr auto tp = tiler::total_padding();
    EXPECT(tp[2] == 4);
    EXPECT(tp[3] == 6);
}

// ======== is_padded ========

// Tiles exactly cover output, no conv padding → not padded
TEST_CASE(is_padded_exact_no_padding)
{
    using tiler = migraphx::
        spatial_tiler<1, migraphx::index_ints<4, 4>, decltype(make_4d_shape<1, 1, 8, 8>())>;
    EXPECT(not tiler::is_padded());
}

// Tiles don't exactly cover output (10 not divisible by 4) → padded
TEST_CASE(is_padded_overhang)
{
    using tiler = migraphx::
        spatial_tiler<1, migraphx::index_ints<4, 4>, decltype(make_4d_shape<1, 1, 10, 10>())>;
    EXPECT(tiler::is_padded());
}

// Tiles exactly cover output but conv padding present → padded
TEST_CASE(is_padded_conv_padding_exact_tiles)
{
    using tiler = migraphx::spatial_tiler<1,
                                          migraphx::index_ints<4, 4>,
                                          decltype(make_4d_shape<1, 1, 8, 8>()),
                                          migraphx::index_ints<1, 1, 1, 1>>;
    EXPECT(tiler::is_padded());
}

// Both overhang and conv padding → padded
TEST_CASE(is_padded_overhang_and_conv_padding)
{
    using tiler = migraphx::spatial_tiler<1,
                                          migraphx::index_ints<4, 4>,
                                          decltype(make_4d_shape<1, 1, 10, 10>()),
                                          migraphx::index_ints<1, 1, 1, 1>>;
    EXPECT(tiler::is_padded());
}

// Edge case: tile overhang equals total padding → still padded
// out_spatial=10, tile=8, tiles_per_dim=2, tiles*tile=16, total_pad=6
// Without the fix: 10 != 16 → padded (only by coincidence).
// With total_padding in formula: 10 != 16+6=22 → padded.
TEST_CASE(is_padded_overhang_equals_padding)
{
    // tiles_per_dim = ceil(10/8) = 2, coverage = 16, total_pad_h=3+3=6
    using tiler = migraphx::spatial_tiler<1,
                                          migraphx::index_ints<8, 8>,
                                          decltype(make_4d_shape<1, 1, 10, 10>()),
                                          migraphx::index_ints<3, 3, 3, 3>>;
    EXPECT(tiler::is_padded());
}

// Only one spatial dim has overhang
TEST_CASE(is_padded_partial_overhang)
{
    // H=8 exactly tiled by tile_h=4. W=10 not divisible by tile_w=4.
    using tiler = migraphx::
        spatial_tiler<1, migraphx::index_ints<4, 4>, decltype(make_4d_shape<1, 1, 8, 10>())>;
    EXPECT(tiler::is_padded());
}

// Large padding values
TEST_CASE(is_padded_large_padding)
{
    using tiler = migraphx::spatial_tiler<1,
                                          migraphx::index_ints<4, 4>,
                                          decltype(make_4d_shape<1, 1, 8, 8>()),
                                          migraphx::index_ints<3, 3, 3, 3>>;
    EXPECT(tiler::is_padded());
}

// ======== has_nonzero ========

TEST_CASE(has_nonzero_all_zero)
{
    EXPECT(not migraphx::has_nonzero(migraphx::index_ints<0, 0, 0, 0>{}));
}

TEST_CASE(has_nonzero_some_nonzero)
{
    EXPECT(migraphx::has_nonzero(migraphx::index_ints<0, 0, 1, 0>{}));
}

TEST_CASE(has_nonzero_all_nonzero)
{
    EXPECT(migraphx::has_nonzero(migraphx::index_ints<1, 2, 3, 4>{}));
}

// ======== halo_lens_for ========

// No padding: halo = output_lens + (input_spatial - out_spatial)
TEST_CASE(halo_lens_no_padding)
{
    // Output 8x8, input 10x10 (e.g. 3x3 conv), tile 4x4
    // out_spatial = {1,1,8,8}, input_spatial = {1,1,10,10}
    // halo_extra = {1,1,10,10} - {1,1,8,8} + {0,0,0,0} = {0,0,2,2}
    // halo_lens = output_lens + halo_extra = {1,1,4,4} + {0,0,2,2} = {1,1,6,6}
    using output_shape = decltype(make_4d_shape<1, 1, 8, 8>());
    using input_shape  = decltype(make_4d_shape<1, 1, 10, 10>());
    using tiler        = migraphx::spatial_tiler<1, migraphx::index_ints<4, 4>, output_shape>;

    constexpr auto hl = tiler::template halo_lens_for<input_shape>();
    EXPECT(hl[2] == 6);
    EXPECT(hl[3] == 6);
}

// With padding: halo = output_lens + (input_spatial - out_spatial + total_padding)
TEST_CASE(halo_lens_with_padding)
{
    // Output 8x8, input 8x8 (same-padding conv), pad {1,1,1,1} → total_pad={0,0,2,2}
    // halo_extra = {1,1,8,8} - {1,1,8,8} + {0,0,2,2} = {0,0,2,2}
    // halo_lens = {1,1,4,4} + {0,0,2,2} = {1,1,6,6}
    using output_shape = decltype(make_4d_shape<1, 1, 8, 8>());
    using input_shape  = decltype(make_4d_shape<1, 1, 8, 8>());
    using tiler        = migraphx::spatial_tiler<1,
                                                 migraphx::index_ints<4, 4>,
                                                 output_shape,
                                                 migraphx::index_ints<1, 1, 1, 1>>;

    constexpr auto hl = tiler::template halo_lens_for<input_shape>();
    EXPECT(hl[2] == 6);
    EXPECT(hl[3] == 6);
}

// ======== ndim ========

TEST_CASE(ndim_4d)
{
    using tiler = migraphx::
        spatial_tiler<1, migraphx::index_ints<4, 4>, decltype(make_4d_shape<1, 1, 8, 8>())>;
    EXPECT(tiler::ndim() == 4);
}
