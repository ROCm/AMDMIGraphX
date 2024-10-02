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
#ifndef MIGRAPHX_GUARD_OPERATORS_ROIALIGN_HPP
#define MIGRAPHX_GUARD_OPERATORS_ROIALIGN_HPP

#include <limits>
#include <migraphx/check_shapes.hpp>
#include <migraphx/op/common.hpp>
#include <migraphx/config.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/dfor.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/shape_for_each.hpp>
#include <array>
#include <cmath>
#include <numeric>
#include <utility>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct roialign
{
    std::string coord_trans_mode = "half_pixel";
    pooling_mode mode            = {pooling_mode::average};
    int64_t output_height        = 1;
    int64_t output_width         = 1;
    int64_t sampling_ratio       = 0;
    float spatial_scale          = 1.0f;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.coord_trans_mode, "coordinate_transformation_mode"),
                    f(self.mode, "mode"),
                    f(self.output_height, "output_height"),
                    f(self.output_width, "output_width"),
                    f(self.sampling_ratio, "sampling_ratio"),
                    f(self.spatial_scale, "spatial_scale"));
    }

    std::string name() const { return "roialign"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(3);
        auto x_lens   = inputs.at(0).lens();
        auto roi_lens = inputs.at(1).lens();
        auto bi_lens  = inputs.at(2).lens();
        auto type     = inputs.at(0).type();

        // check input correct
        if(inputs.at(0).type() != shape::float_type or inputs.at(1).type() != shape::float_type or inputs.at(2).type() != shape::int64_type)
        {
            MIGRAPHX_THROW("ROIALIGN: incorrect type for input 1 or 2 or 3!");
        }

        if(bi_lens.size() != 1)
        {
            MIGRAPHX_THROW("ROIALIGN: batch indices should be 1 dimension!");
        }

        if(roi_lens.size() != 2 or roi_lens.at(1) != 4)
        {
            MIGRAPHX_THROW(
                "ROIALIGN: rois should be 2 dimensions, and the second dim should be 4!");
        }

        if(roi_lens.front() != bi_lens.front())
        {
            MIGRAPHX_THROW("ROIALIGN: rois and batch indices inputs should have the same number!");
        }

        std::vector<std::size_t> out_lens = x_lens;
        out_lens[0]                       = roi_lens[0];
        out_lens[2]                       = output_width;
        out_lens[3]                       = output_height;

        return {type, out_lens};
    }

    struct pos_weight
    {
        // neighbor indices for the bilinear interpolation
        std::array<std::size_t, 4> pos = {0, 0, 0, 0};
        // neighbor weights for the bilinear interpolation
        std::array<float, 4> w = {0.0f, 0.0f, 0.0f, 0.0f};
    };

    auto calc_pos_weight(const std::array<std::size_t, 2>& dims,
                         const shape& comp_s,
                         const std::array<float, 2>& roi_start,
                         const std::array<float, 2>& bin_size,
                         const std::array<std::size_t, 2>& bin_grid_size) const
    {
        std::vector<pos_weight> results(bin_grid_size[0] * bin_grid_size[1] * output_height *
                                        output_width);
std::vector<std::size_t> temp_lens = comp_s.lens();                                        
shape temp_s = {shape::float_type,{temp_lens[1], temp_lens[0], temp_lens[3], temp_lens[2] }};
        shape_for_each(comp_s, [&](const auto& idx_v, size_t index) {

            // The p and i indexes correspond to nested looping parameters in ORT that go in y, x order.  The i[x] value is least significant
            // and iterates the fastest.
            std::array<std::size_t, 2> p = {idx_v[1], idx_v[0]};
            std::array<std::size_t, 2> i = {idx_v[3], idx_v[2]};//  <== these are always the same
// printf("\n IIIII other index %lu , %lu , %lu , %lu  i=%lu   temp_index = %lu \n", p[0], p[1], i[0], i[1], index, temp_s.index({p[0], p[1], i[0], i[1]}));
// printf(" my index= %lu  reverse temp=%lu\n ", comp_s.index({p[1], p[0], i[1], i[0]}), temp_s.index({p[1], p[0], i[1], i[0]}));
// printf(" more index= %lu  reverse ...=%lu\n ", comp_s.index({p[0], p[1], i[0], i[1]}), temp_s.index({p[0], p[1], i[0], i[1]}));
            // xy is scaled coordinates of start point of ROI
            std::array<float, 2> xy{};
            // low, high are floor and ceiling of the xy value (i.e. the bounds of the pixel it lies inside)
            std::array<int64_t, 2> low{};
            std::array<int64_t, 2> high{};

            // size_t adj_index = temp_s.index({p[1], p[0], i[1], i[0]});

            for(auto ii : range(p.size()))
            {
    // if(ii == 0)
    // printf("x: " );
    // else
    // printf("y: " );
                // for width & height dimensions,
                // transform the roi start point to scaled coordinates
// printf("    roi_start[ii] %f    p[ii]  %lu   bin_size[ii] %f   (i[ii] + .5f) %f      bin_grid_size[ii] %lu       \n",
// roi_start[ii], p[ii], bin_size[ii], (i[ii] + .5f),     bin_grid_size[ii] );

                xy[ii] = roi_start[ii] + p[ii] * bin_size[ii] +
                         (i[ii] + .5f) * bin_size[ii] / bin_grid_size[ii];
// printf(" QQQQQQ  L137 x=%f  y=%f  ", xy[0], xy[1]);                                        
                xy[ii] = (coord_trans_mode != "half_pixel") ? (xy[ii] - 0.5f) : xy[ii];
// printf(" L139 %f ", xy[ii]);                        
                if(xy[ii] < -1.0 or xy[ii] > dims[ii])
                {
// printf(" L142 results = pos_weight i=%lu dims=%lu, %lu  \n ", index,  dims[0], dims[1]);                    
                    // results[adj_index] = pos_weight{};  // all zeroes
                    results[index] = pos_weight{};  // all zeroes
                    return;
                }

                xy[ii]   = std::max(xy[ii], 0.0f);
                low[ii]  = xy[ii];
                high[ii] = low[ii] + 1;
// printf(" L148 %f  low[ii] %lu, dims[ii] %lu", xy[ii],  low[ii], dims[ii]);                
                if(low[ii] >= dims[ii] - 1)
                {
                    xy[ii] = high[ii] = low[ii] = dims[ii] - 1;
// printf(" L154 %f ", xy[ii]);                    
                }
// printf(" \n");                
            }
            // printf(" JJJJJ  xy[0]=%f  xy[1] = %f                             dims[1]=%lu  low%ld-%ld  high %ld-%ld   i=%zu      dims[0]=%lu \n\n",
            //                 xy[0], xy[1], dims[1], low[1], low[0],  high[1], high[0], index, dims[0]);
            results[index].pos = {low[1] * dims[0] + low[0],
                                  low[1] * dims[0] + high[0],
                                  high[1] * dims[0] + low[0],
                                  high[1] * dims[0] + high[0]};

            float lx = xy[0] - low[0];
            float ly = xy[1] - low[1];
            float hy = 1.0f - ly;
            float hx = 1.0f - lx;
            // printf(" HHHHH partial pixel values, index=%lu pci=%lu  ly=%f, lx=%f, hy=%f, hx=%f\n\n", index, temp_s.index({p[1], p[0], i[1], i[0]}), 
            //    ly, lx, hy, hx);
            // save weights and indices
            results[index].w = {hy * hx, hy * lx, ly * hx, ly * lx};
// printf(" DDDDD index %d    %f  %f  %f  %f \n", pre_calc_index,
//     float(pc.w1), float(pc.w2), float(pc.w3), float(pc.w4));

        });
// // printf(" AAAAA here we are\n");
//         for(int iix = 0; iix < results.size(); iix++)
//           printf(" SSSSS %d    %lu  %lu  %lu  %lu   %f  %f  %f  %f\n", iix, results[iix].pos[0], results[iix].pos[1], results[iix].pos[2], results[iix].pos[3],
//                    results[iix].w[0], results[iix].w[1], results[iix].w[2], results[iix].w[3]);

        return results;
    }

    struct max_pool
    {
        double init() { return std::numeric_limits<double>::lowest(); }

        double operator()(double x, double y) { return std::max(x, y); }

        double final(double x, std::size_t) { return (x); }
    };

    struct avg_pool
    {
        double init() { return 0.0; }

        double operator()(double x, double y) { return x + y; }

        double final(double x, std::size_t y) { return (y == 0) ? 0.0 : (x / y); }
    };

    // Calculate a pooling value for 1 block of bin_grid_size*bin_grid_size weights
    template <class T, class Op>
    double calc_pooling(const T& data,
                                             const std::array<std::size_t, 2>& bin_grid_size,
                                             const std::vector<pos_weight>& pos_weights,
                                             int64_t& index,
                                             Op op) const
    {
        double output_val   = op.init();
        const int64_t count = bin_grid_size[0] * bin_grid_size[1];
        dfor(bin_grid_size[0], bin_grid_size[1])([&](auto, auto) {
            const auto& pc = pos_weights[index];
            std::array<double, 4> wv;
            // printf(" WWWWW ");
            std::transform(
                pc.w.begin(), pc.w.end(), pc.pos.begin(), wv.begin(), [&](auto w, auto pos) {



// std::cout << " YYYYY data starting at " << &(*(data)) ;
// printf("  %lu, %f->%f   \n", pos, w, *(data + pos) * w);
                    return *(data + pos) * w;
                });
    // for(double aa : wv)
    //   printf(" %d   ", aa);
            // printf("\n");
            output_val = std::accumulate(wv.begin(), wv.end(), output_val, op);
            index += 1;
        });

        output_val = op.final(output_val, count);

        return output_val;
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        const auto& out_lens = output_shape.lens();
        int64_t n_rois       = out_lens[0];
        std::size_t channels = out_lens[1];
        // output dims of height and width, in all 2-dim arrays, the first dim
        // is for height and second dim is for width i.e. (y, x) order
        std::array<std::size_t, 2> out_dims = {out_lens[2], out_lens[3]};
        const auto& x_lens                  = args.at(0).get_shape().lens();
        // input dims of height and width
        std::array<std::size_t, 2> in_dims = {x_lens[3], x_lens[2]};
        auto roi_s                         = args.at(1).get_shape();

        visit_all(result, args.at(0), args.at(1))([&](auto output, auto x, auto roi) {
            const auto* batch_indices = args.at(2).cast<int64_t>();
            par_for(n_rois, [&](auto n) {
                const auto bottom_data   = x.begin();
                const auto roi_batch_ind = batch_indices[n];
                // Do not use rounding; this implementation detail is critical
                float offset = (coord_trans_mode == "half_pixel") ? 0.5 : 0.0;
                std::array<float, 2> roi_starts = {
                    static_cast<float>(roi[roi_s.index({n, 0})] * spatial_scale - offset),
                    static_cast<float>(roi[roi_s.index({n, 1})] * spatial_scale - offset)};
                std::array<float, 2> roi_ends = {
                    static_cast<float>(roi[roi_s.index({n, 2})] * spatial_scale - offset),
                    static_cast<float>(roi[roi_s.index({n, 3})] * spatial_scale - offset)};

                // Force malformed ROIs to be 1x1, output_half_pixel transform mode
                std::array<float, 2> roi_size{};
                std::array<float, 2> bin_size{};
                std::array<std::size_t, 2> bin_grid_size{};

                for(auto ii : range(roi_size.size()))
                {
                    roi_size[ii] = roi_ends[ii] - roi_starts[ii];
                    if(coord_trans_mode != "half_pixel")
                        roi_size[ii] = std::max(roi_size[ii], 1.0f);
// printf("\n KKKKK ii %ld  roi_size %f   roi_batch_ind %ld  out_dims %lu     \n", ii, roi_size[ii] , roi_batch_ind,  out_dims[ii]);
                    bin_size[ii]      = roi_size[ii] / out_dims[ii];
                    bin_grid_size[ii] = (sampling_ratio > 0)
                                            ? sampling_ratio
                                            : std::ceil(roi_size[ii] / out_dims[ii]);
                }

                // we want to precalculate indices and weights shared by all channels,
                // this is the key point of optimization
                std::vector<std::size_t> comp_lens = {
                    out_dims[1], out_dims[0], bin_grid_size[1], bin_grid_size[0]};
                shape comp_s{shape::float_type, comp_lens};
                auto pre_calc =
                    this->calc_pos_weight(in_dims, comp_s, roi_starts, bin_size, bin_grid_size);

                std::vector<std::size_t> comp_lens1 = {channels, out_dims[0], out_dims[1]};
                shape comp_s1{migraphx::shape::float_type, comp_lens1};
                std::vector<int64_t> vec_index(channels, 0);
// printf(" XXXXX  %lu    (bottom_data + %d * %ld + %ld) * %lu * %lu\n",// ORT does this for 2 channels, 2 ROI
//  static_cast<int64_t>((roi_batch_ind * channels + 0) *
//                                                            in_dims[0] * in_dims[1]),
//      int(roi_batch_ind),  channels, (size_t)0, in_dims[0], in_dims[1]);  // offset pointer to data for this ROI (4 total)
    
                    // Iterate through each dimension in [channels, out_dims[1], out_dims[2]]
                    shape_for_each(comp_s1, [&](const auto& idx) {
                    auto c  = idx[0];  // channel count
                    auto ph = idx[1];
                    auto pw = idx[2];

                    const auto offset_bottom_data =
                        bottom_data + static_cast<int64_t>((roi_batch_ind * channels + c) *
                                                           in_dims[0] * in_dims[1]);
                    double output_val;
// printf(" UUUUU  bottom_data %d  %lu %lu pre_calc size=%lu vec_index %lu    ", int(*offset_bottom_data), 
// bin_grid_size[0], bin_grid_size[1],
// pre_calc.size(), vec_index[c]);

// printf("cont.  c=%ld  ph  =  %ld  pw = %ld  n=%ld roi_batch_ind %ld\n", c, ph, pw, n, roi_batch_ind);

                    output_val =
                        (mode == migraphx::op::pooling_mode::average)
                            ? this->calc_pooling(offset_bottom_data,
                                                 bin_grid_size,
                                                 pre_calc,
                                                 vec_index[c],
                                                 avg_pool{})
                            : this->calc_pooling(offset_bottom_data,
                                                 bin_grid_size,
                                                 pre_calc,
                                                 vec_index[c],
                                                 max_pool{});
// printf(" TTTTT idx=%3ld  output_val=%f\n", vec_index[c] % 9 - 1, output_val);                                                 
                    output(n, c, ph, pw) = output_val;
                });
            });
        });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
