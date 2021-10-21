#ifndef MIGRAPHX_GUARD_KERNELS_ROIALIGN_HPP
#define MIGRAPHX_GUARD_KERNELS_ROIALIGN_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/dfor.hpp>
#include <migraphx/kernels/basic_ops.hpp>
#include <args.hpp>
#include <numeric>

namespace migraphx {

struct max_pool
{
    MIGRAPHX_DEVICE_CONSTEXPR auto init() { return lowest(); }

    template <class T>
    MIGRAPHX_DEVICE_CONSTEXPR T operator()(T x, T y)
    {
        return x > y ? x : y;
    }

    template <class T>
    MIGRAPHX_DEVICE_CONSTEXPR T final(T x, std::size_t)
    {
        return (x);
    }
};

struct avg_pool
{
    MIGRAPHX_DEVICE_CONSTEXPR auto init() { return 0.0; }

    template <class T>
    MIGRAPHX_DEVICE_CONSTEXPR T operator()(T x, T y)
    {
        return x + y;
    }

    template <class T>
    MIGRAPHX_DEVICE_CONSTEXPR T final(T x, std::size_t y)
    {
        return (y == 0) ? 0.0 : (x / y);
    }
};

template <class T, class Op>
MIGRAPHX_DEVICE_CONSTEXPR T bilinear_interpolate(const T* data,
                                                 const std::array<std::size_t, 2>& dims,
                                                 std::array<float, 2> xy,
                                                 Op pooling)
{
    std::array<int, 2> low{};
    std::array<int, 2> high{};
    max max_op{};
    for(std::size_t ii = 0; ii < xy.size(); ++ii)
    {
        if(xy[ii] < -1.0f or xy[ii] > dims[ii])
        {
            return 0;
        }

        xy[ii]   = max_op(xy[ii], 0);
        low[ii]  = xy[ii];
        high[ii] = low[ii] + 1;
        if(low[ii] >= dims[ii] - 1)
        {
            xy[ii] = high[ii] = low[ii] = dims[ii] - 1;
        }
    }

    float ly = xy[0] - low[0];
    float lx = xy[1] - low[1];
    float hy = 1.0f - ly;
    float hx = 1.0f - lx;

    std::array<std::size_t, 4> locs = {low[0] * dims[1] + low[1],
                                       low[0] * dims[1] + high[1],
                                       high[0] * dims[1] + low[1],
                                       high[0] * dims[1] + high[1]};
    std::array<T, 4> ws             = {hy * hx, hy * lx, ly * hx, ly * lx};
    auto v01                        = pooling(data[locs[0]] * ws[0], data[locs[1]] * ws[1]);
    auto v23                        = pooling(data[locs[2]] * ws[2], data[locs[3]] * ws[3]);
    return pooling(v01, v23);

    // std::array<T, 4> vals;
    // std::transform(locs.begin(), locs.end(), ws.begin(), vals.begin(), [&](auto pos, auto w){
    //     return data[pos] * w;
    // });

    // T ini_val = pooling.init();
    // return std::accumulate(vals.begin(), vals.end(), ini_val, pooling);
}

template <class T, class Op>
MIGRAPHX_DEVICE_CONSTEXPR T calc_pooling(const T*& data,
                                         const std::array<float, 2>& roi_starts,
                                         const std::array<float, 2>& bin_size,
                                         const std::array<int, 2>& idx,
                                         const std::array<std::size_t, 2>& bin_grid_size,
                                         const std::array<std::size_t, 2>& dims,
                                         float roi_offset,
                                         Op op)
{
    T output_val        = op.init();
    const int64_t count = bin_grid_size[0] * bin_grid_size[1];
    dfor(bin_grid_size[0], bin_grid_size[1])([&](auto iy, auto ix) {
        std::array<float, 2> locs;

        locs[0] =
            roi_starts[0] + idx[0] * bin_size[0] + (iy + 0.5f) * bin_size[0] / bin_grid_size[0];
        locs[1] =
            roi_starts[1] + idx[1] * bin_size[1] + (ix + .5f) * bin_size[1] / bin_grid_size[1];

        locs[0] += roi_offset;
        locs[1] += roi_offset;
        auto val   = bilinear_interpolate(data, dims, locs, op);
        output_val = op(output_val, val);
    });

    // for(int iy = 0; iy < roi_bin_grid_h; ++iy)
    // {
    //     float y = roi_start_h + ph * bin_size_h + (iy + 0.5f) * bin_size_h / roi_bin_grid_h;
    //     y += roi_offset;
    //     for(int ix = 0; ix < roi_bin_grid_w; ++ix)
    //     {
    //         float x = roi_start_w + pw * bin_size_w + (ix + .5f) * bin_size_w / roi_bin_grid_w;
    //         x += roi_offset;
    //         auto val   = bilinear_interpolate(data, height, width, y, x, op);
    //         output_val = op.op(output_val, val);
    //     }
    // }

    return op.final(output_val, count);
}

__device__ void roialign(void* in_x, void* in_rois, void* in_ind, void* y)
{
    const float roi_offset       = ROIS_OFFSET;
    const bool is_avg_pooling    = IS_AVG_POOLING;
    const int64_t sampling_ratio = SAMPLING_RATIO;
    const float spatial_scale    = SPATIAL_SCALE;
    make_tensors()(
        in_x, in_rois, in_ind, y)([=](auto x_t, auto rois_t, auto ind_t, auto y_t) __device__ {
        auto index       = make_index();
        const auto* x    = x_t.data();
        const auto* rois = rois_t.data();
        const auto* ind  = ind_t.data();

        auto* out_ptr = y_t.data();

        // input shape
        auto x_lens      = x_t.get_shape().lens;
        auto channel_num = x_lens[1];
        // auto height      = x_lens[2];
        // auto width       = x_lens[3];
        // input dims of height and width, in all 2-dim arrays, the first dim
        // is for height and second dim is for width
        std::array<std::size_t, 2> in_dims = {x_lens[2], x_lens[3]};

        const auto stride   = index.nglobal();
        auto out_s          = y_t.get_shape();
        auto roi_column_num = rois_t.get_shape().lens[1];
        // auto pooling_height = out_s.lens[2];
        // auto pooling_width  = out_s.lens[3];

        // output dims of height and width, in all 2-dim arrays, the first dim
        // is for height and second dim is for width
        const auto& out_lens                = out_s.lens;
        std::array<std::size_t, 2> out_dims = {out_lens[2], out_lens[3]};

        for(index_int i = index.global; i < out_s.elements(); i += stride)
        {
            auto idx = out_s.multi(i);
            int n    = idx[0];
            int c    = idx[1];
            int ph   = idx[2];
            int pw   = idx[3];

            const auto* offset_rois = rois + (n * roi_column_num);
            const int batch_ind     = ind[n];

            std::array<float, 2> roi_starts = {offset_rois[1] * spatial_scale,
                                               offset_rois[0] * spatial_scale};
            std::array<float, 2> roi_ends   = {offset_rois[3] * spatial_scale,
                                             offset_rois[2] * spatial_scale};

            std::array<float, 2> roi_size{};
            std::array<float, 2> bin_size{};
            std::array<std::size_t, 2> bin_grid_size{};

            for(std::size_t ii = 0; ii < roi_size.size(); ++ii)
            {
                roi_size[ii] = roi_ends[ii] - roi_starts[ii];
                roi_size[ii] = std::max(roi_size[ii], 1.0f);

                bin_size[ii] = roi_size[ii] / out_dims[ii];
                bin_grid_size[ii] =
                    (sampling_ratio > 0) ? sampling_ratio : std::ceil(roi_size[ii] / out_dims[ii]);
            }

            // float roi_width  = roi_end_w - roi_start_w;
            // float roi_height = roi_end_h - roi_start_h;

            // roi_width  = roi_width > 1.0f ? roi_width : 1.0f;
            // roi_height = roi_height > 1.0f ? roi_height : 1.0f;

            // float bin_size_w = roi_width / pooling_width;
            // float bin_size_h = roi_height / pooling_height;

            // We use roi_bin_grid to sample the grid and mimic integral
            // int roi_bin_grid_h =
            //     (sampling_ratio > 0) ? sampling_ratio : ceilf(roi_height / pooling_height);
            // int roi_bin_grid_w =
            //     (sampling_ratio > 0) ? sampling_ratio : ceilf(roi_width / pooling_width);

            const auto* offset_x = x + ((batch_ind * channel_num + c) * in_dims[0] * in_dims[1]);

            if(is_avg_pooling)
            {
                out_ptr[i] = calc_pooling(offset_x,
                                          roi_starts,
                                          bin_size,
                                          {ph, pw},
                                          bin_grid_size,
                                          in_dims,
                                          roi_offset,
                                          avg_pool{});
            }
            else
            {
                out_ptr[i] = calc_pooling(offset_x,
                                          roi_starts,
                                          bin_size,
                                          {ph, pw},
                                          bin_grid_size,
                                          in_dims,
                                          roi_offset,
                                          max_pool{});
            }
        }
    });
}

} // namespace migraphx
#endif
