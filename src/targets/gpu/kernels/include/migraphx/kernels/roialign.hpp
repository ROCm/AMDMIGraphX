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
    for(std::size_t ii = 0; ii < xy.size(); ++ii)
    {
        if(xy[ii] < -1.0f or xy[ii] > dims[ii])
        {
            return 0;
        }

        xy[ii]   = std::max(xy[ii], 0.0f);
        low[ii]  = xy[ii];
        high[ii] = low[ii] + 1;
        if(low[ii] >= dims[ii] - 1)
        {
            xy[ii] = high[ii] = low[ii] = dims[ii] - 1;
        }
    }
    std::array<std::size_t, 4> locs = {low[0] * dims[1] + low[1],
                                       low[0] * dims[1] + high[1],
                                       high[0] * dims[1] + low[1],
                                       high[0] * dims[1] + high[1]};

    float ly            = xy[0] - low[0];
    float lx            = xy[1] - low[1];
    float hy            = 1.0f - ly;
    float hx            = 1.0f - lx;
    std::array<T, 4> ws = {hy * hx, hy * lx, ly * hx, ly * lx};

    auto v01 = pooling(data[locs[0]] * ws[0], data[locs[1]] * ws[1]);
    auto v23 = pooling(data[locs[2]] * ws[2], data[locs[3]] * ws[3]);
    return pooling(v01, v23);
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
        std::array<std::size_t, 2> id = {iy, ix};
        for(std::size_t i = 0; i < idx.size(); ++i)
        {
            locs[i] = roi_starts[i] + idx[i] * bin_size[i] +
                      (id[i] + 0.5f) * bin_size[i] / bin_grid_size[i];
            locs[i] += roi_offset;
        }

        auto val   = bilinear_interpolate(data, dims, locs, op);
        output_val = op(output_val, val);
    });
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
        // input dims of height and width, in all 2-dim arrays, the first dim
        // is for height and second dim is for width
        std::array<std::size_t, 2> in_dims = {x_lens[2], x_lens[3]};

        const auto stride   = index.nglobal();
        auto out_s          = y_t.get_shape();
        auto roi_column_num = rois_t.get_shape().lens[1];

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
