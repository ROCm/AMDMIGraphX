#ifndef MIGRAPHX_GUARD_KERNELS_ROIALIGN_HPP
#define MIGRAPHX_GUARD_KERNELS_ROIALIGN_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/basic_ops.hpp>
#include <args.hpp>

namespace migraphx {

struct max_pool
{
    MIGRAPHX_DEVICE_CONSTEXPR auto init() { return lowest(); }

    template <class T>
    MIGRAPHX_DEVICE_CONSTEXPR T op(T x, T y)
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
    MIGRAPHX_DEVICE_CONSTEXPR T op(T x, T y)
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
__device__ T
bilinear_interpolate(const T* data, const int height, const int width, float y, float x, Op pooling)
{
    // deal with cases that inverse elements are out of feature map boundary
    if(y < -1.0f || y > height || x < -1.0f || x > width)
    {
        return 0;
    }

    y          = (y <= 0) ? 0 : y;
    x          = (x <= 0) ? 0 : x;
    auto y_low = static_cast<int>(y);
    auto x_low = static_cast<int>(x);
    int y_high;
    int x_high;

    y_high = y_low + 1;
    if(y_low >= height - 1)
    {
        y = y_high = y_low = height - 1;
    }

    x_high = x_low + 1;
    if(x_low >= width - 1)
    {
        x = x_high = x_low = width - 1;
    }

    float ly = y - y_low;
    float lx = x - x_low;
    float hy = 1.0f - ly;
    float hx = 1.0f - lx;

    // do bilinear interpolation
    T v1 = data[y_low * width + x_low];
    T v2 = data[y_low * width + x_high];
    T v3 = data[y_high * width + x_low];
    T v4 = data[y_high * width + x_high];
    T w1 = static_cast<T>(hy * hx);
    T w2 = static_cast<T>(hy * lx);
    T w3 = static_cast<T>(ly * hx);
    T w4 = static_cast<T>(ly * lx);

    T val12 = pooling.op(w1 * v1, w2 * v2);
    T val34 = pooling.op(w3 * v3, w4 * v4);

    return pooling.op(val12, val34);
}

template <class T, class Op>
__device__ T calc_pooling(const T* data,
                          float roi_start_h,
                          float roi_start_w,
                          float bin_size_h,
                          float bin_size_w,
                          int ph,
                          int pw,
                          int64_t roi_bin_grid_h,
                          int64_t roi_bin_grid_w,
                          int height,
                          int width,
                          float roi_offset,
                          Op op)
{
    T output_val        = op.init();
    const int64_t count = roi_bin_grid_h * roi_bin_grid_w;
    for(int iy = 0; iy < roi_bin_grid_h; ++iy)
    {
        float y = roi_start_h + ph * bin_size_h + (iy + 0.5f) * bin_size_h / roi_bin_grid_h;
        y += roi_offset;
        for(int ix = 0; ix < roi_bin_grid_w; ++ix)
        {
            float x = roi_start_w + pw * bin_size_w + (ix + .5f) * bin_size_w / roi_bin_grid_w;
            x += roi_offset;
            auto val   = bilinear_interpolate(data, height, width, y, x, op);
            output_val = op.op(output_val, val);
        }
    }

    return op.final(output_val, count);
}

__device__ void roialign(void* in_x, void* in_rois, void* in_ind, void* y)
{
    const float roi_offset       = ROIS_OFFSET;
    const bool is_avg_pooling    = IS_AVG_POOLING;
    const int64_t sampling_ratio = SAMPLING_RATIO;
    const float spatial_scale    = SPATIAL_SCALE;
    make_tensors()(in_x, in_rois, in_ind, y)(
        [=](auto x_t, auto rois_t, auto ind_t, auto y_t) __device__ {
            auto index       = make_index();
            const auto* x    = x_t.data();
            const auto* rois = rois_t.data();
            const auto* ind  = ind_t.data();
            // auto& ind = ind_t;
            auto* out_ptr = y_t.data();

            // input shape
            auto x_lens      = x_t.get_shape().lens;
            auto channel_num = x_lens[1];
            auto height      = x_lens[2];
            auto width       = x_lens[3];

            const auto stride   = index.nglobal();
            auto out_s          = y_t.get_shape();
            auto roi_column_num = rois_t.get_shape().lens[1];
            auto pooling_height = out_s.lens[2];
            auto pooling_width  = out_s.lens[3];
            for(index_int i = index.global; i < out_s.elements(); i += stride)
            {
                auto idx = out_s.multi(i);
                int n    = idx[0];
                int c    = idx[1];
                int ph   = idx[2];
                int pw   = idx[3];

                const auto* offset_rois = rois + (n * roi_column_num);
                const auto batch_ind    = static_cast<int>(ind[n]);

                float roi_start_w = static_cast<float>(offset_rois[0] * spatial_scale);
                float roi_start_h = static_cast<float>(offset_rois[1] * spatial_scale);
                float roi_end_w   = static_cast<float>(offset_rois[2] * spatial_scale);
                float roi_end_h   = static_cast<float>(offset_rois[3] * spatial_scale);

                float roi_width  = roi_end_w - roi_start_w;
                float roi_height = roi_end_h - roi_start_h;

                roi_width  = roi_width > 1.0f ? roi_width : 1.0f;
                roi_height = roi_height > 1.0f ? roi_height : 1.0f;

                float bin_size_w = roi_width / pooling_width;
                float bin_size_h = roi_height / pooling_height;

                const auto* offset_x = x + ((batch_ind * channel_num + c) * height * width);

                // We use roi_bin_grid to sample the grid and mimic integral
                int roi_bin_grid_h =
                    (sampling_ratio > 0) ? sampling_ratio : ceilf(roi_height / pooling_height);
                int roi_bin_grid_w =
                    (sampling_ratio > 0) ? sampling_ratio : ceilf(roi_width / pooling_width);

                if(is_avg_pooling)
                {
                    out_ptr[i] = calc_pooling(offset_x,
                                              roi_start_h,
                                              roi_start_w,
                                              bin_size_h,
                                              bin_size_w,
                                              ph,
                                              pw,
                                              roi_bin_grid_h,
                                              roi_bin_grid_w,
                                              height,
                                              width,
                                              roi_offset,
                                              avg_pool{});
                }
                else
                {
                    out_ptr[i] = calc_pooling(offset_x,
                                              roi_start_h,
                                              roi_start_w,
                                              bin_size_h,
                                              bin_size_w,
                                              ph,
                                              pw,
                                              roi_bin_grid_h,
                                              roi_bin_grid_w,
                                              height,
                                              width,
                                              roi_offset,
                                              max_pool{});
                }
            }
        });
}

} // namespace migraphx
#endif
