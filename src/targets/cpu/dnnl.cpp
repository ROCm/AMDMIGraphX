#include <migraphx/cpu/dnnl.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

dnnl_context& get_dnnl_context()
{
    static dnnl_context ctx{}; // NOLINT
    return ctx;
}

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wswitch-enum"
#endif
dnnl::memory::data_type to_dnnl_memory_data_type(shape::type_t t)
{
    using dt = dnnl::memory::data_type;
    using st = shape::type_t;
    switch(t)
    {
    case st::half_type: return dt::f16;
    case st::float_type: return dt::f32;
    case st::int32_type: return dt::s32;
    case st::int8_type: return dt::s8;
    case st::uint8_type: return dt::u8;
    default: MIGRAPHX_THROW("Unsupported data type");
    }
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif

dnnl::memory::format_tag to_dnnl_memory_format_tag(std::size_t n)
{
    switch(n)
    {
    case 1: return dnnl::memory::format_tag::a;
    case 2: return dnnl::memory::format_tag::ab;
    case 3: return dnnl::memory::format_tag::abc;
    case 4: return dnnl::memory::format_tag::abcd;
    case 5: return dnnl::memory::format_tag::abcde;
    case 6: return dnnl::memory::format_tag::abcdef;
    default: MIGRAPHX_THROW("Unsupported tensor size: " + std::to_string(n));
    }
}

dnnl::memory::desc to_dnnl_memory_desc(const shape& s)
{
    return {to_dnnl_dims(s.lens()), to_dnnl_memory_data_type(s.type()), to_dnnl_dims(s.strides())};
}

dnnl::memory to_dnnl_memory(const dnnl::memory::desc& desc, const argument& a)
{
    return dnnl::memory(desc, get_dnnl_context().engine, a.data());
}

dnnl::memory to_dnnl_memory(const argument& a)
{
    return to_dnnl_memory(to_dnnl_memory_desc(a.get_shape()), a);
}

// clang-format off
#define MIGRAPHX_VISIT_DNNL_ALGO(m) \
        m(undef) \
        m(convolution_auto) \
        m(convolution_direct) \
        m(convolution_winograd) \
        m(deconvolution_direct) \
        m(deconvolution_winograd) \
        m(eltwise_relu) \
        m(eltwise_tanh) \
        m(eltwise_elu) \
        m(eltwise_square) \
        m(eltwise_abs) \
        m(eltwise_sqrt) \
        m(eltwise_swish) \
        m(eltwise_linear) \
        m(eltwise_bounded_relu) \
        m(eltwise_soft_relu) \
        m(eltwise_logistic) \
        m(eltwise_exp) \
        m(eltwise_gelu) \
        m(eltwise_gelu_tanh) \
        m(eltwise_gelu_erf) \
        m(eltwise_log) \
        m(eltwise_clip) \
        m(eltwise_pow) \
        m(eltwise_round) \
        m(eltwise_relu_use_dst_for_bwd) \
        m(eltwise_tanh_use_dst_for_bwd) \
        m(eltwise_elu_use_dst_for_bwd) \
        m(eltwise_sqrt_use_dst_for_bwd) \
        m(eltwise_logistic_use_dst_for_bwd) \
        m(eltwise_exp_use_dst_for_bwd) \
        m(lrn_across_channels) \
        m(lrn_within_channel) \
        m(pooling_max) \
        m(pooling_avg) \
        m(pooling_avg_include_padding) \
        m(pooling_avg_exclude_padding) \
        m(vanilla_rnn) \
        m(vanilla_lstm) \
        m(vanilla_gru) \
        m(lbr_gru) \
        m(binary_add) \
        m(binary_mul) \
        m(binary_max) \
        m(binary_min) \
        m(binary_div) \
        m(resampling_nearest) \
        m(resampling_linear) \
        m(reduction_max) \
        m(reduction_min) \
        m(reduction_sum) \
        m(reduction_mul) \
        m(reduction_mean) \
        m(reduction_norm_lp_max) \
        m(reduction_norm_lp_sum) \
        m(reduction_norm_lp_power_p_max) \
        m(reduction_norm_lp_power_p_sum)
// clang-format on

const std::unordered_map<std::string, dnnl::algorithm>& dnnl_algo_map()
{
    static const std::unordered_map<std::string, dnnl::algorithm> m = {
#define MIGRAPHX_DNNL_ALGO_GENERATE_VISITOR(x) {#x, dnnl::algorithm::x},
        MIGRAPHX_VISIT_DNNL_ALGO(MIGRAPHX_DNNL_ALGO_GENERATE_VISITOR)
#undef MIGRAPHX_DNNL_ALGO_GENERATE_VISITOR
    };
    return m;
}

dnnl::algorithm to_dnnl_algo(const std::string& name)
{
    if(dnnl_algo_map().count(name) == 0)
        MIGRAPHX_THROW("Missing dnnl algo: " + name);
    return dnnl_algo_map().at(name);
}

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
