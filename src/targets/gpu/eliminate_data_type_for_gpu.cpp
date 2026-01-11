#include <migraphx/gpu/eliminate_data_type_for_gpu.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/eliminate_data_type.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {


static void insert_miopen_pooling(std::set<std::string>& u)
{
#if MIGRAPHX_USE_MIOPEN
    u.insert("pooling");
#endif
}

static void insert_gemm_conv(std::set<std::string>& u)
{
    u.insert("convolution");
    u.insert("quant_convolution");
    u.insert("dot");
    u.insert("quant_dot");
}

static eliminate_data_type for_device_functions()
{
    std::set<shape::type_t> unsupported_types(shape::types().begin(), shape::types().end());
    unsupported_types.erase(shape::type_t::float_type);
    unsupported_types.erase(shape::type_t::half_type);
    unsupported_types.erase(shape::type_t::bool_type);
    unsupported_types.erase(shape::type_t::int8_type);
    unsupported_types.erase(shape::type_t::uint8_type);
    unsupported_types.erase(shape::type_t::int32_type);
    unsupported_types.erase(shape::type_t::bf16_type);
    unsupported_types.erase(shape::type_t::tuple_type);

    std::set<std::string> device_functions =
    {
        "logsoftmax",
        "nonzero",
        "prefix_scan_sum",
        "rnn_var_sl_shift_output",
        "multinomial",
        "argmax",
        "argmin",
    };

    return eliminate_data_type{unsupported_types, shape::type_t::float_type, device_functions};
}

static eliminate_data_type for_fp8fnuz()
{
    std::set<std::string> unsupported_ops = {};

    // disable dot & quant_dot if no hipblaslt
    if(not hipblaslt_supported())
    {
        unsupported_ops.insert("dot");
        unsupported_ops.insert("quant_dot");
    }

    // MIOpen doesn't have support for fp8 pooling yet.
    insert_miopen_pooling(unsupported_ops);

    if(not gpu::gfx_has_fp8fnuz_intrinsics())
    {
        insert_gemm_conv(unsupported_ops);
    }
    return eliminate_data_type{{shape::fp8e4m3fnuz_type, shape::fp8e5m2fnuz_type}, shape::float_type, unsupported_ops};
}

static eliminate_data_type for_fp8ocp()
{
    std::set<std::string> unsupported_ops = {};

    // disable dot & quant_dot if no hipblaslt
    if(not hipblaslt_supported())
    {
        unsupported_ops.insert("dot");
        unsupported_ops.insert("quant_dot");
    }

    // MIOpen doesn't have support for fp8 pooling yet.
    insert_miopen_pooling(unsupported_ops);

    if(not gpu::gfx_has_fp8ocp_intrinsics())
    {
        insert_gemm_conv(unsupported_ops);
    }
    return eliminate_data_type{{shape::fp8e4m3fn_type, shape::fp8e5m2_type}, shape::float_type, unsupported_ops};
}

static eliminate_data_type for_gemm_conv()
{
    std::set<std::string> unsupported_ops = {};
    insert_gemm_conv(unsupported_ops);

    return eliminate_data_type{{shape::bool_type,
shape::uint16_type,
shape::int16_type,
shape::int64_type,
shape::uint64_type,
shape::double_type,}, shape::float_type, unsupported_ops};
}

void eliminate_data_type_for_gpu::apply(module_pass_manager& mpm) const
{
    std::set<shape::type_t> unsupported_types;
    // No BF-16 Support on Navi21
    if(not gpu::gfx_has_bf16_intrinsics())
    {
        unsupported_types.insert(shape::type_t::bf16_type);
    }
    if(not unsupported_types.empty())
        mpm.run_pass(eliminate_data_type{unsupported_types, shape::type_t::float_type});

    // workaround for rocBLAS unsupported error when using uint8 in quant_dot, quant_convolution & pooling
    mpm.run_pass(eliminate_data_type{{shape::uint8_type}, shape::float_type, {"quant_convolution", "quant_dot", "pooling"}});

    mpm.run_pass(for_device_functions());
    
    mpm.run_pass(for_fp8fnuz());
    mpm.run_pass(for_fp8ocp());
    
    mpm.run_pass(for_gemm_conv());
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
