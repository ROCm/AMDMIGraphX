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
#include <fstream>
#include <filesystem>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/gpu/context.hpp>

#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/compile_gen.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/env.hpp>
#include <migraphx/reduce_dims.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/module.hpp>
#include <migraphx/env.hpp>
#include <migraphx/file_buffer.hpp>

const std::vector<std::string>&
get_instance(std::size_t i, const std::function<bool(const std::vector<std::string>&)>& pred);

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace gpu {

using namespace migraphx::gpu::gen; // NOLINT

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_LOG_CK_GEMM);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_CK_TUNING);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_CK_DEBUG);

// NOLINTNEXTLINE
static const char* const ck_gemm_kernel = R"__migraphx__(
#include <args.hpp>
#include <migraphx/kernels/ck_gemm.hpp>
#include <migraphx/kernels/pointwise.hpp>

namespace migraphx {

${preamble}

extern "C" {

__global__ void ${kernel}(${params})
{
    transform_args(make_tensors(), rotate_last())(${args})([](auto... xs) {
        ck_gemm<CK_DeviceGemmMultipleD<${instance}>, ${blocks_per_batch}>(xs...);
    });
}

}

} // namespace migraphx

)__migraphx__";

static std::size_t int_div_ceil(std::size_t x, std::size_t y) { return (x + y - 1) / y; }

struct instance
{
    std::vector<std::string> params;
    static const std::size_t block_size_index = 15;

    std::size_t int_at(std::size_t i) const { return std::stoull(params[i]); }

    std::size_t get_block_size() const { return int_at(block_size_index); }

    std::size_t get_pb(std::size_t i) const
    {
        assert(i < 4);
        return int_at(block_size_index + 1 + i);
    }

    std::array<std::size_t, 3> get_pad(const std::array<std::size_t, 3>& config) const
    {
        std::array<std::size_t, 3> result{};
        for(auto i : range(config.size()))
        {
            result[i] = int_div_ceil(config[i], get_pb(i)) * get_pb(i) - config[i];
        }
        return result;
    }

    std::size_t get_grid_size(const std::array<std::size_t, 3>& config) const
    {
        return int_div_ceil(config[0], get_pb(1)) * int_div_ceil(config[1], get_pb(1));
    }

    void set_ds_layout(const std::string& s)
    {
        assert(params[2] == "ck::Tuple<>");
        params[2] = s;
    }

    void set_ds_type(const std::string& s)
    {
        assert(params[8] == "ck::Tuple<>");
        params[8] = s;
    }

    void set_ds_op(const std::string& s)
    {
        assert(params[12] == "ck_passthrough");
        params[12] = s;
    }

    void set_gemm(const std::string& s)
    {
        assert(params[13] == "ck::tensor_operation::device::GemmSpecialization::Default");
        params[13] = s;
    }

    std::string str() const { return join_strings(params, ","); }
};

template <class F, class Action>
auto action_decorate(F f, Action action)
{
    return [=](auto&&... xs) {
        action();
        f(std::forward<decltype(xs)>(xs)...);
    };
}

using tuning_entry = std::pair<std::vector<shape>, size_t>;
static std::vector<tuning_entry> read_tuning(const std::string& s)
{
    if(not fs::exists(s))
        return {};
    return from_value<std::vector<tuning_entry>>(from_json_string(read_string(s)));
}

static std::size_t get_tuning_for(const std::vector<shape>& inputs)
{
    static auto tuning = read_tuning(string_value_of(MIGRAPHX_CK_TUNING{}, ""));
    if(tuning.empty())
        std::cout << "*********** Warning: No CK tuning!" << std::endl;
    auto it = std::find_if(
        tuning.begin(), tuning.end(), [&](const auto& p) { return p.first == inputs; });
    if(it == tuning.end())
    {
        std::cout << "*********** Warning: CK tuning missing for config!" << std::endl;
        return 4;
    }
    return it->second;
}

struct ck_gemm_compiler : compiler<ck_gemm_compiler>
{
    static std::string get_layout(const shape& s)
    {
        return s.transposed() ? "ck::tensor_layout::gemm::ColumnMajor"
                              : "ck::tensor_layout::gemm::RowMajor";
    }

    static std::string get_type(const shape& s)
    {
        if(s.type() == shape::half_type)
            return "ck::half_t";
        return shape::cpp_type(s.type());
    }

    template <class Iterator, class F>
    static std::string ck_tuple(Iterator start, Iterator last, F f)
    {
        std::vector<std::string> s;
        std::transform(start, last, std::back_inserter(s), f);
        return "ck::Tuple<" + join_strings(s, ",") + ">";
    }

    std::vector<std::string> names() const { return {"ck_gemm", "gpu::ck_gemm"}; }

    operation compile_op(context& /* ctx */, const std::vector<shape>& inputs, const value& v) const
    {
        auto a_shape = inputs[0];
        auto b_shape = inputs[1];
        auto c_shape = inputs.back();

        auto rank = a_shape.lens().size();

        std::array<char, 3> keys{'M', 'N', 'K'};
        std::array<std::size_t, 3> config{
            c_shape.lens()[rank - 2], c_shape.lens().back(), a_shape.lens().back()};

        auto tuning_val = v.get("tuning_val", get_tuning_for({a_shape, b_shape, c_shape}));
        auto ip         = instance{get_instance(tuning_val, [&](const auto& x) -> bool {
            return get_layout(a_shape) == x[0] and get_layout(b_shape) == x[1] and
                   get_layout(c_shape) == x[3] and get_type(a_shape) == x[4] and
                   get_type(b_shape) == x[5] and get_type(c_shape) == x[9];
        })};
        assert(inputs.size() < 4 or v.contains("post"));
        if(v.contains("post"))
        {
            ip.set_ds_layout(ck_tuple(inputs.begin() + 2, inputs.end() - 1, &get_layout));
            ip.set_ds_type(ck_tuple(inputs.begin() + 2, inputs.end() - 1, &get_type));
            ip.set_ds_op(v.at("post").to<std::string>());
        }

        auto padding = ip.get_pad(config);
        std::string gemm_type;
        for(auto i : range(padding.size()))
        {
            if(padding[i] != 0)
                gemm_type += keys[i];
        }
        if(gemm_type.empty())
            gemm_type = "Default";
        else
            gemm_type += "Padding";
        ip.set_gemm("ck::tensor_operation::device::GemmSpecialization::" + gemm_type);

        auto blocks_per_batch = ip.get_grid_size(config);
        auto batch_count = std::accumulate(
            c_shape.lens().rbegin() + 2, c_shape.lens().rend(), std::size_t{1}, std::multiplies<std::size_t>());


        hip_compile_options options;
        auto block_size = ip.get_block_size();
        auto grid_size  = batch_count * blocks_per_batch;
        options.set_launch_params(v, grid_size * block_size, block_size);
        options.inputs         = inputs;
        options.output         = c_shape;
        options.kernel_name    = v.get("kernel", "ck_gemm_kernel");
        options.virtual_inputs = inputs;

        if(v.get("check", false) or enabled(MIGRAPHX_CK_DEBUG{}))
            options.params += " -DMIGRAPHX_CK_CHECK=1";

        auto src = interpolate_string(ck_gemm_kernel,
                                      {{"instance", ip.str()},
                                       {"params", enum_params(inputs.size(), "void * private_p")},
                                       {"args", enum_params(inputs.size(), "private_p")},
                                       {"blocks_per_batch", to_string(blocks_per_batch)},
                                       {"preamble", v.get("preamble", std::string{})},
                                       {"kernel", options.kernel_name}});

        return compile_hip_code_object(src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        auto v      = op.to_value();
        v["kernel"] = "ck_gemm_kernel";
        if(not ins->module_inputs().empty())
        {
            auto* pm      = ins->module_inputs().front();
            v["preamble"] = generate_pointwise(*pm, "post_ck_gemm_function") +
                            "\nMIGRAPHX_LIFT_CLASS(post_ck_gemm, post_ck_gemm_function);";
            v["post"]   = "ck_function_adaptor<post_ck_gemm>";
            v["kernel"] = "ck_gemm_" + generate_name_from_ops(*pm) + "_kernel";
        }

        auto shapes = to_shapes(ins->inputs());
        return action_decorate(replace(compile_op(ctx, shapes, v)), [=] {
            if(enabled(MIGRAPHX_LOG_CK_GEMM{}))
            {
                std::vector<shape> gemm_shapes{shapes[0], shapes[1], shapes.back()};
                std::cout << "ck_gemm: " << to_json_string(to_value(gemm_shapes)) << std::endl;
            }
        });
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
