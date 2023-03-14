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

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/gemm_add_add_fastgelu.hpp"
#include "ck/library/tensor_operation_instance/solution_instances/gemm_multiple_d_xdlop_cshuffle.hpp"

#include <iostream>

const std::vector<std::string>&
get_instance(std::size_t i, const std::function<bool(const std::vector<std::string>&)>& pred);

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace gpu {

using namespace migraphx::gpu::gen; // NOLINT

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_LOG_CK_GEMM);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_CK_TUNING);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_CK_TUNING_VALUE);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_CK_DEBUG);

// NOLINTNEXTLINE
static const char* const ck_gemm_kernel = R"__migraphx__(
#include <args.hpp>
#include <migraphx/kernels/ck_gemm.hpp>
#include <migraphx/kernels/pointwise.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle.hpp>

namespace migraphx {

${preamble}

extern "C" {

__global__ void ${kernel}(${params})
{
    transform_args(make_tensors(), rotate_last())(${args})([](auto... xs) {
        ck_gemm<${solution}, ${blocks_per_batch}>(xs...);
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
        return int_div_ceil(config[0], get_pb(0)) * int_div_ceil(config[1], get_pb(1));
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

static bool transposed_matrix(const shape& s) { return s.strides().back() != 1; }

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

static float matrix_distance(const shape& x, const shape& y)
{
    if(x.type() != y.type())
        return std::numeric_limits<float>::max();
    if(transposed_matrix(x) != transposed_matrix(y))
        return std::numeric_limits<float>::max();
    auto sum_squared = std::inner_product(x.lens().rbegin(),
                                          x.lens().rbegin() + 2,
                                          y.lens().rbegin(),
                                          0,
                                          std::plus<>{},
                                          [](auto a, auto b) { return (a - b) * (a - b); });
    return std::sqrt(sum_squared);
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
        std::vector<std::pair<float, std::size_t>> w;
        std::transform(tuning.begin(), tuning.end(), std::back_inserter(w), [&](const auto& p) {
            if(inputs.size() < 3 or p.first.size() < 3)
                MIGRAPHX_THROW("Invalid CK config");
            auto avg_distance = std::inner_product(
                p.first.begin(),
                p.first.begin() + 3,
                inputs.begin(),
                0.0f,
                std::plus<>{},
                [](const auto& x, const auto& y) { return matrix_distance(x, y) / 3.0f; });
            return std::make_pair(avg_distance, p.second);
        });
        std::sort(w.begin(), w.end());
        std::size_t default_value = 4;
        if(not w.empty())
            default_value = w.front().second;
        auto tuning_val = value_of(MIGRAPHX_CK_TUNING_VALUE{}, default_value);
        std::cout << "*********** Warning: CK try tuning: " << tuning_val << std::endl;
        return tuning_val;
    }
    return it->second;
}

struct ck_gemm_compiler : compiler<ck_gemm_compiler>
{
    static std::string get_layout(const shape& s)
    {
        return transposed_matrix(s) ? "ck::tensor_layout::gemm::ColumnMajor"
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

    static std::vector<shape> adjust_inputs(std::vector<shape> inputs, bool& swap_inputs)
    {
        swap_inputs  = false;
        auto c_shape = inputs.back();
        if(not transposed_matrix(c_shape))
            return inputs;
        std::vector<int64_t> perm(c_shape.lens().size());
        std::iota(perm.begin(), perm.end(), 0);
        std::swap(perm[perm.size() - 1], perm[perm.size() - 2]);
        std::transform(inputs.begin(), inputs.end(), inputs.begin(), [&](shape s) {
            return reorder_shape(s, perm);
        });
        swap_inputs = true;
        return inputs;
    }

    static std::size_t get_batch_count(const shape& s)
    {
        return std::accumulate(
            s.lens().rbegin() + 2, s.lens().rend(), std::size_t{1}, std::multiplies<std::size_t>());
    }

    static void fold_batch_dims(shape& s)
    {
        auto lens = s.lens();
        if(lens.size() <= 2)
            return;
        auto batch_count = get_batch_count(s);
        auto m1          = lens.at(lens.size() - 2);
        auto m2          = lens.at(lens.size() - 1);
        if(transposed_matrix(s))
            s = shape{s.type(), {m1, m2 * batch_count}};
        else
            s = shape{s.type(), {m1 * batch_count, m2}};
    }

    static void remove_batch_dims(shape& s)
    {
        auto lens = s.lens();
        if(lens.size() <= 2)
            return;
        auto m1 = lens.at(lens.size() - 2);
        auto m2 = lens.at(lens.size() - 1);
        s       = shape{s.type(), {m1, m2}};
    }

    std::vector<std::string> names() const { return {"ck_gemm", "gpu::ck_gemm"}; }

    operation compile_op(context& /* ctx */, const std::vector<shape>& inputs, const value& v) const
    {
        auto a_shape      = inputs[0];
        auto b_shape      = inputs[1];
        auto c_shape      = inputs.back();
        auto tuning_value = get_tuning_for({a_shape, b_shape, c_shape});

        auto rank           = a_shape.lens().size();
        auto b_strides      = b_shape.strides();
        bool can_fold_batch = rank >= 3 and b_strides[rank - 3] == 0;

        auto batch_count = get_batch_count(c_shape);
        auto m           = c_shape.lens()[rank - 2];
        m                = can_fold_batch ? m * batch_count : m;
        auto n           = c_shape.lens().back();
        auto k           = a_shape.lens().back();

        const auto numDTensors = inputs.size() - 3;
        const bool transA      = transposed_matrix(a_shape);
        const bool transB      = transposed_matrix(b_shape);
        const bool transCDE    = transposed_matrix(c_shape);
        const auto a_type      = get_type(a_shape);
        const auto b_type      = get_type(b_shape);
        const auto cde_type =
            ck_tuple(inputs.begin() + 2, inputs.end() - 1, &get_type); // get_type(c_shape);
        const auto cde_layout = ck_tuple(inputs.begin() + 2, inputs.end() - 1, &get_layout);

        std::string ck_passthrough =
            "ck_passthrough"; //"ck::tensor_operation::element_wise::PassThrough";
        std::string cde_op = ck_passthrough;
        assert(inputs.size() < 4 or v.contains("post"));
        if(v.contains("post"))
        {
            cde_op = v.at("post").to<std::string>();
        }

        auto problem =
            ck::tensor_operation::device::instance::Problem{static_cast<ck::index_t>(m),
                                                            static_cast<ck::index_t>(n),
                                                            static_cast<ck::index_t>(k),
                                                            static_cast<ck::index_t>(numDTensors),
                                                            static_cast<ck::index_t>(tuning_value),
                                                            transA,
                                                            transB,
                                                            transCDE,
                                                            a_type,
                                                            b_type,
                                                            cde_type,
                                                            ck_passthrough,
                                                            ck_passthrough,
                                                            cde_op,
                                                            cde_layout};
        const auto solution   = problem.GetSolution();
        auto blocks_per_batch = problem.GetGridSize();
        auto block_size       = problem.GetBlockSize();

        hip_compile_options options;
        auto grid_size = can_fold_batch ? blocks_per_batch : batch_count * blocks_per_batch;
        options.set_launch_params(v, grid_size * block_size, block_size);
        options.inputs         = inputs;
        options.output         = c_shape;
        options.kernel_name    = v.get("kernel", "ck_gemm_kernel");
        options.virtual_inputs = inputs;
        if(can_fold_batch)
        {
            auto vinputs = inputs;
            fold_batch_dims(vinputs[0]);
            remove_batch_dims(vinputs[1]);
            std::for_each(vinputs.begin() + 2, vinputs.end(), fold_batch_dims);
            options.virtual_inputs = vinputs;
        }

        if(v.get("check", false) or enabled(MIGRAPHX_CK_DEBUG{}))
            options.params += " -DMIGRAPHX_CK_CHECK=1";

        auto src = interpolate_string(ck_gemm_kernel,
                                      {{"solution", solution},
                                       {"params", enum_params(inputs.size(), "void * private_p")},
                                       {"args", enum_params(inputs.size(), "private_p")},
                                       {"blocks_per_batch", to_string(blocks_per_batch)},
                                       {"preamble", v.get("preamble", std::string{})},
                                       {"kernel", options.kernel_name}});
        std::cout << src << std::endl;
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
