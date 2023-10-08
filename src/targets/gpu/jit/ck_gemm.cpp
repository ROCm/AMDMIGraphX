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
#include <migraphx/filesystem.hpp>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/gpu/context.hpp>

#include <migraphx/env.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/gpu/compile_gen.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/module.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/reduce_dims.hpp>
#include <migraphx/stringutils.hpp>

#include "ck/host/device_gemm_multiple_d.hpp"

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace gpu {

using namespace migraphx::gpu::gen; // NOLINT

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_LOG_CK_GEMM);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_CK_TUNING);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_CK_TUNING_VALUE);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_CK_DEBUG);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TUNE_CK);

// NOLINTNEXTLINE
static const char* const ck_gemm_kernel = R"__migraphx__(
#include <args.hpp>
#include <migraphx/kernels/ck_gemm.hpp>
#include <migraphx/kernels/pointwise.hpp>
#include <migraphx/kernels/ops.hpp>
#include <${include}>

namespace migraphx {

${preamble}

extern "C" {

MIGRAPHX_GLOBAL void ${kernel}(${params})
{
    transform_args(make_tensors(), rotate_last())(${args})([](auto... xs) {
        ck_gemm<${solution}, ${blocks_per_batch}>(xs...);
    });
}

}

} // namespace migraphx

)__migraphx__";

// NOLINTNEXTLINE
static const char* const disable_warning_pragma = R"__migraphx__(
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
${content}
#pragma clang diagnostic pop
)__migraphx__";

template <class P>
static std::string ck_disable_warnings(P p)
{
    return interpolate_string(disable_warning_pragma,
                              {{"content", std::string{p.first, p.second}}});
}

static std::unordered_map<std::string, std::string> create_ck_header_strings()
{
    std::unordered_map<std::string, std::string> result;
    auto ck_headers = ck::host::GetHeaders();

    std::transform(
        ck_headers.begin(), ck_headers.end(), std::inserter(result, result.begin()), [&](auto&& p) {
            return std::make_pair(p.first, ck_disable_warnings(p.second));
        });
    return result;
}

static std::vector<src_file> create_ck_headers()
{
    static const auto& header_strings = create_ck_header_strings();
    std::vector<src_file> srcs;
    std::transform(
        header_strings.begin(), header_strings.end(), std::back_inserter(srcs), [&](auto&& p) {
            return src_file{p.first, p.second};
        });
    return srcs;
}

static const std::vector<src_file>& ck_headers()
{
    static const auto& headers = create_ck_headers();
    return headers;
}

static bool transposed_matrix(const shape& s) { return s.strides().back() != 1; }

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
    {
        std::cout << "*********** Warning: No CK tuning! for config:" << std::endl;
        std::cout << "  " << inputs[0] << std::endl;
        std::cout << "  " << inputs[1] << std::endl;
        std::cout << "  " << inputs[2] << std::endl;
    }
    auto it = std::find_if(
        tuning.begin(), tuning.end(), [&](const auto& p) { return p.first == inputs; });
    if(it == tuning.end())
    {
        std::cout << "*********** Warning: CK tuning missing for config!" << std::endl;
        std::cout << "  " << inputs[0] << std::endl;
        std::cout << "  " << inputs[1] << std::endl;
        std::cout << "  " << inputs[2] << std::endl;
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

    static ck::host::DataType get_type(const shape& s)
    {
        if(s.type() == shape::half_type)
            return ck::host::DataType::Half;
        else if(s.type() == shape::float_type)
            return ck::host::DataType::Float;
        else if(s.type() == shape::int8_type)
            return ck::host::DataType::Int8;
        else if(s.type() == shape::int32_type)
            return ck::host::DataType::Int32;
        MIGRAPHX_THROW("Unsupported ck type");
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

    static bool standard_batch(const shape& s)
    {
        if(s.lens().size() < 3)
            return true;
        std::vector<std::size_t> lens(s.lens().begin(), s.lens().end() - 2);
        std::vector<std::size_t> strides(s.strides().begin(), s.strides().end() - 2);
        auto base = *(s.lens().end() - 2) * *(s.lens().end() - 1);
        std::transform(strides.begin(), strides.end(), strides.begin(), [&](auto stride) {
            return stride / base;
        });
        return shape{s.type(), lens, strides}.standard();
    }

    bool can_fold_batch(const std::vector<shape>& inputs) const
    {
        const auto& b_shape = inputs[1];
        if(std::any_of(inputs.begin() + 2, inputs.end() - 1, [](auto input) {
               return not standard_batch(input);
           }))
            return false;
        const auto& b_strides = b_shape.strides();
        return std::all_of(
            b_strides.begin(), b_strides.end() - 2, [](auto stride) { return stride == 0; });
    }

    ck::host::device_gemm_multiple_d::Problem create_problem(const std::vector<shape>& inputs,
                                                             const value& v) const
    {
        const auto& a_shape = inputs[0];
        const auto& b_shape = inputs[1];
        const auto& c_shape = inputs.back();

        // cppcheck-suppress unreadVariable
        auto rank = a_shape.ndim();

        auto batch_count = get_batch_count(c_shape);
        auto m           = c_shape.lens()[rank - 2];
        m                = can_fold_batch(inputs) ? m * batch_count : m;
        auto n           = c_shape.lens().back();
        auto k           = a_shape.lens().back();

        const bool trans_a = transposed_matrix(a_shape);
        const bool trans_b = transposed_matrix(b_shape);
        const bool trans_e = transposed_matrix(c_shape);
        const auto a_type  = get_type(a_shape);
        const auto b_type  = get_type(b_shape);
        const auto e_type  = get_type(c_shape);
        std::vector<bool> ds_layout;
        std::transform(inputs.begin() + 2,
                       inputs.end() - 1,
                       std::back_inserter(ds_layout),
                       [](const auto& i) { return transposed_matrix(i); });
        std::vector<ck::host::DataType> ds_type;
        std::transform(inputs.begin() + 2,
                       inputs.end() - 1,
                       std::back_inserter(ds_type),
                       [](const auto& i) { return get_type(i); });

        std::string ck_passthrough = "ck_passthrough";
        std::string cde_op         = ck_passthrough;
        assert(inputs.size() < 4 or v.contains("post"));
        if(v.contains("post"))
        {
            cde_op = v.at("post").to<std::string>();
        }

        return ck::host::device_gemm_multiple_d::Problem{m,
                                                         n,
                                                         k,
                                                         trans_a,
                                                         trans_b,
                                                         trans_e,
                                                         ds_layout,
                                                         a_type,
                                                         b_type,
                                                         e_type,
                                                         ds_type,
                                                         ck_passthrough,
                                                         ck_passthrough,
                                                         cde_op};
    }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        const auto& a_shape = inputs[0];
        const auto& b_shape = inputs[1];
        const auto& c_shape = inputs.back();
        auto tuning_value   = v.get("tuning_value", 4);
        if(not v.contains("tuning_value"))
            tuning_value = get_tuning_for({a_shape, b_shape, c_shape});
        auto batch_count = get_batch_count(c_shape);
        auto problem     = create_problem(inputs, v);

        const auto include_header   = problem.GetIncludeHeader();
        const auto solutions        = problem.GetSolutions(ctx.get_current_device().get_gfx_name());
        const auto& solution        = solutions.at(tuning_value);
        const auto template_str     = solution.template_str;
        const auto blocks_per_batch = solution.grid_size;
        const auto block_size       = solution.block_size;

        hip_compile_options options;
        options.additional_src_files = ck_headers();
        auto grid_size = can_fold_batch(inputs) ? blocks_per_batch : batch_count * blocks_per_batch;
        options.set_launch_params(v, grid_size * block_size, block_size);
        options.inputs         = inputs;
        options.output         = c_shape;
        options.kernel_name    = v.get("kernel", "ck_gemm_kernel");
        options.virtual_inputs = inputs;
        if(can_fold_batch(inputs))
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
                                      {{"solution", template_str},
                                       {"include", include_header},
                                       {"params", enum_params(inputs.size(), "void * private_p")},
                                       {"args", enum_params(inputs.size(), "private_p")},
                                       {"blocks_per_batch", to_string(blocks_per_batch)},
                                       {"preamble", v.get("preamble", std::string{})},
                                       {"kernel", options.kernel_name}});

        return compile_hip_code_object(src, options);
    }

    value create_settings(instruction_ref ins, const operation& op) const
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
        return v;
    }

    compiler_replace
    compile(context& ctx, instruction_ref ins, const operation& op, const value& solution) const
    {
        auto shapes = to_shapes(ins->inputs());
        auto v      = create_settings(ins, op);
        if(not solution.is_null())
            v["tuning_value"] = solution;
        return {compile_op(ctx, shapes, v),
                [=](module& m, instruction_ref ins2, const operation& code_object) {
                    if(enabled(MIGRAPHX_LOG_CK_GEMM{}))
                    {
                        std::vector<shape> gemm_shapes{
                            shapes[0], shapes[1], shapes.back().with_type(shapes[0].type())};
                        std::cout << "gpu::ck_gemm: " << to_json_string(to_value(gemm_shapes))
                                  << std::endl;
                    }
                    m.replace_instruction(ins2, code_object, ins2->inputs());
                }};
    }

    optional<tuning_config>
    get_tuning_config(context& ctx, instruction_ref ins, const operation& op, bool exhaustive) const
    {
        if(not exhaustive and not enabled(MIGRAPHX_TUNE_CK{}))
            return nullopt;
        tuning_config tc;
        auto shapes    = to_shapes(ins->inputs());
        auto problem   = create_problem(shapes, create_settings(ins, op));
        auto solutions = problem.GetSolutions(ctx.get_current_device().get_gfx_name());
        tc.solutions.resize(solutions.size());
        std::iota(tc.solutions.begin(), tc.solutions.end(), 0);
        std::vector<shape> gemm_shapes{shapes[0], shapes[1], shapes.back()};
        tc.problem = to_value(gemm_shapes);
        return tc;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
