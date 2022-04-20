#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>

#include <migraphx/cpp_generator.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/reduce_dims.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

static const char* const pointwise_kernel = R"__migraphx__(
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/pointwise.hpp>
#include <args.hpp>

namespace migraphx {

${preamble}

extern "C" {
__global__ void kernel(${params}) 
{
    auto idx = make_index();
    pointwise(idx, auto_preload(idx), vectorize<${vec_size}, ${vec_axis}>())(${lambda}, ${args});
}
    
}

} // namespace migraphx

)__migraphx__";

struct pointwise_compiler : compiler<pointwise_compiler>
{
    std::vector<std::string> names() const { return {"pointwise"}; }

    static std::size_t oversubscribe(const std::vector<shape>& inputs)
    {
        if(std::any_of(inputs.begin(), inputs.end(), [](const auto& s) { return s.broadcasted(); }))
            return 1;
        else
            return 256;
    }
    static std::vector<std::size_t> vector_sizes(const std::vector<shape>& inputs)
    {
        // If all inputs is half then only use half2
        if(std::all_of(inputs.begin(), inputs.end(), [](const auto& s) { return s.type() == shape::half_type; }))
            return {2};
        return {4, 2};

    }
    static auto vectorize_elements(const std::vector<shape>& inputs)
    {
        std::size_t n    = 1;
        std::size_t rank = inputs.front().lens().size();
        std::size_t axis = rank;
        for(const auto& input:inputs)
        {
            if (input.broadcasted())
                continue;
            axis = std::distance(input.strides().begin(), std::find(input.strides().begin(), input.strides().end(), 1));
            break;
        }
        if (axis >= rank)
            return std::make_pair(n, axis);
        auto sizes = vector_sizes(inputs);
        std::vector<std::size_t> max_vec_size;
        std::transform(inputs.begin(), inputs.end(), std::back_inserter(max_vec_size), [&](const auto& input) -> std::size_t {
            auto stride = input.strides()[axis];
            auto len = input.lens()[axis];
            if (stride != 0 and stride != 1)
                return 1;
            auto it = std::find_if(sizes.begin(), sizes.end(), [&](auto i) {
                return (len % i) == 0;
            });
            if (it != sizes.end())
                return *it;
            return 1;
        });
        n = *std::min_element(max_vec_size.begin(), max_vec_size.end());
        return std::make_pair(n, axis);
    }
    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        options.inputs            = inputs;
        options.output            = inputs.back();
        options.virtual_inputs    = reduce_dims(inputs);
        options.params            = "-Wno-float-equal";
        auto [vec_size, vec_axis] = vectorize_elements(options.virtual_inputs);
        options.set_launch_params(
            v,
            compute_global_for(ctx, options.output.elements() / vec_size, oversubscribe(inputs)));
        auto src = interpolate_string(pointwise_kernel,
                                      {{"params", enum_params(inputs.size(), "void * private_p")},
                                       {"args", enum_params(inputs.size(), "private_p")},
                                       {"lambda", v.at("lambda").to<std::string>()},
                                       {"vec_size", std::to_string(vec_size)},
                                       {"vec_axis", std::to_string(vec_axis)},
                                       {"preamble", v.get("preamble", std::string{})}});
        return compile_hip_code_object(src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation&) const
    {
        assert(not ins->module_inputs().empty());
        auto* pm = ins->module_inputs().front();
        run_passes(*pm, {eliminate_common_subexpression{}, dead_code_elimination{}});
        cpp_generator g;
        g.fmap([](const std::string& fname) { return "migraphx::" + fname; });
        g.add_point_op("where", "${function:where}(${0}, ${1}, ${2})");
        g.add_point_op("prelu", "${function:where}(${0} < 0, ${0} * ${1}, ${0})");
        g.add_point_op("sign",
                       "${function:where}(${0} > 0, 1, ${function:where}(${0} < 0, -1, 0))");
        g.add_point_op("equal", "migraphx::abs(${0} == ${1})");
        g.add_point_op("less", "migraphx::abs(${0} < ${1})");
        g.add_point_op("greater", "migraphx::abs(${0} > ${1})");
        g.add_point_op("not", "migraphx::abs(not ${0})");
        // Add explict conversions
        g.fresult(
            [](const shape& s) { return "migraphx::convert<" + shape::cpp_type(s.type()) + ">"; });
        auto name = g.create_function(
            g.generate_module(*pm).set_attributes({"__device__"}).set_generic_types(*pm));
        std::string lambda = "MIGRAPHX_LIFT(" + name + ")";
        return replace(
            compile_op(ctx, to_shapes(ins->inputs()), {{"lambda", lambda}, {"preamble", g.str()}}));
    }
};
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
