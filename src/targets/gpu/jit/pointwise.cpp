#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>

#include <migraphx/cpp_generator.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/reduce_dims.hpp>
#include <migraphx/permutation.hpp>
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
__global__ void ${kernel}(${params}) 
{
    auto idx = make_index();
    pointwise(idx, auto_preload<${preloads}>(idx), vectorize<${vec_size}, ${axis}>())(${lambda}, ${args});
}
    
}

} // namespace migraphx

)__migraphx__";

static std::vector<std::string> get_op_names(const module& m)
{
    std::vector<std::string> result;
    for(auto& ins : m)
    {
        if(starts_with(ins.name(), "@"))
            continue;
        result.push_back(ins.name());
    }
    return result;
}

struct pointwise_compiler : compiler<pointwise_compiler>
{
    std::vector<std::string> names() const { return {"pointwise", "contiguous"}; }

    static std::size_t oversubscribe_if(bool b)
    {
        if(b)
            return 256;
        else
            return 1;
    }
    static std::size_t find_fast_axis(const std::vector<shape>& inputs)
    {
        auto permutation = find_permutation(inputs);
        auto it          = std::max_element(permutation.begin(), permutation.end());
        return it - permutation.begin();
    }
    static std::vector<bool> preload(std::size_t axis, const std::vector<shape>& inputs)
    {
        const std::size_t max_lds_bytes = 4096;
        std::vector<bool> result;
        std::transform(inputs.begin(),
                       inputs.end(),
                       std::back_inserter(result),
                       [&](const shape& input) { return input.strides()[axis] == 0; });
        auto bytes = std::inner_product(inputs.begin(),
                                        inputs.end(),
                                        result.begin(),
                                        std::size_t{0},
                                        std::plus<>{},
                                        [](const shape& s, bool b) -> std::size_t {
                                            if(b)
                                                return s.bytes();
                                            return 0;
                                        });
        if(bytes < max_lds_bytes)
            return result;
        // TODO: Try to partially preload items
        std::fill(result.begin(), result.end(), false);
        return result;
    }
    static std::string preload_str(const std::vector<bool>& bs)
    {
        std::vector<std::string> bool_strs;
        std::transform(bs.begin(), std::prev(bs.end()), std::back_inserter(bool_strs), [](bool b) {
            if(b)
                return "true";
            return "false";
        });
        return "false, " + join_strings(bool_strs, ", ");
    }
    static std::vector<std::size_t> vector_sizes(const std::vector<shape>& inputs)
    {
        // If all inputs is half then only use half2
        if(std::all_of(inputs.begin(), inputs.end(), [](const auto& s) {
               return s.type() == shape::half_type;
           }))
            return {2};
        return {4, 2};
    }
    static auto vectorize_elements(std::size_t axis, const std::vector<shape>& inputs)
    {
        auto sizes = vector_sizes(inputs);
        std::vector<std::size_t> max_vec_size;
        std::transform(inputs.begin(),
                       inputs.end(),
                       std::back_inserter(max_vec_size),
                       [&](const auto& input) -> std::size_t {
                           auto stride = input.strides()[axis];
                           auto len    = input.lens()[axis];
                           if(stride != 0 and stride != 1)
                               return 1;
                           auto it = std::find_if(
                               sizes.begin(), sizes.end(), [&](auto i) { return (len % i) == 0; });
                           if(it != sizes.end())
                               return *it;
                           return 1;
                       });
        return *std::min_element(max_vec_size.begin(), max_vec_size.end());
    }
    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        options.inputs         = inputs;
        options.output         = inputs.back();
        options.virtual_inputs = reduce_dims(inputs);
        options.params         = "-Wno-float-equal";
        auto axis              = find_fast_axis(options.virtual_inputs);
        auto vec_size          = vectorize_elements(axis, options.virtual_inputs);
        auto preloads          = preload(axis, options.virtual_inputs);
        auto is_preloading =
            std::accumulate(preloads.begin(), preloads.end(), false, std::logical_or<>{});
        options.kernel_name = v.get("kernel", "kernel");
        options.set_launch_params(v,
                                  compute_global_for(ctx,
                                                     options.output.elements() / vec_size,
                                                     oversubscribe_if(not is_preloading)));
        auto src = interpolate_string(pointwise_kernel,
                                      {{"kernel", options.kernel_name},
                                       {"params", enum_params(inputs.size(), "void * private_p")},
                                       {"args", enum_params(inputs.size(), "private_p")},
                                       {"lambda", v.at("lambda").to<std::string>()},
                                       {"vec_size", std::to_string(vec_size)},
                                       {"axis", std::to_string(axis)},
                                       {"preloads", preload_str(preloads)},
                                       {"preamble", v.get("preamble", std::string{})}});
        return compile_hip_code_object(src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        if(op.name() == "contiguous")
        {
            return replace(compile_op(
                ctx,
                to_shapes(ins->inputs()),
                {{"lambda", "[](auto x) { return x; }"}, {"kernel", "contiguous_kernel"}}));
        }
        else
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
            g.fresult([](const shape& s) {
                return "migraphx::convert<" + shape::cpp_type(s.type()) + ">";
            });
            auto name = g.create_function(
                g.generate_module(*pm).set_attributes({"__device__"}).set_generic_types(*pm));
            std::string lambda = "MIGRAPHX_LIFT(" + name + ")";
            auto op_names      = get_op_names(*pm);
            op_names.push_back("kernel");
            auto op_name_string = join_strings(op_names, "_");
            return replace(compile_op(
                ctx,
                to_shapes(ins->inputs()),
                {{"lambda", lambda}, {"preamble", g.str()}, {"kernel", op_name_string}}));
        }
    }
};
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
