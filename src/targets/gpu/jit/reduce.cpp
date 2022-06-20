#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/compile_gen.hpp>

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

using namespace migraphx::gpu::gen; // NOLINT

static const char* const simple_reduce_kernel = R"__migraphx__(
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/reduce.hpp>
#include <migraphx/kernels/vectorize.hpp>
#include <args.hpp>

namespace migraphx {

${preamble}

extern "C" {
__global__ void reduce_kernel(void* input_p, void* output_p) 
{
    
    transform_args(make_tensors(), ${transformers})(input_p, output_p)([](auto input, auto output) {

        simple_reduce<reduce::${algo}>(${reduction}, ${init}, input, output, ${read}, ${write});
    });
}
    
}

} // namespace migraphx

)__migraphx__";

static std::size_t get_reduce_elements(const std::vector<shape>& inputs)
{
    return inputs.front().elements() / inputs.back().elements();
}
static std::size_t get_reduce_elements(const std::vector<instruction_ref>& inputs)
{
    return get_reduce_elements(to_shapes(inputs));
}

static std::vector<std::size_t> get_reduce_lens(const std::vector<std::size_t>& input_lens,
                                                const std::vector<std::size_t>& output_lens)
{
    std::vector<std::size_t> reduce_lens;
    std::transform(output_lens.begin(),
                   output_lens.end(),
                   input_lens.begin(),
                   std::back_inserter(reduce_lens),
                   [](auto x, auto y) -> std::size_t {
                       if(x == y)
                           return 1;
                       else
                           return y;
                   });
    return reduce_lens;
}

static std::string get_reduce_algo(const std::vector<shape>& inputs)
{
    auto rlens      = get_reduce_lens(inputs.front().lens(), inputs.back().lens());
    const auto init = std::numeric_limits<std::size_t>::max();
    // The minimum stride
    auto min_stride = std::inner_product(
        rlens.begin(),
        rlens.end(),
        inputs.front().strides().begin(),
        init,
        [](auto x, auto y) { return std::min(x, y); },
        [](auto len, auto stride) { return len == 1 ? init : stride; });
    if(min_stride > 2)
        return "lane";
    return "block";
}

struct reduce_compiler : compiler<reduce_compiler>
{
    std::vector<std::string> names() const
    {
        return {"reduce", "reduce_sum", "reduce_mean", "reduce_max", "reduce_min", "reduce_prod"};
    }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        options.inputs         = inputs;
        options.output         = inputs.back();
        options.virtual_inputs = reduce_dims(inputs);
        auto faxis             = find_fast_axis({options.virtual_inputs.front()});
        vectorize vec{};
        // Vectorize if the axis is a reduction axis
        if(options.virtual_inputs.back().lens()[faxis] == 1)
        {
            vec = vectorize::elements(faxis, options.virtual_inputs);
        }
        auto relements = get_reduce_elements(options.virtual_inputs) / vec.size;
        auto nelements = options.virtual_inputs.back().elements();
        auto algo      = v.get("algo", get_reduce_algo(options.virtual_inputs));
        if(algo == "block")
        {
            auto block_size = compute_block_size(relements, 256);
            options.set_launch_params(
                v, compute_global_for(ctx, nelements * block_size, 256), block_size);
        }
        else if(algo == "lane")
        {
            options.set_launch_params(v, compute_global_for(ctx, nelements, 256));
        }
        else
        {
            MIGRAPHX_THROW("Unknown reduce algo: " + algo);
        }
        options.kernel_name  = "reduce_kernel";
        std::string identity = "[](auto x) { return x; }";
        auto src             = interpolate_string(simple_reduce_kernel,
                                      {{"reduction", v.at("reduction").to<std::string>()},
                                       {"init", v.get("init", std::string{"0"})},
                                       {"read", v.get("read", identity)},
                                       {"write", v.get("write", identity)},
                                       {"algo", algo},
                                       {"transformers", make_transformer_args(vec)},
                                       {"preamble", v.get("preamble", std::string{})}});
        options.params += "-Wno-float-equal";
        return compile_hip_code_object(src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        value v              = value::object{};
        auto reduce_elements = get_reduce_elements(ins->inputs());
        if(op.name() == "reduce_sum")
        {
            v["reduction"] = "op::sum{}";
        }
        else if(op.name() == "reduce_mean")
        {
            v["reduction"] = "op::sum{}";
            v["write"]     = "op::mean{" + std::to_string(reduce_elements) + "}";
        }
        else if(op.name() == "reduce_max")
        {
            v["reduction"] = "op::max{}";
            v["init"]      = "lowest{}";
        }
        else if(op.name() == "reduce_min")
        {
            v["reduction"] = "op::min{}";
            v["init"]      = "highest{}";
        }
        else if(op.name() == "reduce_prod")
        {
            v["reduction"] = "op::product{}";
            v["init"]      = "1";
        }
        else
        {
            MIGRAPHX_THROW("Unsupported reduce");
        }
        return replace(compile_op(ctx, to_shapes(ins->inputs()), v));
    }
};
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
