#include <migraphx/gpu/mlir_conv.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

#include <migraphx/op/convolution.hpp>

#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/convolution.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/program.hpp>

#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/program.hpp>
#include <migraphx/gpu/kernel.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/gpu/compile_hip.hpp>

#include <utility>
#include <functional>
#include <algorithm>

#ifdef MIGRAPHX_MLIR_MIOPEN_SUPPORT
#include <Miir.h>
#endif // MIGRAPHX_MLIR_MIOPEN_SUPPORT

#include <cstdio>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct mlir_apply
{
    module* mod           = nullptr;
    const mlir_conv* pass = nullptr;

    const char* mlir_kernel_name = "migraphx_conv2d";

    std::unordered_map<uint64_t, instruction_ref> literal_map{};

    struct execution_spec
    {
        migraphx::value::binary binary;
        size_t global_size;
        size_t local_size;
        execution_spec(migraphx::value::binary&& binary_m, size_t global_s, size_t local_s)
            : binary(std::move(binary_m)), global_size(global_s), local_size(local_s)
        {
        }
    };

    std::unordered_map<std::string, std::shared_ptr<execution_spec>> binary_map{};

    context& get_context() const
    {
        assert(pass != nullptr);
        assert(pass->ctx != nullptr);
        return *pass->ctx;
    }

    void init() const
    {
        assert(mod != nullptr);
        assert(pass != nullptr);
    }

    std::shared_ptr<execution_spec> make_mlir_binary(instruction_ref op_r)
    {
        std::shared_ptr<execution_spec> result;

#ifdef MIGRAPHX_MLIR_MIOPEN_SUPPORT
        auto conv  = any_cast<op::convolution>(op_r->get_operator());
        auto inp_t = op_r->inputs().at(0)->get_shape();
        auto flt_t = op_r->inputs().at(1)->get_shape();
        auto out_t = op_r->get_shape();

        auto get_type_str = [](const shape& s) -> const char* {
            switch(s.type())
            {
            case shape::float_type: return "f32";
            case shape::half_type: return "f16";
            case shape::bool_type:
            case shape::double_type:
            case shape::uint8_type:
            case shape::int8_type:
            case shape::uint16_type:
            case shape::int16_type:
            case shape::int32_type:
            case shape::int64_type:
            case shape::uint32_type:
            case shape::uint64_type:
            case shape::tuple_type: break;
            }
            return nullptr;
        };

        const auto* inp_t_s = get_type_str(inp_t);
        const auto* flt_t_s = get_type_str(flt_t);
        const auto* out_t_s = get_type_str(out_t);

        if(out_t_s == nullptr || inp_t_s == nullptr || flt_t_s == nullptr)
            return result;

        std::string mlir_options = "--kernel_name " + std::string(mlir_kernel_name);

        // platform spec
        auto& device = get_context().get_current_device();
        char dev_name[64];
        sprintf(dev_name, "gfx%lu%02lu", device.get_device_major(), device.get_device_minor());
        mlir_options += " --arch " + std::string(dev_name) + " --num_cu " +
                        std::to_string(device.get_cu_count()); // ???

        // Conv spec
        mlir_options +=
            " --operation "
            "conv2d"
            " --batchsize " +
            std::to_string(conv.group) + " --groupsize " + std::to_string(1) + " --padding_h " +
            std::to_string(conv.padding[0]) + " --padding_w " + std::to_string(conv.padding[1]) +
            " --conv_stride_h " + std::to_string(conv.stride[0]) + " --conv_stride_w " +
            std::to_string(conv.stride[1]) + " --dilation_h " + std::to_string(conv.dilation[0]) +
            " --dilation_w " + std::to_string(conv.dilation[1]);

        // Input spec
        mlir_options += " --in_layout "
                        "NCHWG"
                        " --in_type " +
                        std::string(inp_t_s) + " --in_channels " + std::to_string(inp_t.lens()[1]) +
                        " --in_h " + std::to_string(inp_t.lens()[2]) + " --in_w " +
                        std::to_string(inp_t.lens()[3]);

        // Filter spec
        mlir_options += " --fil_layout "
                        "NCHWG"
                        " --fil_type " +
                        std::string(flt_t_s) + " --fil_h " + std::to_string(flt_t.lens()[2]) +
                        " --fil_w " + std::to_string(flt_t.lens()[3]);

        // Output spec
        mlir_options += " --out_layout "
                        "NCHWG"
                        " --out_type " +
                        std::string(out_t_s) + " --out_channels " +
                        std::to_string(out_t.lens()[1]) + " --out_h " +
                        std::to_string(out_t.lens()[2]) + " --out_w " +
                        std::to_string(out_t.lens()[3]);

        auto bin_i = binary_map.find(mlir_options);
        if(bin_i == binary_map.end())
        {
            size_t bin_size = 0;

            using mlir_handle = MIGRAPHX_MANAGE_PTR(MiirHandle, miirDestroyHandle);
            auto handle       = mlir_handle(miirCreateHandle(mlir_options.c_str()));

            if(miirLowerBin(handle.get()) == MIIR_SUCCESS &&
               miirBufferGet(handle.get(), nullptr, &bin_size) == MIIR_SUCCESS)
            {
                migraphx::value::binary bin(bin_size);
                if(miirBufferGet(handle.get(), reinterpret_cast<char*>(bin.data()), &bin_size) ==
                   MIIR_SUCCESS)
                {
                    size_t global_size;
                    size_t block_size;
                    if(miirGetExecutionDims(handle.get(), &global_size, &block_size) ==
                       MIIR_SUCCESS)
                    {
                        result = std::make_shared<execution_spec>(
                            std::move(bin), global_size, block_size);
                    }
                }
            }

            binary_map[mlir_options] = result;
        }
        else
        {
            result = bin_i->second;
        }
#else  // MIGRAPHX_MLIR_MIOPEN_SUPPORT
        (void)op_r;
#endif // MIGRAPHX_MLIR_MIOPEN_SUPPORT
        return result;
    }

    instruction_ref get_literal(uint64_t value)
    {
        auto fi = literal_map.find(value);
        if(fi != literal_map.end())
            return fi->second;
        auto lit = mod->add_literal(value);
        literal_map.emplace(value, lit);
        return lit;
    }

    operation make_code_object_op(instruction_ref op_r, const std::shared_ptr<execution_spec>& spec)
    {
        // each pointer is expanded out to a MemRefDescriptor
        auto inp_t = op_r->inputs().at(0)->get_shape();
        auto flt_t = op_r->inputs().at(1)->get_shape();
        auto out_t = op_r->get_shape();

        auto i64 = shape(shape::uint64_type);

        std::vector<shape> expected_inputs = {
            flt_t, flt_t, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64,  i64,   inp_t,
            inp_t, i64,   i64, i64, i64, i64, i64, i64, i64, i64, i64, i64,  out_t, out_t,
            i64,   i64,   i64, i64, i64, i64, i64, i64, i64, i64, i64, out_t};

        return migraphx::make_op("gpu::code_object",
                                 {
                                     {"code_object", spec->binary},
                                     {"symbol_name", mlir_kernel_name},
                                     {"global", spec->global_size},
                                     {"local", spec->local_size},
                                     {"expected_inputs", migraphx::to_value(expected_inputs)},
                                     {"output", migraphx::to_value(out_t)},
                                 });
    }

    void add_memref_descriptor(std::vector<instruction_ref>& refs, instruction_ref inst)
    {
        const size_t offset = 0;
        auto inst_t         = inst->get_shape();
        refs.push_back(inst);
        refs.push_back(inst);
        refs.push_back(get_literal(offset)); // offset

        // dim sizes
        std::transform(inst_t.lens().begin(),
                       inst_t.lens().end(),
                       std::back_inserter(refs),
                       [&](const auto& lval) { return get_literal(lval); });
        refs.push_back(get_literal(1)); // G

        // dim strides
        std::transform(inst_t.strides().begin(),
                       inst_t.strides().end(),
                       std::back_inserter(refs),
                       [&](const auto& lval) { return get_literal(lval); });
        refs.push_back(get_literal(1)); // G
    }

    instruction_ref insert_allocation(instruction_ref ins, const shape& s) const
    {
        return mod->insert_instruction(ins, hip_allocate{s});
    }

    void replace_conv_op(instruction_ref ins)
    {
        auto conv_bin = make_mlir_binary(ins);
        if(conv_bin)
        {
            auto conv = make_code_object_op(ins, conv_bin);

            auto inp = ins->inputs().at(0);
            auto flt = ins->inputs().at(1);
            auto out = insert_allocation(ins, ins->get_shape());

            std::vector<instruction_ref> refs;
            refs.reserve(3 * 13 + 1);
            add_memref_descriptor(refs, flt);
            add_memref_descriptor(refs, inp);
            add_memref_descriptor(refs, out);
            refs.push_back(out);

            mod->replace_instruction(ins, conv, refs);
        }
    }

    void apply()
    {
        init();
        for(auto it : iterator_for(*mod))
        {
            if(it->name() == "convolution")
            {
                replace_conv_op(it);
            }
        }
    }
};

void mlir_conv::apply(module& m) const { mlir_apply{&m, this}.apply(); }

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
