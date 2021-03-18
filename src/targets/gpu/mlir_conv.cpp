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

#ifdef    MIGRAPHX_MLIR_MIOPEN_SUPPORT
#include <Miir.h>
#endif // MIGRAPHX_MLIR_MIOPEN_SUPPORT

#include <stdio.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct mlir_apply
{
    module* prog                 = nullptr;
    const mlir_conv* pass        = nullptr;

    const char *mlir_kernel_name = "migraphx_conv2d";

    std::unordered_map<uint64_t, instruction_ref> literal_map{};

    struct execution_spec {
      migraphx::value::binary binary;
      size_t global_size;
      size_t local_size;
    };
    std::unordered_map<std::string, execution_spec*> binary_map{};

    context& get_context() const
    {
        assert(pass != nullptr);
        assert(pass->ctx != nullptr);
        return *pass->ctx;
    }

    void init()
    {
        assert(prog != nullptr);
        assert(pass != nullptr);
    }

    execution_spec *make_mlir_binary(instruction_ref op_r)
    {
        execution_spec *result = nullptr;

#ifdef    MIGRAPHX_MLIR_MIOPEN_SUPPORT
        auto conv  = any_cast<op::convolution>(op_r->get_operator());
        auto inp_t = op_r->inputs().at(0)->get_shape();
        auto flt_t = op_r->inputs().at(1)->get_shape();
        auto out_t = op_r->get_shape();

        auto check_type = [](const shape &s) {
          switch (s.type()) {
          case shape::float_type: return true;
          default: break;
          }
          return false;
        };

        if (!check_type(out_t) || !check_type(inp_t) || !check_type(flt_t))
          return nullptr;

        std::string mlir_options =
          " --kernel_name " + std::string(mlir_kernel_name);

        // platform spec
        auto &device = get_context().get_current_device();
        mlir_options += 
          " --arch " + device.get_device_name() +
          " --num_cu " + std::to_string(device.get_cu_count()); // ???

        // Conv spec
        mlir_options += 
          " --operation conv2d"
          " --batchsize " + std::to_string(conv.group) +
          " --padding_h " + std::to_string(conv.padding[0]) +
          " --padding_w " + std::to_string(conv.padding[1]) +
          " --conv_stride_h " + std::to_string(conv.stride[0]) +
          " --conv_stride_w " + std::to_string(conv.stride[1]) +
          " --dilation_h " + std::to_string(conv.dilation[0]) +
          " --dilation_w " + std::to_string(conv.dilation[1]);

        // Input spec
        mlir_options += 
          " --in_layout NCHW"
          " --in_type "       "fp32"
          " --in_channels " + std::to_string(inp_t.lens()[1]) +
          " --in_h "        + std::to_string(inp_t.lens()[2]) +
          " --in_w "        + std::to_string(inp_t.lens()[3]);
        
        // Filter spec
        mlir_options += 
          " --fil_layout NCHW"
          " --fil_type "      "fp32"
          " --fil_h "       + std::to_string(flt_t.lens()[2]) +
          " --fil_w "       + std::to_string(flt_t.lens()[3]);
        
        // Output spec
        mlir_options += 
          " --out_layout NCHW"
          " --out_type "       "fp32"
          " --out_channels " + std::to_string(out_t.lens()[1]) +
          " --out_h "        + std::to_string(out_t.lens()[2]) +
          " --out_w "        + std::to_string(out_t.lens()[3]);

        auto bin_i = binary_map.find(mlir_options);
        if (bin_i == binary_map.end()) {
          size_t bin_size = 0;
          char *bin_buffer = nullptr;

          auto mlir_handle = miirCreateHandle(mlir_options.c_str());

          if (miirLowerBin(mlir_handle) == MIIR_SUCCESS &&
              miirBufferGet(mlir_handle, nullptr, &bin_size) == MIIR_SUCCESS) {
            bin_buffer = new char[bin_size];
            if (miirBufferGet(mlir_handle, bin_buffer, &bin_size) == MIIR_SUCCESS) {
              size_t global_size, block_size;
              if (miirGetExecutionDims(mlir_handle, &global_size, &block_size) == MIIR_SUCCESS) {
                printf("MLIR - %s\n", mlir_options.c_str());
                result = new execution_spec{{bin_buffer, bin_size}, global_size, block_size};
              }
            }
          }
          miirDestroyHandle(mlir_handle);
        
          binary_map[mlir_options] = result;
        } else {
          result = bin_i->second;
        }
#endif // MIGRAPHX_MLIR_MIOPEN_SUPPORT
        return result;
    }

    instruction_ref get_literal(uint64_t value)
    {
      auto fi = literal_map.find(value);
      if (fi != literal_map.end())
        return fi->second;
      auto lit = prog->add_literal(value);
      literal_map.emplace(value, lit);
      return lit;
    }
  
    operation make_code_object_op(instruction_ref op_r, execution_spec *spec)
    {
      // each pointer is expanded out to a MemRefDescriptor
      auto inp_t = op_r->inputs().at(0)->get_shape();
      auto flt_t = op_r->inputs().at(1)->get_shape();
      auto out_t = op_r->get_shape();

      auto i64 = shape(shape::uint64_type);

      std::vector<shape> expected_inputs = {
        flt_t, flt_t, i64, i64, i64, i64, i64, i64, i64, i64, i64,
        inp_t, inp_t, i64, i64, i64, i64, i64, i64, i64, i64, i64,
        out_t, out_t, i64, i64, i64, i64, i64, i64, i64, i64, i64,
        out_t
      };

      return migraphx::make_op("gpu::code_object",
                             {{"code_object", spec->binary},
                              {"symbol_name", mlir_kernel_name},
                              {"global", spec->global_size},
                              {"local", spec->local_size},
                              {"expected_inputs", migraphx::to_value(expected_inputs)},
                              {"output", migraphx::to_value(out_t)},
                                  });
    }

    void add_memref_descriptor(std::vector<instruction_ref> & refs, instruction_ref inst)
    {
      const size_t offset = 0;
      auto inst_t = inst->get_shape();
      refs.push_back(inst);
      refs.push_back(inst);
      refs.push_back(get_literal(offset)); // offset

      auto &lens = inst_t.lens();
      for (int i = 0; i < lens.size(); ++i) {
        refs.push_back(get_literal(lens[i]));
      }
      auto &strides = inst_t.strides();
      for (int i = 0; i < strides.size(); ++i) {
        refs.push_back(get_literal(strides[i]));
      }
    }
  
    instruction_ref insert_allocation(instruction_ref ins, const shape& s)
    {
        return prog->insert_instruction(ins, hip_allocate{s});
    }

    void replace_conv_op(instruction_ref ins)
    {
      auto conv_bin = make_mlir_binary(ins);
      if (conv_bin != nullptr) {
        auto conv = make_code_object_op(ins, conv_bin);

        auto inp = ins->inputs().at(0);
        auto flt = ins->inputs().at(1);
        auto out = insert_allocation(ins, ins->get_shape());
              
        std::vector<instruction_ref> refs;
        refs.reserve(3*11 + 1);
        add_memref_descriptor(refs, flt);
        add_memref_descriptor(refs, inp);
        add_memref_descriptor(refs, out);
        refs.push_back(out);
              
        prog->replace_instruction(ins, conv, refs);
      }
    }

    void apply()
    {
        init();
        for(auto it = prog->begin(); it != prog->end(); it++) {
            if (it->name() == "convolution") {
                replace_conv_op(it);
            }
        }
    }

};

void mlir_conv::apply(module& p) const
{
  if (enable_mlir) {
    mlir_apply{&p, this}.apply();
  }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
