#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_MIOPEN_MACHINE_MODEL_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_MIOPEN_MACHINE_MODEL_HPP
#include <string>
#include <unordered_map>
#include <migraphx/pass_config.hpp>
#include <migraphx/operation.hpp>

namespace migraphx {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_NULL_STREAM)

struct op_info
{
    op_info()
    {
        // First in pair denotes weight.  Second in pair tells
        // that the instruction is run ONLY on CPU.
        weight_map["convolution"]         = std::make_pair(4, 0);
        weight_map["pooling"]             = std::make_pair(2, 0);
        weight_map["gemm"]                = std::make_pair(2, 0);
        weight_map["broadcast"]           = std::make_pair(1, 1);
        weight_map["multibroadcast"]      = std::make_pair(1, 1);
        weight_map["contiguous"]          = std::make_pair(1, 1);
        weight_map["transpose"]           = std::make_pair(1, 1);
        weight_map["load"]                = std::make_pair(1, 1);
        weight_map["@param"]              = std::make_pair(1, 1);
        weight_map["@literal"]            = std::make_pair(1, 1);
        weight_map["hip::load_literal"]   = std::make_pair(1, 1);
        weight_map["hip::allocate"]       = std::make_pair(0, 1);
        weight_map["@outline"]            = std::make_pair(0, 1);
        weight_map["slice"]               = std::make_pair(1, 1);
        weight_map["squeeze"]             = std::make_pair(1, 1);
        weight_map["unsqueeze"]           = std::make_pair(1, 1);
        weight_map["gpu::convolution"]    = std::make_pair(4, 0);
        weight_map["gpu::conv_bias_relu"] = std::make_pair(4, 0);
        weight_map["gpu::pooling"]        = std::make_pair(2, 0);
        weight_map["gpu::gemm"]           = std::make_pair(2, 0);
        weight_map["gpu::concat"]         = std::make_pair(1, 0);
        weight_map["hip::add_relu"]       = std::make_pair(2, 0);
    }

    std::pair<int, int> operator()(const operation& op)
    {
        if(weight_map.find(op.name()) != weight_map.end())
        {
            return weight_map[op.name()];
        }
        else
        {
            return std::make_pair(1, is_context_free(op) ? 1 : 0);
        }
    }
    std::unordered_map<std::string, std::pair<int, int>> weight_map;
};

struct stream_info
{
    int num_of_streams()
    {
        if(!enabled(MIGRAPHX_DISABLE_NULL_STREAM{}))
            return 0;
        else
            return 4;
    }
};
} // namespace gpu
} // namespace migraphx

#endif
