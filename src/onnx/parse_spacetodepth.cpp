#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_spacetodepth : op_parser<parse_spacetodepth>
{
    std::vector<op_desc> operators() const { return {{"SpaceToDepth"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        auto s = args[0]->get_shape();
        // blocksize attribute of SpaceToDepth
        int blocksize = 1; // if blockSize of 1 then, this is a no-op
        if(contains(info.attributes, "blocksize"))
        {
            blocksize = info.attributes.at("blocksize").i();
        }
        if(blocksize < 1)
        {
            // blockSize less than 1 would rather result in DepthToSpace instead of SpaceToDepth
            MIGRAPHX_THROW("SpaceToDepth: blocksize is less than 1");
        }
        // calculate dimensions
        auto res_lens = s.lens(); // {N, C, H, W}
        if(((res_lens[2] % blocksize) == 0) and ((res_lens[3] % blocksize) == 0))
        {
            // Co = C * (blocksize ^ 2)
            res_lens[1] = res_lens[1] * blocksize * blocksize;
            // Ho = (H / blocksize)
            res_lens[2] = res_lens[2] / blocksize;
            // Wo = (W / blocksize)
            res_lens[3] = res_lens[3] / blocksize;
        } // res_shape = (N, Co, Ho, Wo)
        else
            MIGRAPHX_THROW("SpaceToDepth: div by blocksize quotient not int ");

        auto trans_lens = s.lens(); // {N, C, H, W}
        trans_lens[2]   = res_lens[2];
        trans_lens[3]   = blocksize;
        trans_lens.push_back(res_lens[3]);
        trans_lens.push_back(blocksize); // {N, C, Ho, blocksize, Wo, blocksize}
        std::vector<int64_t> perm = {0, 3, 5, 1, 2, 4};
        auto temp1 = info.add_instruction(make_op("reshape", {{"dims", trans_lens}}), args[0]);
        auto temp2 = info.add_instruction(make_op("transpose", {{"permutation", perm}}), temp1);
        return info.add_instruction(make_op("reshape", {{"dims", res_lens}}),
                                    info.make_contiguous(temp2));
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
