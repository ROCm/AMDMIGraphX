#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_depthtospace : op_parser<parse_depthtospace>
{
    std::vector<op_desc> operators() const { return {{"DepthToSpace"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        auto s = args[0]->get_shape();
        // mode attribute of DepthToSpace
        auto mode = std::string("DCR");
        if(contains(info.attributes, "mode"))
        {
            mode = info.attributes.at("mode").s(); // DCR or CRD?
        }
        // blocksize attribute of DepthToSpace
        int blocksize = 0;
        if(contains(info.attributes, "blocksize"))
        {
            blocksize = info.attributes.at("blocksize").i();
        }
        if(blocksize < 1)
        {
            MIGRAPHX_THROW("DepthToSpace: blocksize is less than 1");
        }
        // calculate dimensions
        auto lens1            = s.lens();
        auto lens2            = s.lens();
        unsigned long divisor = std::pow(blocksize, 2);
        if((lens2[1] % divisor) == 0)
            lens2[1] = lens2[1] / divisor;
        else
            MIGRAPHX_THROW("DepthToSpace: div by blocksize quotient not int ");
        lens1.push_back(lens1[2]);
        lens1.push_back(lens1[3]);
        lens2[2] = lens2[2] * blocksize;
        lens2[3] = lens2[3] * blocksize;
        lens1[2] = blocksize;
        std::vector<int64_t> perm;
        if(mode == "DCR")
        {
            lens1[3] = lens1[1] / divisor;
            lens1[1] = blocksize;
            perm     = {0, 3, 4, 1, 5, 2};
        }
        else if(mode == "CRD")
        {
            lens1[1] = lens1[1] / divisor;
            lens1[3] = blocksize;
            perm     = {0, 1, 4, 2, 5, 3};
        }
        else
            MIGRAPHX_THROW("DepthToSpace: mode attribute cannot be read.");

        auto temp1 = info.add_instruction(make_op("reshape", {{"dims", lens1}}), args[0]);
        auto temp2 = info.add_instruction(make_op("transpose", {{"permutation", perm}}), temp1);
        return info.add_instruction(make_op("reshape", {{"dims", lens2}}),
                                    info.make_contiguous(temp2));
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
