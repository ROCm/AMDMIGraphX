#include <migraphx/config.hpp>
#include <migraphx/cpu/pointwise.hpp>
#include <migraphx/op/concat.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

struct dnnl_concat : dnnl_extend_op<dnnl_concat, dnnl::concat, op::concat>
{
    std::vector<int> arg_map(int size) const
    {
        std::vector<int> result(size);
        std::iota(result.begin(), result.end(), MIGRAPHX_DNNL_PREFIX(ARG_MULTIPLE_SRC));
        return result;
    }
    // Custom desc class since its missing in dnnl
    struct desc
    {
        dnnl::memory::desc dst;
        std::size_t axis = 1;
        std::vector<dnnl::memory::desc> srcs;
    };
    desc get_desc(const std::unordered_map<int, dnnl::memory::desc>& m) const
    {
        std::vector<dnnl::memory::desc> srcs;
        srcs.reserve(m.size() - 1);

        for(auto i = 0; i < m.size() - 1; i++)
        {
            srcs.push_back(m.at(MIGRAPHX_DNNL_PREFIX(ARG_MULTIPLE_SRC) + i));
        }
        return {m.at(MIGRAPHX_DNNL_PREFIX(ARG_DST)), std::size_t(op.axis), srcs};
    }

    auto get_primitive_desc(const desc& d, const dnnl::primitive_attr& attr) const
    {
        return dnnl::concat::primitive_desc(d.dst, d.axis, d.srcs, get_dnnl_context().engine, attr);
    }
};

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
