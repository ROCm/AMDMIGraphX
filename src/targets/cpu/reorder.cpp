#include <migraphx/config.hpp>
#include <migraphx/cpu/dnnl.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

struct dnnl_reorder : dnnl_op<dnnl_reorder, dnnl::reorder>
{
    std::string name() const { return "dnnl::reorder"; }

    shape adjust_shape(const shape& x, int) const { return x; }

    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        auto r = inputs.back();
        // Call to get_primitive to make sure an algo is available
        this->get_primitive(this->to_memory_desc(r, inputs));
        return r;
    }
    // Custom desc class since its missing in dnnl
    struct desc
    {
        dnnl::memory::desc src;
        dnnl::memory::desc dst;
    };
    desc get_desc(const std::unordered_map<int, dnnl::memory::desc>& m) const
    {
        return {m.at(MIGRAPHX_DNNL_PREFIX(ARG_SRC)), m.at(MIGRAPHX_DNNL_PREFIX(ARG_DST))};
    }

    auto get_primitive_desc(const desc& d, const dnnl::primitive_attr& attr) const
    {
        auto& engine = get_dnnl_context().engine;
        return dnnl::reorder::primitive_desc(engine, d.src, engine, d.dst, attr);
    }
};

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
