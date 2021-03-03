#include <migraphx/config.hpp>
#include <migraphx/cpu/dnnl.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

struct dnnl_reorder : dnnl_op<dnnl_reorder, dnnl::reorder>
{
    template <class Self, class F>
    static auto reflect(Self&, F)
    {
        return pack();
    }

    std::string name() const { return "dnnl::reorder"; }

    shape adjust_shape(const shape& x, int) const { return x; }

    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        return inputs.back();
    }
    // Custom desc class since its missing in dnnl
    struct desc
    {
        dnnl::memory::desc src;
        dnnl::memory::desc dst;
    };
    desc get_desc(const std::unordered_map<int, dnnl::memory::desc>& m) const
    {
        return {m.at(DNNL_ARG_SRC), m.at(DNNL_ARG_DST)};
    }

    auto get_primitive_desc(const desc& d) const
    {
        auto& engine = get_dnnl_context().engine;
        return dnnl::reorder::primitive_desc(engine, d.src, engine, d.dst);
    }
};

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
