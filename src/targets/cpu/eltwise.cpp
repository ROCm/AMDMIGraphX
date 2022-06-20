#include <migraphx/config.hpp>
#include <migraphx/cpu/pointwise.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

struct dnnl_eltwise : dnnl_op<dnnl_eltwise, dnnl::eltwise_forward>
{
    std::string algo;
    float alpha = 0;
    float beta  = 0;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack_join(self.reflect_base(self, f),
                         pack(f(self.algo, "algo"), f(self.alpha, "alpha"), f(self.beta, "beta")));
    }

    std::string group() const { return this->name() + "::" + algo; }

    std::string name() const { return "dnnl::eltwise"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        // Compensate for allocation
        inputs.pop_back();
        check_shapes{this->trim_post_op_inputs(inputs), *this}.has(1).packed();
        auto s = inputs.at(0);
        auto r = s;
        if(not s.packed())
            r = shape{s.type(), s.lens()};
        // Call to get_primitive to make sure an algo is available
        this->get_primitive(this->to_memory_desc(r, inputs));
        return r;
    }

    dnnl::eltwise_forward::desc get_desc(const std::unordered_map<int, dnnl::memory::desc>& m) const
    {
        return {dnnl::prop_kind::forward_inference,
                to_dnnl_algo(algo),
                m.at(MIGRAPHX_DNNL_PREFIX(ARG_SRC_0)),
                alpha,
                beta};
    }
};

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
