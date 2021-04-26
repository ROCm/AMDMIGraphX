#include <migraphx/config.hpp>
#include <migraphx/cpu/dnnl.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

struct dnnl_layernorm : dnnl_op<dnnl_layernorm, dnnl::layer_normalization_forward>
{
    float epsilon = 1e-12f;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.epsilon, "epsilon"));
    }

    std::string name() const { return "dnnl::layernorm"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        // Compensate for allocation
        inputs.pop_back();
        check_shapes{this->trim_post_op_inputs(inputs), *this}.has(1);
        auto s = inputs.at(0);
        // Call to get_primitive to make sure an algo is available
        this->get_primitive(this->to_memory_desc(s, inputs));
        return s;
    }

    dnnl::layer_normalization_forward::desc
    get_desc(const std::unordered_map<int, dnnl::memory::desc>& m) const
    {
        return {dnnl::prop_kind::forward_inference,
                m.at(DNNL_ARG_SRC),
                1e-12f,
                dnnl::normalization_flags::none};
    }
};

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
