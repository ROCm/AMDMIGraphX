#include <migraphx/config.hpp>
#include <migraphx/cpu/dnnl.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

struct dnnl_binary : dnnl_op<dnnl_binary, dnnl::binary>
{
    std::string algo;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack_join(self.reflect_base(self, f), pack(f(self.algo, "algo")));
    }

    std::string group() const { return this->name() + "::" + algo; }

    std::string name() const { return "dnnl::binary"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        // Compensate for allocation
        inputs.pop_back();
        check_shapes{this->trim_post_op_inputs(inputs), *this}.has(2);
        auto s0 = inputs.at(0);
        auto s1 = inputs.at(1);
        auto r  = s0;
        if(s0 != s1 or !s0.packed())
        {
            r = shape{s0.type(), s0.lens()};
        }
        // Call to get_primitive to make sure an algo is available
        this->get_primitive(this->to_memory_desc(r, inputs));
        return r;
    }

    dnnl::binary::desc get_desc(const std::unordered_map<int, dnnl::memory::desc>& m) const
    {
        return {to_dnnl_algo(algo), m.at(DNNL_ARG_SRC_0), m.at(DNNL_ARG_SRC_1), m.at(DNNL_ARG_DST)};
    }
};

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
