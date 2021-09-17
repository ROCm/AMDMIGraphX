#include <migraphx/config.hpp>
#include <migraphx/cpu/dnnl.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

struct dnnl_reduction : dnnl_op<dnnl_reduction, dnnl::reduction>
{
    std::string algo;
    std::vector<std::int64_t> axes{};
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack_join(self.reflect_base(self, f),
                         pack(f(self.algo, "algo"), f(self.axes, "axes")));
    }

    std::string name() const { return "dnnl::reduction"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        // Compensate for allocation
        inputs.pop_back();
        check_shapes{this->trim_post_op_inputs(inputs), *this}.has(1).standard();
        auto s    = inputs.at(0);
        auto lens = s.lens();
        for(auto axis : axes)
        {
            lens[axis] = 1;
        }
        auto r = shape{s.type(), lens};
        // Call to get_primitive to make sure an algo is available
        this->get_primitive(this->to_memory_desc(r, inputs));
        return r;
    }

    dnnl::reduction::desc get_desc(const std::unordered_map<int, dnnl::memory::desc>& m) const
    {
        return {to_dnnl_algo(algo),
                m.at(MIGRAPHX_DNNL_PREFIX(ARG_SRC)),
                m.at(MIGRAPHX_DNNL_PREFIX(ARG_DST)),
                0,
                0};
    }
};

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
