#include <migraphx/config.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/reflect.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/context.hpp>
#include <migraphx/cpu/context.hpp>
#include <migraphx/cpu/dnnl.hpp>
#include <migraphx/op/convolution.hpp>
#include <migraphx/op/quant_convolution.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

template <class V, class T, class... Ts>
void visit_quantize_impl(V&& v, T&& x, Ts&&... xs)
{
    x.visit([&](auto y) { visit_all(xs...)([&](auto... ys) { v(y, ys...); }); });
}

template <class T, class... Ts>
auto visit_quantize(T&& x, Ts&&... xs)
{
    return [&](auto v) {
        // Workaround for https://gcc.gnu.org/bugzilla/show_bug.cgi?id=70100
        visit_quantize_impl(v, x, xs...);
    };
}

template <class Op>
struct cpu_convolution : auto_register_op<cpu_convolution<Op>>
{
    Op op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    std::string name() const { return "cpu::" + op.name(); }
    shape compute_shape(const std::vector<shape>& inputs) const { return op.compute_shape(inputs); }
    argument compute(context&, shape output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        visit_quantize(result, args[0], args[1])([&](auto output, auto input, auto weights) {
            auto in_lens = input.get_shape().lens();

            auto wei_lens = weights.get_shape().lens();
            auto wei_n    = wei_lens[0];
            auto wei_c    = wei_lens[1];
            std::vector<std::size_t> win_size(wei_lens.begin() + 1, wei_lens.end());

            par_for(output_shape.elements(), [&](auto i) {
                auto idx_o = output_shape.multi(i);
                auto w     = idx_o[1];
                auto n_dim = idx_o.size();

                std::vector<std::ptrdiff_t> win_start;
                for(std::size_t dim = 2; dim < n_dim; ++dim)
                {
                    auto d_2 = dim - 2;
                    win_start.push_back(std::ptrdiff_t(idx_o[dim] * op.stride[d_2]) -
                                        std::ptrdiff_t(op.padding[d_2]));
                }
                const auto group_id = w / (wei_n / op.group);

                shape win_shape{output_shape.type(), win_size};

                double acc = 0.0;
                shape_for_each(win_shape, [&](auto idx_win) {
                    auto k           = idx_win[0];
                    const auto in_ch = group_id * wei_c + k;
                    std::vector<std::ptrdiff_t> idx(idx_o.begin(), idx_o.end());
                    idx[1] = in_ch;
                    std::transform(idx_win.begin() + 1,
                                   idx_win.end(),
                                   win_start.begin(),
                                   idx.begin() + 2,
                                   [](std::ptrdiff_t ii, std::ptrdiff_t jj) { return ii + jj; });
                    std::vector<std::ptrdiff_t> idx_wei(idx_o.size());
                    idx_wei[0] = w;
                    std::copy(idx_win.begin(), idx_win.end(), idx_wei.begin() + 1);
                    if(std::all_of(idx.begin() + 2, idx.end(), [&](auto ii) { return ii >= 0; }) and
                       std::equal(idx.begin(),
                                  idx.end(),
                                  in_lens.begin(),
                                  in_lens.end(),
                                  std::less<std::ptrdiff_t>{}))
                    {
                        acc +=
                            input(idx.begin(), idx.end()) * weights(idx_wei.begin(), idx_wei.end());
                    }
                });

                output[i] = acc;
            });
        });
        return result;
    }
};
template struct cpu_convolution<op::quant_convolution>;
template struct cpu_convolution<op::convolution>;

#if USE_DNNL
struct dnnl_convolution : auto_register_op<dnnl_convolution>
{
    op::convolution op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    std::string name() const { return "dnnl::" + op.name(); }
    shape compute_shape(const std::vector<shape>& inputs) const { return op.compute_shape(inputs); }
    argument compute(context& ctx, shape output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        // In DNNL dilation is zero-based
        auto dilation = op.dilation;
        std::transform(
            dilation.begin(), dilation.end(), dilation.begin(), [](auto x) { return x - 1; });
        execute_dnnl<dnnl::convolution_forward>(ctx,
                                                {{DNNL_ARG_SRC, args[0]},
                                                 {DNNL_ARG_WEIGHTS, args[1]},
                                                 { DNNL_ARG_DST,
                                                   result }})([&](auto m) {
            return dnnl::convolution_forward::desc(dnnl::prop_kind::forward_inference,
                                                   dnnl::algorithm::convolution_auto,
                                                   m.at(DNNL_ARG_SRC).get_desc(),
                                                   m.at(DNNL_ARG_WEIGHTS).get_desc(),
                                                   m.at(DNNL_ARG_DST).get_desc(),
                                                   to_dnnl_dims(op.stride),
                                                   to_dnnl_dims(dilation),
                                                   to_dnnl_dims(op.padding),
                                                   to_dnnl_dims(op.padding));
        });
        return result;
    }
};
#endif

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
