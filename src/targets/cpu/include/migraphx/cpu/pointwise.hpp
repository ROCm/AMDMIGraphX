#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_CPU_POINTWISE_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_CPU_POINTWISE_HPP

#include <migraphx/config.hpp>
#include <migraphx/context.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/cpu/context.hpp>
#include <migraphx/register_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

struct multi_index
{
    multi_index(const shape& s, std::size_t i) : n(s.lens().size())
    {
        assert(n < max_size);
        std::copy(s.lens().begin(), s.lens().end(), dims);
        s.multi_copy(i, index, index + max_size);
    }

    std::size_t size() const { return n; }

    std::size_t* begin() { return index; }
    const std::size_t* begin() const { return index; }

    std::size_t* end() { return index + size(); }
    const std::size_t* end() const { return index + size(); }

    std::size_t offset(const shape& s) const { return s.index(begin(), end()); }

    void carry()
    {
        std::size_t overflow = 0;
        for(std::ptrdiff_t i = size() - 1; i > 0; i--)
        {
            auto z = index[i] + overflow;
            // Reset overflow
            overflow = 0;
            // Compute overflow using while loop instead of mod
            while(z >= dims[i])
            {
                z -= dims[i];
                overflow += 1;
            }
            index[i] = z;
        }
        index[0] += overflow;
    }

    multi_index& operator++()
    {
        index[size() - 1]++;
        carry();
        return *this;
    }
    multi_index operator++(int)
    {
        multi_index result = *this;
        ++*this;
        return result;
    }

    private:
    static const std::size_t max_size = 6;
    std::size_t index[max_size];
    std::size_t dims[max_size];
    std::size_t n;
};

template <class... Ts>
auto pointwise(Ts... xs)
{
    return [=](context& ctx, const shape& base_shape, std::size_t min_grain, auto f) {
        assert(base_shape.lens().size() <= 6);
        ctx.bulk_execute(base_shape.elements(), min_grain, [=](auto start, auto end) mutable {
            multi_index mi(base_shape, start);
            for(auto i = start; i < end; i++)
            {
                f(xs.data()[mi.offset(xs.get_shape())]...);
                ++mi;
            }
        });
    };
}

template <class Op>
struct cpu_unary : auto_register_op<cpu_unary<Op>>
{
    Op op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }
    std::string name() const { return "cpu::" + op.name(); }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        auto s = inputs.at(0);
        return {s.type(), s.lens()};
    }

    argument compute(context& ctx, const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};

        visit_all(result, args[0])([&](auto output, auto input) {
            auto op2 = op;
            pointwise(output, input)(
                ctx, output.get_shape(), 1024, [op2](auto& y, auto x) { y = op2.apply()(x); });
        });

        return result;
    }
};

template <class Op>
struct cpu_binary : auto_register_op<cpu_binary<Op>>
{
    Op op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }
    std::string name() const { return "cpu::" + op.name(); }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        auto s = inputs.at(0);
        return {s.type(), s.lens()};
    }

    argument compute(context& ctx, const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};

        visit_all(result, args[0], args[1])([&](auto output, auto input1, auto input2) {
            auto op2 = op;
            pointwise(output, input1, input2)(
                ctx, output.get_shape(), 1024, [op2](auto& z, auto x, auto y) {
                    z = op2.apply()(x, y);
                });
        });

        return result;
    }
};

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
