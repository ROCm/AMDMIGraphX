#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_CPU_POINTWISE_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_CPU_POINTWISE_HPP

#include <migraphx/config.hpp>
#include <migraphx/context.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/cpu/context.hpp>
#include <migraphx/reduce_dims.hpp>
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

    void increment(std::size_t i)
    {
        index[size() - 1] += i;
        carry();
    }

    multi_index& operator+=(std::size_t i)
    {
        increment(i);
        return *this;
    }

    multi_index& operator++()
    {
        increment(1);
        return *this;
    }
    multi_index operator++(int)
    {
        multi_index result = *this;
        increment(1);
        return result;
    }

    private:
    static const std::size_t max_size = 5;
    std::size_t index[max_size];
    std::size_t dims[max_size];
    std::size_t n;
};

struct reduce_dims_base
{
    std::vector<shape> reduce_shapes;

    void finalize(context&, const shape&, const std::vector<shape>& inputs)
    {
        reduce_shapes = reduce_dims(inputs);
    }

    argument get_arg(const std::vector<argument>& args, std::size_t i) const
    {
        if(reduce_shapes.empty())
            return args[i];
        return args.at(i).reshape(reduce_shapes.at(i));
    }

    argument get_output() const
    {
        argument a{reduce_shapes[0]};
        return a;
    }
};

template <class X, class... Xs>
bool is_standard_offset(const X& x, const Xs&... xs)
{
    if(all_of({x, xs...}, [](const auto& s) { return s.standard(); }))
        return true;
    if(all_of({x, xs...}, [](const auto& s) { return s.packed(); }) and
       all_of({xs...}, [&](const auto& s) { return s == x; }))
        return true;
    return false;
}

template <class... Ts>
auto pointwise(Ts... xs)
{
    return [=](context& ctx, const shape& base_shape, std::size_t min_grain, auto f) {
        if(is_standard_offset(xs.get_shape()...))
        {
            ctx.bulk_execute(base_shape.elements(), min_grain, [=](auto start, auto end) mutable {
                for(auto i = start; i < end; i++)
                {
                    f(xs.data()[i]...);
                }
            });
        }
        else
        {
            assert(base_shape.lens().size() <= 6);
            ctx.bulk_execute(base_shape.elements(), min_grain, [=](auto start, auto end) mutable {
                multi_index mi(base_shape, start);
                for(auto i = start; i < end; i++)
                {
                    f(xs.data()[mi.offset(xs.get_shape())]...);
                    ++mi;
                }
            });
        }
    };
}

template <class Op>
struct cpu_unary : reduce_dims_base, auto_register_op<cpu_unary<Op>>
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
    // cppcheck-suppress constParameter
    argument compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const
    {
        argument result = get_arg(args, args.size() - 1);

        visit_all(result, get_arg(args, 0))([&](auto output, auto input) {
            auto op2 = op;
            pointwise(output, input)(
                ctx, output.get_shape(), 1024, [op2](auto& y, auto x) { y = op2.apply()(x); });
        });

        return result.reshape(output_shape);
    }

    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};

template <class Op>
struct cpu_binary : reduce_dims_base, auto_register_op<cpu_binary<Op>>
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
        check_shapes{inputs, *this}.has(3);
        auto s = inputs.at(0);
        return {s.type(), s.lens()};
    }

    // cppcheck-suppress constParameter
    argument compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const
    {
        argument result = get_arg(args, args.size() - 1);

        visit_all(result, get_arg(args, 0), get_arg(args, 1))(
            [&](auto output, auto input1, auto input2) {
                auto op2 = op;
                pointwise(output, input1, input2)(
                    ctx, output.get_shape(), 1024, [op2](auto& z, auto x, auto y) {
                        z = op2.apply()(x, y);
                    });
            });

        return result.reshape(output_shape);
    }

    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
