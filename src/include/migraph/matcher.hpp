#ifndef MIGRAPH_GUARD_RTGLIB_MATCHER_HPP
#define MIGRAPH_GUARD_RTGLIB_MATCHER_HPP

#include <migraph/functional.hpp>
#include <migraph/ranges.hpp>
#include <migraph/instruction.hpp>
#include <migraph/program.hpp>
#include <migraph/type_name.hpp>
#include <unordered_map>

namespace migraph {

namespace matchers {

struct matcher_context
{
    matcher_context(instruction_ref i) : last(i) {}
    std::unordered_map<std::string, instruction_ref> instructions;
    instruction_ref not_found() const { return last; }

    private:
    instruction_ref last;
};

template <class P>
struct predicate_matcher
{
    P p;

    instruction_ref match(matcher_context& ctx, instruction_ref ins) const
    {
        assert(ins != ctx.not_found());
        if(p(ins))
            return ins;
        return ctx.not_found();
    }
};

template <class F>
struct function_matcher
{
    F f;

    instruction_ref match(matcher_context& ctx, instruction_ref ins) const
    {
        assert(ins != ctx.not_found());
        return f(ctx, ins);
    }
};

template <class F>
function_matcher<F> make_function_matcher(F f)
{
    return {f};
}

template <class M>
auto bind_match(M m, std::string name)
{
    return make_function_matcher([=](matcher_context& ctx, instruction_ref ins) {
        auto result = m.match(ctx, ins);
        if(result != ctx.not_found())
            ctx.instructions.emplace(name, ins);
        return result;
    });
}

template <class M>
struct bindable_matcher
{
    M m;

    auto bind(std::string name) { return bind_match(m, name); }

    instruction_ref match(matcher_context& ctx, instruction_ref ins) const
    {
        return m.match(ctx, ins);
    }
};

template <class M>
bindable_matcher<M> make_bindable_matcher(M m)
{
    return {m};
}

template <class F>
bindable_matcher<function_matcher<F>> make_bf_matcher(F f)
{
    return {{f}};
}

template <class F>
bindable_matcher<predicate_matcher<F>> make_bp_matcher(F f)
{
    return {{f}};
}

using bool_list = std::initializer_list<bool>;

struct id_matcher
{
    instruction_ref match(matcher_context&, instruction_ref ins) const { return ins; }
};

template <class M>
struct basic_matcher
{
    M m;

    template <class... Ts>
    auto operator()(Ts... ms) const
    {
        // Copy m because we cant capture `this` by value
        auto mm = m;
        return make_bf_matcher([=](matcher_context& ctx, instruction_ref ins) {
            auto result = mm.match(ctx, ins);
            if(result != ctx.not_found())
            {
                bool matches = fold([&](auto x, auto y) {
                    return x and y.match(ctx, result) != ctx.not_found();
                })(true, ms...);
                if(matches)
                    return result;
            }
            return ctx.not_found();
        });
    }

    auto bind(std::string name) { return bind_match(m, name); }

    instruction_ref match(matcher_context& ctx, instruction_ref ins) const
    {
        return m.match(ctx, ins);
    }
};

template <class M>
basic_matcher<M> make_basic_matcher(M m)
{
    return {m};
}

template <class F>
basic_matcher<function_matcher<F>> make_basic_fun_matcher(F f)
{
    return {{f}};
}

template <class P>
basic_matcher<predicate_matcher<P>> make_basic_pred_matcher(P p)
{
    return {{p}};
}

#define MIGRAPH_BASIC_MATCHER(name, ...)                              \
    struct name##_m                                                   \
    {                                                                 \
        instruction_ref match(__VA_ARGS__) const;                     \
    };                                                                \
    const constexpr auto name = migraph::matchers::basic_matcher<name##_m>{{}}; \
    inline instruction_ref name##_m::match(__VA_ARGS__) const

#define MIGRAPH_PRED_MATCHER(name, ...)                                                  \
    struct name##_m                                                                      \
    {                                                                                    \
        bool operator()(__VA_ARGS__) const;                                              \
    };                                                                                   \
    const constexpr auto name = migraph::matchers::basic_matcher<predicate_matcher<name##_m>>{{}}; \
    inline bool name##_m::operator()(__VA_ARGS__) const

struct matcher_result
{
    std::unordered_map<std::string, instruction_ref> instructions;
    instruction_ref result;
};

template <class M>
matcher_result match_instruction(program& p, instruction_ref ins, M&& m)
{
    assert(ins != p.end());
    matcher_result result;
    matcher_context ctx{p.end()};
    result.result       = m.match(ctx, ins);
    result.instructions = ctx.instructions;
    return result;
}

template <class... Ts>
auto all_of(Ts... ms)
{
    return make_bf_matcher([=](matcher_context& ctx, instruction_ref ins) {
        bool matches = fold([&](auto x, auto y) {
            return x and y.match(ctx, ins) != ctx.not_found();
        })(true, ms...);
        if(matches)
            return ins;
        return ctx.not_found();
    });
}

template <class... Ts>
auto none_of(Ts... ms)
{
    return make_bf_matcher([=](matcher_context& ctx, instruction_ref ins) {
        bool matches = fold([&](auto x, auto y) {
            return x and y.match(ctx, ins) == ctx.not_found();
        })(true, ms...);
        if(matches)
            return ins;
        return ctx.not_found();
    });
}

template <class... Ts>
auto any_of(Ts... ms)
{
    return make_bf_matcher([=](matcher_context& ctx, instruction_ref ins) {
        bool matches = fold(
            [&](auto x, auto y) { return x or y.match(ctx, ins) != ctx.not_found(); })(true, ms...);
        if(matches)
            return ins;
        return ctx.not_found();
    });
}

MIGRAPH_PRED_MATCHER(standard_shape, instruction_ref ins) { return ins->get_shape().standard(); }

inline auto name(std::string name)
{
    return make_basic_pred_matcher([=](instruction_ref ins) { return ins->name() == name; });
}

inline auto arg(std::size_t i)
{
    return make_basic_fun_matcher([=](matcher_context& ctx, instruction_ref ins) {
        if(i < ins->inputs().size())
            return ins->inputs()[i];
        return ctx.not_found();
    });
}

// Workaround for bugs in clang
template <std::size_t...>
struct args_impl_ints
{
};

template <std::size_t... Ns, class... Ms>
auto args_impl(args_impl_ints<Ns...>, Ms... ms)
{
    return matchers::all_of(arg(Ns)(ms)...);
}

template <class... Ms>
auto args(Ms... ms)
{
    return sequence_c<sizeof...(Ms)>([=](auto... is) {
        // It needs to be written as `decltype(is)::value` for gcc 5
        return args_impl(args_impl_ints<decltype(is)::value...>{}, ms...);
    });
}

} // namespace matchers

} // namespace migraph

#endif
