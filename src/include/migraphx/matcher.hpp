#ifndef MIGRAPHX_GUARD_RTGLIB_MATCHER_HPP
#define MIGRAPHX_GUARD_RTGLIB_MATCHER_HPP

#include <migraphx/functional.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/config.hpp>
#include <unordered_map>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace match {

struct matcher_context
{
    matcher_context(instruction_ref i) : last(i) {}
    std::unordered_map<std::string, instruction_ref> instructions;
    instruction_ref not_found() const { return last; }

    template <class M>
    bool matched(M m, instruction_ref ins)
    {
        return m.match(*this, ins) != this->not_found();
    }

    private:
    instruction_ref last;
};

/// Convert a predicate function into a matcher
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

/// Convert a function into a matcher
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

/// Convert a function into a matcher
template <class F>
function_matcher<F> make_function_matcher(F f)
{
    return {f};
}

/// Converts a matcher to bind the instruction to name
template <class M>
auto bind_match(M m, std::string name)
{
    return make_function_matcher(
        [ =, name = std::move(name) ](matcher_context & ctx, instruction_ref ins) {
            auto result = m.match(ctx, ins);
            if(result != ctx.not_found())
                ctx.instructions.emplace(name, ins);
            return result;
        });
}

/// Convert a matcher to a bindable matcher
template <class M>
struct bindable_matcher
{
    M m;

    auto bind(std::string name) const { return bind_match(m, std::move(name)); }

    instruction_ref match(matcher_context& ctx, instruction_ref ins) const
    {
        return m.match(ctx, ins);
    }
};

/// Create a bindable matcher
template <class M>
bindable_matcher<M> make_bindable_matcher(M m)
{
    return {m};
}

/// Create a bindable matcher from a function
template <class F>
bindable_matcher<function_matcher<F>> make_bf_matcher(F f)
{
    return {{f}};
}

/// Create a bindable matcher from a predicate function
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

/// The basic matcher provides the all_of composability of the matcher
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

    auto bind(std::string name) const { return bind_match(m, std::move(name)); }

    instruction_ref match(matcher_context& ctx, instruction_ref ins) const
    {
        return m.match(ctx, ins);
    }
};

/// Create a basic matcher from a matcher
template <class M>
basic_matcher<M> make_basic_matcher(M m)
{
    return {m};
}

/// Create a basic matcher from a function
template <class F>
basic_matcher<function_matcher<F>> make_basic_fun_matcher(F f)
{
    return {{f}};
}

/// Create a basic matcher from a predicate function
template <class P>
basic_matcher<predicate_matcher<P>> make_basic_pred_matcher(P p)
{
    return {{p}};
}

/// This macro takes care of the boilerplate for defining a matcher
#define MIGRAPHX_BASIC_MATCHER(name, ...)                                     \
    struct name##_m                                                           \
    {                                                                         \
        instruction_ref match(__VA_ARGS__) const;                             \
    };                                                                        \
    const constexpr auto name = migraphx::match::basic_matcher<name##_m>{{}}; \
    inline instruction_ref name##_m::match(__VA_ARGS__) const

/// This macro takes care of the boilerplate for defining a predicate matcher
#define MIGRAPHX_PRED_MATCHER(name, ...)                                                  \
    struct name##_m                                                                       \
    {                                                                                     \
        bool operator()(__VA_ARGS__) const;                                               \
    };                                                                                    \
    const constexpr auto name =                                                           \
        migraphx::match::basic_matcher<migraphx::match::predicate_matcher<name##_m>>{{}}; \
    inline bool name##_m::operator()(__VA_ARGS__) const

struct matcher_result
{
    std::unordered_map<std::string, instruction_ref> instructions;
    instruction_ref result;
};

/// Match a single instruction
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

/// Find matches for an instruction in the program
template <class... Ms>
void find_matches(program& p, instruction_ref ins, Ms&&... ms)
{
    bool match = false;
    each_args(
        [&](auto&& m) {
            if(match)
                return;
            auto r = match_instruction(p, ins, m.matcher());
            if(r.result == p.end())
                return;
            m.apply(p, r);
            match = true;
        },
        ms...);
}

/// Find matches in a program
template <class... Ms>
void find_matches(program& p, Ms&&... ms)
{
    for(auto ins : iterator_for(p))
    {
        find_matches(p, ins, ms...);
    }
}

struct lazy_and
{
    template <class F, class G>
    bool operator()(F f, G g) const
    {
        return f() and g();
    }
};

struct lazy_or
{
    template <class F, class G>
    bool operator()(F f, G g) const
    {
        return f() or g();
    }
};

template <class Op, bool Start, bool Matches>
struct folder
{
    template<class... Ms>
    static bool fold_match(matcher_context& ctx, instruction_ref ins, Ms... ms)
    {
        Op op;
        auto matched = [&](auto m) { return [&] { return ctx.matched(m, ins); }; };
        return
            fold([&](auto x, auto y) { return op(always(x), matched(y)); })(Start, ms...);
    }


    template <class... Ts>
    auto operator()(Ts... ms) const
    {
        return make_bf_matcher([=](matcher_context& ctx, instruction_ref ins) {
            bool matches = folder::fold_match(ctx, ins, ms...);
            if(matches == Matches)
                return ins;
            return ctx.not_found();
        });
    }

    template <class Selector>
    auto operator[](Selector select) const
    {
        return [=](auto... ms) {
            return make_bf_matcher([=](matcher_context& ctx, instruction_ref start) {
                Op op;
                bool matches = Start;
                select(start, [&](auto ins) {
                    auto fm = [&] { return folder::fold_match(ctx, ins, ms...); };
                    matches = op(always(matches), fm);
                });
                if(matches == Matches)
                    return start;
                return ctx.not_found();
            });
        };
    }
};

const constexpr auto all_of  = folder<lazy_and, true, true>{};
const constexpr auto any_of  = folder<lazy_or, false, true>{};
const constexpr auto none_of = folder<lazy_or, false, false>{};

inline auto inputs()
{
    return [](auto ins, auto f) {
        for(auto&& x : ins->inputs())
            f(x);
    };
}

inline auto outputs()
{
    return [](auto ins, auto f) {
        for(auto&& x : ins->outputs())
            f(x);
    };
}

MIGRAPHX_PRED_MATCHER(any, instruction_ref) { return true; }
MIGRAPHX_PRED_MATCHER(none, instruction_ref) { return false; }
MIGRAPHX_PRED_MATCHER(standard_shape, instruction_ref ins) { return ins->get_shape().standard(); }
MIGRAPHX_PRED_MATCHER(broadcast_shape, instruction_ref ins)
{
    return ins->get_shape().broadcasted();
}

MIGRAPHX_PRED_MATCHER(transpose_shape, instruction_ref ins)
{
    return ins->get_shape().transposed();
}

MIGRAPHX_PRED_MATCHER(same_input_shapes, instruction_ref ins)
{
    if(ins->inputs().empty())
        return false;
    auto s = ins->inputs().front()->get_shape();
    return std::all_of(
        ins->inputs().begin(), ins->inputs().end(), [&](auto x) { return x->get_shape() == s; });
}

MIGRAPHX_BASIC_MATCHER(output, matcher_context& ctx, instruction_ref ins)
{
    if(ins->outputs().size() == 1)
        return ins->outputs().front();
    return ctx.not_found();
}

MIGRAPHX_BASIC_MATCHER(used_once, matcher_context& ctx, instruction_ref ins)
{
    if(ins->outputs().size() == 1)
        return ins;
    if(ins->outputs().empty() and std::next(ins) == ctx.not_found())
        return ins;
    return ctx.not_found();
}

template <class... Ms>
auto skip_output(Ms... ms)
{
    auto m = any_of(ms...);
    return make_basic_fun_matcher([=](matcher_context& ctx, instruction_ref start) {
        return fix<instruction_ref>([&](auto self, auto ins) {
            if(ins->outputs().size() == 1)
            {
                auto next = ins->outputs().front();
                if(ctx.matched(m, next))
                {
                    auto skipped_next = self(next);
                    if(skipped_next != ctx.not_found())
                        return skipped_next;
                }
                return next;
            }
            return ctx.not_found();
        })(start);
    });
}

inline auto name(std::string s)
{
    return make_basic_pred_matcher(
        [ =, s = std::move(s) ](instruction_ref ins) { return ins->name() == s; });
}

inline auto name(std::unordered_set<std::string> names)
{
    return make_basic_pred_matcher([ =, names = std::move(names) ](instruction_ref ins) {
        return names.count(ins->name()) > 0;
    });
}

inline auto nargs(std::size_t n)
{
    return make_basic_pred_matcher([=](instruction_ref ins) { return ins->inputs().size() == n; });
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
    return match::all_of(nargs(sizeof...(Ns)), arg(Ns)(ms)...);
}

template <class... Ms>
auto args(Ms... ms)
{
    return sequence_c<sizeof...(Ms)>([=](auto... is) {
        // It needs to be written as `decltype(is)::value` for gcc 5
        return args_impl(args_impl_ints<decltype(is)::value...>{}, ms...);
    });
}

inline auto either_arg(std::size_t i, std::size_t j)
{
    return [=](auto m1, auto m2) {
        return match::any_of(match::all_of(arg(i)(m1), arg(j)(m2)),
                             match::all_of(arg(j)(m1), arg(i)(m2)));
    };
}

template <class M>
auto same_shape(M m)
{
    return make_basic_fun_matcher([=](matcher_context& ctx, instruction_ref ins) {
        auto i = m.match(ctx, ins);
        if(i != ctx.not_found() and i->get_shape() == ins->get_shape())
            return ins;
        return ctx.not_found();
    });
}

template <class... Ms>
auto same_shape(Ms... ms)
{
    return all_of(same_shape(ms)...);
}

} // namespace match
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
