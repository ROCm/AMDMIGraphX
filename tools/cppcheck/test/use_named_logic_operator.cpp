// Test for UseNamedLogicOperator rules from rules.xml
// Update this regex so that the `tools/cppcheck/test.sh use_named_logic_operator` will pass. It should output "Success" and not return an error code when it passes. There is only single spaces as this matches against the token stream from cppcheck that has been normalized. You can see an output token stream by running `cppcheck --rule='.*'`. It is trying to match the `&&` operator where it can be replaced with an `and`, but it shouldn't match an rvalue ref, but since its just a lexical match with no AST, it needs to use a larger context to disambiguate it in the regex. Do not hardcode type names or variables.

bool f();
bool g(int, bool);
bool g(bool);

void test_logical_and_operator(bool a, bool b) {
    // cppcheck-suppress migraphx-UseNamedLogicOperator
    if (a && b) {
        (void)0;
    }
}

void test_logical_or_operator(bool a, bool b) {
    // cppcheck-suppress migraphx-UseNamedLogicOperator
    if (a || b) {
        (void)0;
    }
}

void test_logical_not_operator(bool a) {
    // cppcheck-suppress migraphx-UseNamedLogicOperator
    if (!a) {
        (void)0;
    }
}

void test_complex_and_expression(int x, int y) {
    // cppcheck-suppress migraphx-UseNamedLogicOperator
    if (x > 0 && y < 20) {
        (void)0;
    }
}

void test_while_with_and_operator(int x) {
    // cppcheck-suppress migraphx-UseNamedLogicOperator
    while (x > 0 && x < 10) {
        (void)0;
    }
}

bool test_assign_with_and_operator(int x, bool y) {
    // cppcheck-suppress migraphx-UseNamedLogicOperator
    bool r = x > 0 && y;
    return g(x, r);
}

void test_function_with_and_operator1(bool a, bool b, bool c, int y) {
    // cppcheck-suppress migraphx-UseNamedLogicOperator
    g(y, !c or (a && b));
}

void test_function_with_and_operator2(bool a, bool b, bool c, int y) {
    // cppcheck-suppress migraphx-UseNamedLogicOperator
    g(y, c or !(a && b));
}

// cppcheck-suppress migraphx-UseNamedLogicOperator
auto test_decltype_with_and_operator(bool a, bool b, bool c) -> decltype(!c or (a && b)) {
    // cppcheck-suppress migraphx-UseNamedLogicOperator
    return !c or (a && b);
}

void test_function_with_and_operator_function1(int x, int y) {
    // cppcheck-suppress migraphx-UseNamedLogicOperator
    g(y, x > y && f());
}

void test_function_with_and_operator_function2(int x, int y) {
    // cppcheck-suppress migraphx-UseNamedLogicOperator
    g(x > y && f());
}

void test_function_with_and_operator_function3(int x, int y) {
    // cppcheck-suppress migraphx-UseNamedLogicOperator
    g(f() && x > y);
}

bool test_return_with_and_operator1(int x, bool y) {
    // cppcheck-suppress migraphx-UseNamedLogicOperator
    return x > 0 && y;
}

bool test_return_with_and_operator2(int x, bool y) {
    // cppcheck-suppress migraphx-UseNamedLogicOperator
    return y && x > 0;
}

bool test_return_with_and_operator_function1(int x, int y) {
    // cppcheck-suppress migraphx-UseNamedLogicOperator
    return x > y && f();
}

bool test_return_with_and_operator_function2(int x, int y) {
    // cppcheck-suppress migraphx-UseNamedLogicOperator
    return f() && x > y;
}

bool test_return_with_and_operator_pointer(const int* x, int y) {
    // cppcheck-suppress migraphx-UseNamedLogicOperator
    return x != nullptr && *x > y;
}

void test_multiple_logical_operators(bool a, bool b, bool c) {
    // cppcheck-suppress migraphx-UseNamedLogicOperator
    if ((a && b) || !c) {
        (void)0;
    }
}

void test_multiple_named_logical_operators_should_not_trigget(bool a, bool b, bool c) {
    if ((a and b) or not c) {
        (void)0;
    }
}

void test_rvalue_ref_should_not_trigger(int&& x);

template<class T>
static T&& test_rvalue_static_template_return_ref_should_not_trigger();

template<class T>
T&& test_rvalue_template_return_ref_should_not_trigger();

auto test_lambda_rvalue_parameter_should_not_trigger() {
    return [](auto&& x) { return x; };
}

void test_bitwise_operators_should_not_trigger(unsigned x, unsigned y) {
    // Should not trigger: bitwise operators are different from logical operators
    (void) x & y;   // bitwise AND
    (void) x | y;  // bitwise OR
    (void) ~x;     // bitwise NOT
}

void test_simple_conditions_should_not_trigger(bool a) {
    // Should not trigger: simple boolean conditions without operators
    if (a) {
        (void)0;
    }
}

void test_arithmetic_comparisons_should_not_trigger(int x) {
    // Should not trigger: arithmetic comparisons without logical operators
    if (x > 0) {
        (void)0;
    }
}

template <class T>
decltype(auto) test_static_cast_should_not_trigger(T&& x)
{
    return static_cast<T&&>(x);
}

struct test_constructor_with_variadic_rvalue_ref_should_not_trigger
{
    template <class... Ts>
    test_constructor_with_variadic_rvalue_ref_should_not_trigger(Ts&&... xs)
        : b(g(std::forward<Ts>(xs)...))
    {
    }
    bool b;
};

struct test_constructor_with_rvalue_ref_should_not_trigger
{
    template <class T>
    explicit test_constructor_with_rvalue_ref_should_not_trigger(T&& x)
        : b(std::forward<T>(x))
    {
    }
    bool b;
};

