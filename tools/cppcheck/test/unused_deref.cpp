// Test for UnusedDeref rule from rules.xml

void test_redundant_deref_with_increment()
{
    int x = 5;
    // cppcheck-suppress UnusedDeref
    int* ptr = &x;
    // cppcheck-suppress clarifyStatement
    *ptr++;
}

void test_redundant_deref_with_decrement()
{
    int x = 5;
    // cppcheck-suppress UnusedDeref
    int* ptr = &x;
    // cppcheck-suppress clarifyStatement
    *ptr--;
}

void test_redundant_deref_with_increment_variant()
{
    int x = 5;
    // cppcheck-suppress UnusedDeref
    int* ptr = &x;
    // cppcheck-suppress clarifyStatement
    *ptr++;
}

void test_proper_dereference_should_not_trigger()
{
    // Should not trigger: proper dereference for reading value
    int x    = 5;
    int* ptr = &x;
    int y    = *ptr;
    (void)y; // Suppress unused variable warning
}

void test_assignment_through_dereference_should_not_trigger()
{
    // Should not trigger: assignment through dereference
    int x    = 5;
    int* ptr = &x;
    *ptr     = 10;
}

void test_increment_without_dereference_should_not_trigger()
{
    // Should not trigger: increment without dereference
    int x    = 5;
    int* ptr = &x;
    ptr++;
    (void)ptr;
}

void test_dereference_in_expression_should_not_trigger()
{
    // Should not trigger: dereference used in expression
    int x    = 5;
    int* ptr = &x;
    int z    = (*ptr) + 1;
    (void)z; // Suppress unused variable warning
}
