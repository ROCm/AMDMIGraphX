# MiGraphX Cppcheck Test Suite

This directory contains comprehensive tests for the MiGraphX cppcheck addon (`migraphx.py`) and XML rules (`rules.xml`).

## Test Organization

Each test file corresponds to a specific check from either the Python addon or XML rules:

### Python Addon Tests (`migraphx.py`)

1. **avoid_branching_statement_as_last_in_loop.cpp** - Tests `AvoidBranchingStatementAsLastInLoop`
2. **conditional_assert.cpp** - Tests `ConditionalAssert`
3. **empty_catch_statement.cpp** - Tests `EmptyCatchStatement`
4. **empty_do_while_statement.cpp** - Tests `EmptyDoWhileStatement`
5. **empty_else_block.cpp** - Tests `EmptyElseBlock`
6. **empty_for_statement.cpp** - Tests `EmptyForStatement`
7. **empty_if_statement.cpp** - Tests `EmptyIfStatement`
8. **empty_switch_statement.cpp** - Tests `EmptySwitchStatement`
9. **empty_while_statement.cpp** - Tests `EmptyWhileStatement`
10. **for_loop_should_be_while_loop.cpp** - Tests `ForLoopShouldBeWhileLoop`
11. **goto_statement.cpp** - Tests `GotoStatement`
12. **lambda_attribute.cpp** - Tests `LambdaAttribute`
13. **multiple_unary_operator.cpp** - Tests `MultipleUnaryOperator`
14. **mutable_variable.cpp** - Tests `MutableVariable`
15. **nested_blocks.cpp** - Tests `NestedBlocks`
16. **redundant_cast.cpp** - Tests `RedundantCast`
17. **redundant_conditional_operator.cpp** - Tests `RedundantConditionalOperator`
18. **redundant_if_statement.cpp** - Tests `RedundantIfStatement`
19. **redundant_local_variable.cpp** - Tests `RedundantLocalVariable`
20. **unnecessary_empty_condition.cpp** - Tests `UnnecessaryEmptyCondition`
21. **use_device_launch.cpp** - Tests `UseDeviceLaunch`
22. **use_manage_pointer.cpp** - Tests `UseManagePointer`
23. **use_smart_pointer.cpp** - Tests `UseSmartPointer`
24. **use_stl_algorithms.cpp** - Tests `useStlAlgorithms`
25. **matcher_nested_parentheses.cpp** - Tests `MatcherNestedParentheses`

### XML Rules Tests (`rules.xml`)

26. **unused_deref.cpp** - Tests `UnusedDeref` rule
27. **strlen_empty_string.cpp** - Tests `StrlenEmptyString` rule
28. **define_rules.cpp** - Tests `defineUpperCase` and `definePrefix` rules
29. **use_named_logic_operator.cpp** - Tests `UseNamedLogicOperator` rules
30. **unnecessary_else_statement.cpp** - Tests `UnnecessaryElseStatement` rule
31. **inverted_logic.cpp** - Tests `InvertedLogic` rule
32. **use_stl_algorithm_loops.cpp** - Tests `useStlAlgorithm` rules for loop patterns

## Test Structure

Each test file contains:

- **Positive test cases**: Code patterns that should trigger the check, marked with `// cppcheck-suppress [CheckId]` to suppress the expected warning
- **Negative test cases**: Code patterns that should NOT trigger the check

## Running Tests

Execute the test script to run all tests:

```bash
./test.sh
```

The script will:
1. Run cppcheck with the migraphx addon and XML rules on all test files
2. Display detailed processing information
3. Report any issues found
4. Show success message if all tests pass

## Expected Output

The test run shows:
- Rule processing for each XML rule
- Warning/error detection for problematic code patterns
- Suppression matching for expected warnings
- Compilation warnings and errors (which are expected for test cases)

## Notes

- Some `Unmatched suppression` warnings are expected and indicate areas where test cases may need refinement
- Compilation errors in test files are expected as they test edge cases and problematic patterns
- The test suite validates both positive detection (catching bad patterns) and negative detection (not flagging good patterns)

## Test Coverage

This test suite provides comprehensive coverage for:
- All Python addon checks in `migraphx.py`
- All XML pattern-matching rules in `rules.xml`
- Both positive and negative test cases for each check
- Edge cases and boundary conditions

The tests ensure that the MiGraphX coding standards are properly enforced through static analysis.
