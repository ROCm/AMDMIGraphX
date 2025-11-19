// Test for UnnecessaryEmptyCondition check
#include <vector>

void test_positive_cases() {
    std::vector<int> container = {1, 2, 3, 4, 5};
    
    // Should trigger: unnecessary empty check before range-based for
    // cppcheck-suppress migraphx-UnnecessaryEmptyCondition
    if (!container.empty()) {
        for (auto item : container) {
            // Process item
            int x = item * 2;
        }
    }
    
    // Should trigger: another case with different variable name
    std::vector<std::string> items = {"a", "b", "c"};
    // cppcheck-suppress migraphx-UnnecessaryEmptyCondition
    if (!items.empty()) {
        for (const auto& item : items) {
            // Process item
            std::string processed = item + "_processed";
        }
    }
}

void test_negative_cases() {
    std::vector<int> container = {1, 2, 3, 4, 5};
    
    // Should not trigger: different containers
    std::vector<int> other_container = {6, 7, 8};
    if (!container.empty()) {
        for (auto item : other_container) {
            int x = item * 2;
        }
    }
    
    // Should not trigger: no empty check
    for (auto item : container) {
        int x = item * 2;
    }
    
    // Should not trigger: traditional for loop
    if (!container.empty()) {
        for (size_t i = 0; i < container.size(); i++) {
            int x = container[i] * 2;
        }
    }
    
    // Should not trigger: checking size instead of empty
    if (container.size() > 0) {
        for (auto item : container) {
            int x = item * 2;
        }
    }
}
