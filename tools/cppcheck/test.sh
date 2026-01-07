#!/bin/bash
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create a temporary build directory
BUILD_DIR=$(mktemp -d -t cppcheck-build-XXXXXX)

# Ensure cleanup on exit (success or failure)
trap "rm -rf '$BUILD_DIR'" EXIT

# ANSI color codes
if [[ -t 1 ]]; then
    # Only use colors if stdout is a terminal
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    NC='\033[0m' # No Color
else
    # No colors if output is redirected
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    BOLD=''
    NC=''
fi

# Function to show usage
show_usage() {
    echo -e "${BOLD}Usage:${NC} $0 [options] [test_name1] [test_name2] ..."
    echo -e "${BOLD}Options:${NC}"
    echo -e "  ${CYAN}-h, --help${NC}              Show this help message"
    echo -e "  ${CYAN}-l, --list${NC}              List all available tests"
    echo -e "  ${CYAN}--no-inline-suppr${NC}       Disable inline suppressions (--inline-suppr)"
    echo -e "  ${CYAN}--cxx <compiler>${NC}        Use specified compiler for syntax check (default: c++)"
    echo -e "  ${CYAN}--no-syntax-check${NC}       Skip the C++ syntax validation step"
    echo ""
    echo -e "${BOLD}Examples:${NC}"
    echo -e "  ${GREEN}$0${NC}                          # Run all tests"
    echo -e "  ${GREEN}$0 conditional_assert${NC}       # Run only conditional_assert.cpp"
    echo -e "  ${GREEN}$0 empty_if empty_for${NC}       # Run empty_if_statement.cpp and empty_for_statement.cpp"
    echo -e "  ${GREEN}$0 --no-inline-suppr${NC}        # Run all tests without inline suppressions"
    echo -e "  ${GREEN}$0 --no-inline-suppr empty_if${NC}  # Run empty_if_statement.cpp without inline suppressions"
    echo -e "  ${GREEN}$0 conditional_assert --no-inline-suppr${NC}  # Run conditional_assert.cpp without inline suppressions"
    echo -e "  ${GREEN}$0 --cxx g++${NC}                # Use g++ for syntax check"
    echo -e "  ${GREEN}$0 --cxx clang++${NC}            # Use clang++ for syntax check"
    echo -e "  ${GREEN}$0 --no-syntax-check${NC}        # Skip syntax validation"
    echo -e "  ${GREEN}$0 -l${NC}                       # List all available tests"
}

# Function to list available tests
list_tests() {
    echo -e "${BOLD}Available tests:${NC}"
    for file in $SCRIPT_DIR/test/*.cpp; do
        basename=$(basename "$file" .cpp)
        echo -e "  ${BLUE}-${NC} ${CYAN}$basename${NC}"
    done
}

# Parse command line arguments
INLINE_SUPPR_FLAG="--inline-suppr"
CXX_COMPILER="c++"
SYNTAX_CHECK=1
TEST_NAMES=()

# Process all arguments and separate options from test names
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -l|--list)
            list_tests
            exit 0
            ;;
        --no-inline-suppr)
            INLINE_SUPPR_FLAG=""
            shift
            ;;
        --cxx)
            if [[ -z "$2" || "$2" == -* ]]; then
                echo -e "${RED}Error:${NC} --cxx requires a compiler argument"
                exit 1
            fi
            CXX_COMPILER="$2"
            shift 2
            ;;
        --no-syntax-check)
            SYNTAX_CHECK=0
            shift
            ;;
        -*)
            echo -e "${RED}Error:${NC} Unknown option: ${YELLOW}$1${NC}"
            show_usage
            exit 1
            ;;
        *)
            TEST_NAMES+=("$1")
            shift
            ;;
    esac
done

# Determine which test files to run
if [ ${#TEST_NAMES[@]} -eq 0 ]; then
    # No arguments: run all tests
    TEST_FILES="$SCRIPT_DIR/test/*.cpp"
    echo -e "${BOLD}${BLUE}Running cppcheck with migraphx addon and rules on ALL test files...${NC}"
else
    # Arguments provided: run specific tests
    TEST_FILES=""
    for test_name in "${TEST_NAMES[@]}"; do
        # Handle both with and without .cpp extension
        test_file="$SCRIPT_DIR/test/${test_name%.cpp}.cpp"
        if [ -f "$test_file" ]; then
            TEST_FILES="$TEST_FILES $test_file"
        else
            echo -e "${RED}Error:${NC} Test file not found: ${YELLOW}$test_file${NC}"
            echo -e "Use ${CYAN}-h${NC} or ${CYAN}--help${NC} to see available tests"
            exit 1
        fi
    done
    echo -e "${BOLD}${BLUE}Running cppcheck with migraphx addon and rules on selected test files...${NC}"
fi

echo -e "${BOLD}Test files to be processed:${NC}"
for file in $TEST_FILES; do
    echo -e "  ${BLUE}-${NC} ${CYAN}$(basename "$file")${NC}"
done
echo ""

# Run syntax check on test files
if [ "$SYNTAX_CHECK" -eq 1 ]; then
    echo -e "${BOLD}${BLUE}Checking C++ syntax with ${CXX_COMPILER}...${NC}"
    SYNTAX_FAILED=0
    for file in $TEST_FILES; do
        if ! $CXX_COMPILER -std=c++17 -fsyntax-only "$file" 2>&1; then
            SYNTAX_FAILED=1
        fi
    done
    if [ "$SYNTAX_FAILED" -eq 1 ]; then
        echo -e "${RED}Error:${NC} Syntax check failed"
        exit 1
    fi
    echo -e "${GREEN}Syntax check passed${NC}"
    echo ""
fi

# Run cppcheck with the selected test files
# Note: We suppress built-in cppcheck style checks that are unrelated to migraphx rules
# to focus testing on the addon and custom XML rules only.
cppcheck -j $(nproc) \
    -D CPPCHECK=1 \
    --error-exitcode=1 \
    --enable=information,warning,style \
    --addon=$SCRIPT_DIR/migraphx.py \
    --rule-file=$SCRIPT_DIR/rules.xml $INLINE_SUPPR_FLAG \
    --suppress=constVariable \
    --suppress=constVariablePointer \
    --suppress=knownConditionTrueFalse \
    --suppress=missingIncludeSystem \
    --suppress=noConstructor \
    --suppress=unassignedVariable \
    --suppress=unreachableCode \
    --suppress=unreadVariable \
    --suppress=unmatchedSuppression \
    --suppress=unusedAllocatedMemory \
    --suppress=unusedStructMember \
    --cppcheck-build-dir="$BUILD_DIR" $TEST_FILES

echo ""
echo -e "${BOLD}${GREEN}Selected tests passed successfully!${NC}"
echo -e "${BOLD}Tested files:${NC}"
for file in $TEST_FILES; do
    echo -e "  ${GREEN}✓${NC} ${CYAN}$(basename "$file")${NC}"
done
echo ""
echo -e "${BOLD}${GREEN}Success!${NC} Selected cppcheck rules and addon checks are working correctly."
