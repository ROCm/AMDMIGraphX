
include(CheckCXXSourceCompiles)

function(find_parallel_stl_check RESULT)
    set(CMAKE_REQUIRED_LIBRARIES ${ARGN})
    set(CMAKE_REQUIRED_FLAGS)
    if(NOT MSVC)
        set(CMAKE_REQUIRED_FLAGS "-std=c++17")
    endif()
    string(MD5 _flags_hash "${CMAKE_REQUIRED_FLAGS} ${CMAKE_REQUIRED_LIBRARIES}")
    set(_source "
#include <execution>

int main() {
    int* i = nullptr;
    std::sort(std::execution::par, i, i);
}
")
    check_cxx_source_compiles("${_source}" _has_execution_${_flags_hash})
    set(${RESULT} ${_has_execution_${_flags_hash}} PARENT_SCOPE)
endfunction()

set(ParallelSTL_FOUND Off)
set(ParallelSTL_LIBRARIES)
set(ParallelSTL_USES_TBB Off)
find_parallel_stl_check(ParallelSTL_HAS_EXECUTION_PAR)
if(ParallelSTL_HAS_EXECUTION_PAR)
    set(ParallelSTL_FOUND On)
else()
    find_package(TBB QUIET)
    if(TARGET TBB::tbb)
        find_parallel_stl_check(ParallelSTL_TBB_HAS_EXECUTION_PAR TBB::tbb)
        if(ParallelSTL_TBB_HAS_EXECUTION_PAR)
            set(ParallelSTL_USES_TBB On)
            set(ParallelSTL_LIBRARIES TBB::tbb)
            message(STATUS "Using TBB for parallel execution")
        endif()
    endif()
endif()

foreach(VAR ParallelSTL_FOUND ParallelSTL_LIBRARIES ParallelSTL_USES_TBB)
    string(TOUPPER ${VAR} ParallelSTL_VAR)
    set(${ParallelSTL_VAR} ${${VAR}})
endforeach()

