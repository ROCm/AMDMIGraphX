if(COMMAND find_python)
    return()
endif()

set(PYBIND11_NOPYTHON On)
find_package(pybind11 REQUIRED)
macro(find_python version)
    find_program(PYTHON_CONFIG_${version} python${version}-config)
    if(EXISTS ${PYTHON_CONFIG_${version}})
        execute_process(COMMAND ${PYTHON_CONFIG_${version}} --includes OUTPUT_VARIABLE _python_include_args)
        separate_arguments(_python_includes UNIX_COMMAND "${_python_include_args}")
        string(REPLACE "-I" "" _python_includes "${_python_includes}")
        add_library(python${version}::headers INTERFACE IMPORTED GLOBAL)
        set_target_properties(python${version}::headers PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${_python_includes}"
        )
        execute_process(COMMAND ${PYTHON_CONFIG_${version}} --prefix OUTPUT_VARIABLE _python_prefix)
        string(STRIP "${_python_prefix}" _python_prefix)
        set(PYTHON_${version}_EXECUTABLE "${_python_prefix}/bin/python${version}" CACHE PATH "")
    endif()
endmacro()
function(py_extension name version)
    set(_python_module_extension)
    execute_process(COMMAND ${PYTHON_CONFIG_${version}} --extension-suffix OUTPUT_VARIABLE _python_module_extension)
    string(STRIP "${_python_module_extension}" _python_module_extension)
    set_target_properties(${name} PROPERTIES PREFIX "" SUFFIX "${_python_module_extension}")
endfunction()
function(py_add_module NAME)
    set(options)
    set(oneValueArgs PYTHON_VERSION PYTHON_MODULE)
    set(multiValueArgs)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    set(PYTHON_VERSION ${PARSE_PYTHON_VERSION})

    add_library(${NAME} MODULE ${PARSE_UNPARSED_ARGUMENTS})
    pybind11_strip(${NAME})
    py_extension(${NAME} ${PYTHON_VERSION})
    target_link_libraries(${NAME} PRIVATE pybind11::module pybind11::lto python${PYTHON_VERSION}::headers)
    set_target_properties(${NAME} PROPERTIES 
        OUTPUT_NAME ${PARSE_PYTHON_MODULE}
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
    )

endfunction()
set(PYTHON_SEARCH_VERSIONS 2.7 3.5 3.6 3.7 3.8 3.9)

set(_PYTHON_VERSIONS)
foreach(PYTHON_VERSION ${PYTHON_SEARCH_VERSIONS})
    find_python(${PYTHON_VERSION})
    if(TARGET python${PYTHON_VERSION}::headers)
        message(STATUS "Python ${PYTHON_VERSION} found.")
        list(APPEND _PYTHON_VERSIONS ${PYTHON_VERSION})
    else()
        message(STATUS "Python ${PYTHON_VERSION} not found.")
    endif()
endforeach()

# Make the variable global
set(PYTHON_VERSIONS "${_PYTHON_VERSIONS}" CACHE INTERNAL "" FORCE)
