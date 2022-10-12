# ######################################################################################################################
# Copyright (C) 2019-2021 Advanced Micro Devices, Inc.
# ######################################################################################################################

set(ROCM_WARN_TOOLCHAIN_VAR
    ON
    CACHE BOOL "")
set(ROCM_ERROR_TOOLCHAIN_VAR
    OFF
    CACHE BOOL "")
# environment variable control of options. Note prefix is ROCMCHECKS
if(DEFINED ENV{ROCMCHECKS_WARN_TOOLCHAIN_VAR})
    set(ROCM_WARN_TOOLCHAIN_VAR $ENV{ROCMCHECKS_WARN_TOOLCHAIN_VAR})
endif()
if(DEFINED ENV{ROCMCHECKS_ERROR_TOOLCHAIN_VAR})
    set(ROCM_ERROR_TOOLCHAIN_VAR $ENV{ROCMCHECKS_ERROR_TOOLCHAIN_VAR})
endif()

function(rocm_check_toolchain_var var access value list_file)
    set(message_type STATUS)
    if(ROCM_ERROR_TOOLCHAIN_VAR)
        set(message_type SEND_ERROR)
        set(message_title " ROCMChecks ERROR ")
    elseif(ROCM_WARN_TOOLCHAIN_VAR)
        set(message_type WARNING)
        set(message_title "ROCMChecks WARNING")
    endif()
    if(access STREQUAL "MODIFIED_ACCESS")
        set(cmake_module Off)
        get_filename_component(base "${list_file}" DIRECTORY)
        # Skip warning in cmake's built-in modules
        if("${base}" STREQUAL "${CMAKE_ROOT}/Modules")
            set(cmake_module On)
        elseif("${base}" MATCHES ".*/CMakeFiles/${CMAKE_VERSION}$")
            set(cmake_module On)
        endif()
        if(NOT cmake_module)
            message( "
*******************************************************************************
*------------------------------- ${message_title} --------------------------*
  Options and properties should be set on a cmake target where possible. The
  variable '${var}' may be set by the cmake toolchain, either by
  calling 'cmake -D${var}=\"${value}\"'
  or set in a toolchain file and added with
  'cmake -DCMAKE_TOOLCHAIN_FILE=<toolchain-file>'. ROCMChecks now calling:")
            message(${message_type} "'${var}' is set at ${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt:<line#> shown below:")
            message( "*-----------------------------------------------------------------------------*
*******************************************************************************
")
        endif()
    endif()
endfunction()
if(UNIX AND (ROCM_WARN_TOOLCHAIN_VAR OR ROCM_ERROR_TOOLCHAIN_VAR))
    foreach(LANG C CXX Fortran)
        variable_watch(CMAKE_${LANG}_COMPILER rocm_check_toolchain_var)
        variable_watch(CMAKE_${LANG}_FLAGS rocm_check_toolchain_var)
        variable_watch(CMAKE_${LANG}_LINK_EXECUTABLE rocm_check_toolchain_var)
        variable_watch(CMAKE_${LANG}_SIZEOF_DATA_PTR rocm_check_toolchain_var)
        variable_watch(CMAKE_${LANG}_STANDARD_INCLUDE_DIRECTORIES rocm_check_toolchain_var)
        variable_watch(CMAKE_${LANG}_STANDARD_LIBRARIES rocm_check_toolchain_var)
    endforeach()
    variable_watch(CMAKE_EXE_LINKER_FLAGS rocm_check_toolchain_var)
    variable_watch(CMAKE_MODULE_LINKER_FLAGS rocm_check_toolchain_var)
    variable_watch(CMAKE_SHARED_LINKER_FLAGS rocm_check_toolchain_var)
    variable_watch(CMAKE_STATIC_LINKER_FLAGS rocm_check_toolchain_var)
endif()
