# ######################################################################################################################
# Copyright (C) 2017 Advanced Micro Devices, Inc.
# ######################################################################################################################

include(CMakePackageConfigHelpers)

function(rocm_configure_package_config_file INPUT_FILE OUTPUT_FILE)
    set(options)
    set(oneValueArgs INSTALL_DESTINATION PREFIX)
    set(multiValueArgs PATH_VARS)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(PARSE_UNPARSED_ARGUMENTS)
        message(
            FATAL_ERROR
                "Unknown keywords given to rocm_configure_package_config_file(): \"${PARSE_UNPARSED_ARGUMENTS}\"")
    endif()

    if(NOT PARSE_INSTALL_DESTINATION)
        message(FATAL_ERROR "INSTALL_DESTINATION is required for rocm_configure_package_config_file()")
    endif()

    if(IS_ABSOLUTE "${CMAKE_INSTALL_PREFIX}")
        set(INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
    else()
        get_filename_component(INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" ABSOLUTE)
    endif()

    if(IS_ABSOLUTE "${PARSE_INSTALL_DESTINATION}")
        set(ABSOLUTE_INSTALL_DIR "${PARSE_INSTALL_DESTINATION}")
    else()
        set(ABSOLUTE_INSTALL_DIR "${INSTALL_PREFIX}/${PARSE_INSTALL_DESTINATION}")
    endif()

    file(RELATIVE_PATH PACKAGE_RELATIVE_PATH "${ABSOLUTE_INSTALL_DIR}" "${INSTALL_PREFIX}")
    file(RELATIVE_PATH PACKAGE_INSTALL_RELATIVE_DIR "${INSTALL_PREFIX}" "${ABSOLUTE_INSTALL_DIR}")

    set(CHECK_PREFIX)
    if(PARSE_PREFIX)
        # On windows there is no symlinks
        if(WIN32)
            set(CHECK_PREFIX
                "
if(NOT \"\${PACKAGE_PREFIX_DIR}/${PACKAGE_INSTALL_RELATIVE_DIR}\" EQUAL \"\${CMAKE_CURRENT_LIST_DIR}\")
    set(PACKAGE_PREFIX_DIR ${INSTALL_PREFIX})
endif()
")
        endif()
    endif()

    foreach(_var ${PARSE_PATH_VARS})
        if(NOT DEFINED ${_var})
            message(FATAL_ERROR "Undefined path variable: ${_var}")
        endif()
        if(IS_ABSOLUTE "${${_var}}")
            string(REPLACE "${INSTALL_PREFIX}" "\${PACKAGE_PREFIX_DIR}" PACKAGE_${_var} "${${_var}}")
        else()
            set(PACKAGE_${_var} "\${PACKAGE_PREFIX_DIR}/${${_var}}")
        endif()
    endforeach()

    get_filename_component(INPUT_NAME "${INPUT_FILE}" NAME)

    set(PACKAGE_INIT
        "
####################################################################################
# Auto generated @PACKAGE_INIT@ by rocm_configure_package_config_file()
# from ${INPUT_NAME}
# Any changes to this file will be overwritten by the next CMake run
####################################################################################

get_filename_component(_ROCM_CMAKE_CURRENT_LIST_FILE_REAL \"\${CMAKE_CURRENT_LIST_FILE}\" REALPATH)
get_filename_component(_ROCM_CMAKE_CURRENT_LIST_DIR_REAL \"\${_ROCM_CMAKE_CURRENT_LIST_FILE_REAL}\" DIRECTORY)
get_filename_component(PACKAGE_PREFIX_DIR \"\${_ROCM_CMAKE_CURRENT_LIST_DIR_REAL}/${PACKAGE_RELATIVE_PATH}\" ABSOLUTE)

${CHECK_PREFIX}

macro(set_and_check _var _file)
    set(\${_var} \"\${_file}\")
    if(NOT EXISTS \"\${_file}\")
        message(FATAL_ERROR \"File or directory \${_file} referenced by variable \${_var} does not exist !\")
    endif()
endmacro()

include(CMakeFindDependencyMacro OPTIONAL RESULT_VARIABLE _ROCMCMakeFindDependencyMacro_FOUND)
if (NOT _ROCMCMakeFindDependencyMacro_FOUND)
    macro(find_dependency dep)
        if (NOT \${dep}_FOUND)
            set(rocm_fd_version)
            if (\${ARGC} GREATER 1)
                set(rocm_fd_version \${ARGV1})
            endif()
            set(rocm_fd_exact_arg)
            if(\${CMAKE_FIND_PACKAGE_NAME}_FIND_VERSION_EXACT)
                set(rocm_fd_exact_arg EXACT)
            endif()
            set(rocm_fd_quiet_arg)
            if(\${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
                set(rocm_fd_quiet_arg QUIET)
            endif()
            set(rocm_fd_required_arg)
            if(\${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED)
                set(rocm_fd_required_arg REQUIRED)
            endif()
            find_package(\${dep} \${rocm_fd_version}
                \${rocm_fd_exact_arg}
                \${rocm_fd_quiet_arg}
                \${rocm_fd_required_arg}
            )
            string(TOUPPER \${dep} cmake_dep_upper)
            if (NOT \${dep}_FOUND AND NOT \${cmake_dep_upper}_FOUND)
                set(\${CMAKE_FIND_PACKAGE_NAME}_NOT_FOUND_MESSAGE
                    \"\${CMAKE_FIND_PACKAGE_NAME} could not be found because dependency \${dep} could not be found.\")
                set(\${CMAKE_FIND_PACKAGE_NAME}_FOUND False)
                return()
            endif()
            set(rocm_fd_version)
            set(rocm_fd_required_arg)
            set(rocm_fd_quiet_arg)
            set(rocm_fd_exact_arg)
        endif()
    endmacro()
endif()

macro(check_required_components _NAME)
    foreach(comp \${\${_NAME}_FIND_COMPONENTS})
        if(NOT \${_NAME}_\${comp}_FOUND)
            if(\${_NAME}_FIND_REQUIRED_\${comp})
                set(\${_NAME}_FOUND FALSE)
            endif()
        endif()
    endforeach()
endmacro()

####################################################################################

    ")

    configure_file("${INPUT_FILE}" "${OUTPUT_FILE}" @ONLY)

endfunction()
