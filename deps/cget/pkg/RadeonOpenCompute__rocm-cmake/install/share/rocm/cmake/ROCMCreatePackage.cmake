# ######################################################################################################################
# Copyright (C) 2017-2019 Advanced Micro Devices, Inc.
# ######################################################################################################################

cmake_policy(SET CMP0057 NEW)

set(ROCM_DISABLE_LDCONFIG
    OFF
    CACHE BOOL "")

get_filename_component(REAL_ROCM "${CMAKE_INSTALL_PREFIX}" REALPATH)
get_filename_component(REAL_ROCM_DIR "${ROCM_DIR}" REALPATH)
if(REAL_ROCM MATCHES "rocm-([0-9]+(\\.[0-9]+)+)")
    set(ROCM_PLATFORM_VERSION "${CMAKE_MATCH_1}" CACHE STRING "The version of the ROCm platform.")
elseif(REAL_ROCM_DIR MATCHES "rocm-([0-9]+(\\.[0-9]+)+)")
    set(ROCM_PLATFORM_VERSION "${CMAKE_MATCH_1}" CACHE STRING "The version of the ROCm platform.")
endif()
if(DEFINED ROCM_PLATFORM_VERSION AND ROCM_PLATFORM_VERSION VERSION_LESS 4.5.0)
    set(ROCM_DEP_ROCMCORE FALSE CACHE BOOL "Add dependency on rocm-core package")
else()
    set(ROCM_DEP_ROCMCORE TRUE CACHE BOOL "Add dependency on rocm-core package")
endif()

include(CMakeParseArguments)
include(GNUInstallDirs)
include(ROCMSetupVersion)
include(ROCMUtilities)

find_program(MAKE_NSIS_EXE makensis)
find_program(RPMBUILD_EXE rpmbuild)
find_program(DPKG_EXE dpkg)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME runtime)
set(ROCM_PACKAGE_CREATED FALSE CACHE INTERNAL "Track whether rocm_create_package has been called.")

function(rocm_package_add_rpm_dependencies)
    set(options QUIET)
    set(oneValueArgs COMPONENT)
    set(multiValueArgs DEPENDS SHARED_DEPENDS STATIC_DEPENDS)
    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(${ROCM_PACKAGE_CREATED} AND NOT PARSE_QUIET)
        message(AUTHOR_WARNING "rocm_package_add_rpm_dependencies called after rocm_create_package!")
    endif()

    if(DEFINED PARSE_COMPONENT)
        string(TOUPPER "${PARSE_COMPONENT}" COMPONENT_VAR)
        set(REQ_VAR "CPACK_RPM_${COMPONENT_VAR}_PACKAGE_REQUIRES")
    else()
        set(REQ_VAR "CPACK_RPM_PACKAGE_REQUIRES")
    endif()

    set(CURRENT_DEPENDS "${${REQ_VAR}}")

    if (DEFINED PARSE_DEPENDS)
        rocm_join_if_set(", " CURRENT_DEPENDS ${PARSE_DEPENDS})
    endif()

    if(DEFINED PARSE_SHARED_DEPENDS AND BUILD_SHARED_LIBS)
        rocm_join_if_set(", " CURRENT_DEPENDS ${PARSE_SHARED_DEPENDS})
    endif()

    if(DEFINED PARSE_STATIC_DEPENDS AND NOT BUILD_SHARED_LIBS)
        rocm_join_if_set(", " CURRENT_DEPENDS ${PARSE_STATIC_DEPENDS})
    endif()
    set(${REQ_VAR} "${CURRENT_DEPENDS}" PARENT_SCOPE)
endfunction()

function(rocm_package_add_deb_dependencies)
    set(options QUIET)
    set(oneValueArgs COMPONENT)
    set(multiValueArgs DEPENDS SHARED_DEPENDS STATIC_DEPENDS)
    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(${ROCM_PACKAGE_CREATED} AND NOT PARSE_QUIET)
        message(AUTHOR_WARNING "rocm_package_add_deb_dependencies called after rocm_create_package!")
    endif()

    if(DEFINED PARSE_COMPONENT)
        string(TOUPPER "CPACK_DEBIAN_${PARSE_COMPONENT}_PACKAGE_DEPENDS" REQ_VAR)
    else()
        set(REQ_VAR "CPACK_DEBIAN_PACKAGE_DEPENDS")
    endif()

    set(NEW_DEPENDS "")
    if(DEFINED PARSE_DEPENDS)
        list(APPEND NEW_DEPENDS ${PARSE_DEPENDS})
    endif()

    if(DEFINED PARSE_SHARED_DEPENDS AND BUILD_SHARED_LIBS)
        list(APPEND NEW_DEPENDS "${PARSE_SHARED_DEPENDS}")
    endif()

    if(DEFINED PARSE_STATIC_DEPENDS AND NOT BUILD_SHARED_LIBS)
        list(APPEND NEW_DEPENDS "${PARSE_STATIC_DEPENDS}")
    endif()

    set(CURRENT_DEPENDS "${${REQ_VAR}}")
    foreach(DEP IN LISTS NEW_DEPENDS)
        string(REGEX REPLACE
            "^([a-zA-Z0-9][-a-zA-Z0-9+.]+)[ \t\n]+([<>]?=)[ \t\n]+(.*)$"
            "\\1 (\\2 \\3)" DEP_EQ "${DEP}")
        string(REGEX REPLACE
            "^([a-zA-Z0-9][-a-zA-Z0-9+.]+)[ \t\n]+([<>])[ \t\n]+(.*)$"
            "\\1 (\\2\\2 \\3)" DEP_ALL "${DEP_EQ}")
        rocm_join_if_set(", " CURRENT_DEPENDS "${DEP_ALL}")
    endforeach()
    set(${REQ_VAR} "${CURRENT_DEPENDS}" PARENT_SCOPE)
endfunction()

macro(rocm_package_add_dependencies)
    set(_list_var "${ARGN}")
    if ("QUIET" IN_LIST _list_var)
        rocm_package_add_deb_dependencies(${ARGN})
        rocm_package_add_rpm_dependencies(${ARGN})
    else()
        if(${ROCM_PACKAGE_CREATED})
            message(AUTHOR_WARNING "rocm_package_add_dependencies called after rocm_create_package!")
        endif()
        rocm_package_add_deb_dependencies(QUIET ${ARGN})
        rocm_package_add_rpm_dependencies(QUIET ${ARGN})
    endif()
endmacro()

macro(rocm_package_add_rocm_core_dependency)
    # Optionally add depenency on rocm-core
    # This mainly empty package exists to allow all of rocm
    # to be removed in one step by removing rocm-core
    if(ROCM_DEP_ROCMCORE)
        rocm_join_if_set(", " CPACK_DEBIAN_PACKAGE_DEPENDS "rocm-core")
        rocm_join_if_set(", " CPACK_RPM_PACKAGE_REQUIRES "rocm-core")
    endif()
endmacro()

function(rocm_parse_python_syspath DIR_PATH PKG_NAME)
    set(PYTHON_SITE_PACKAGES
        "/usr/lib/python3/dist-packages;/usr/lib/python2.7/dist-packages"
        CACHE STRING "The site packages used for packaging")
    #Group the statements to function
    file(APPEND ${PROJECT_BINARY_DIR}/debian/postinst
        "
        set_libdir(){
    ")
    file(APPEND ${PROJECT_BINARY_DIR}/debian/prerm
        "
        rm_libdir(){
    ")
    foreach(PYTHON_SITE ${PYTHON_SITE_PACKAGES})
        file(
            APPEND ${PROJECT_BINARY_DIR}/debian/postinst
            "
            mkdir -p ${PYTHON_SITE}
            echo \"${DIR_PATH}\" > ${PYTHON_SITE}/${PKG_NAME}.pth
        ")

        file(
            APPEND ${PROJECT_BINARY_DIR}/debian/prerm
            "
            rm ${PYTHON_SITE}/${PKG_NAME}.pth
        ")
    endforeach()
    #end function and invoke the function
    file(APPEND ${PROJECT_BINARY_DIR}/debian/postinst
        "
        }
        set_libdir
    ")
    file(APPEND ${PROJECT_BINARY_DIR}/debian/prerm
        "
        }
        rm_libdir
    ")
endfunction()

macro(rocm_set_cpack_gen)
    # If CPACK_GENERATOR value has been given, then just use it
    if(NOT CPACK_GENERATOR)
        # If there is a PKGTYPE, use that as the desired type
        if(DEFINED ENV{ROCM_PKGTYPE})
            set(CPACK_GENERATOR "" ) # Create the variable if needed
            string(TOUPPER $ENV{ROCM_PKGTYPE} CPACK_GENERATOR) # PKGTYPE is typically lower case
        else()
            # Otherwise see what we can find
            message(INFO "rocm_set_cpack_gen didn't find ROCM_PKGTYPE in environment")
            set(CPACK_GENERATOR "TGZ;ZIP")
            if(EXISTS ${MAKE_NSIS_EXE})
                list(APPEND CPACK_GENERATOR "NSIS")
            endif()

            if(EXISTS ${RPMBUILD_EXE})
                list(APPEND CPACK_GENERATOR "RPM")
            endif()

            if(EXISTS ${DPKG_EXE})
                list(APPEND CPACK_GENERATOR "DEB")
            endif()
        endif()
    endif()
    # Set up some additional variables depending on which generator we are going to use
    if (CPACK_GENERATOR MATCHES ".*RPM.*")
        if(PARSE_COMPONENTS)
            set(CPACK_RPM_COMPONENT_INSTALL ON)
        endif()
    endif()
    if (CPACK_GENERATOR MATCHES ".*DEB.*")
        if(EXISTS ${DPKG_EXE})
            if(PARSE_COMPONENTS)
                set(CPACK_DEB_COMPONENT_INSTALL ON)
                execute_process(
                    COMMAND dpkg --print-architecture
                    RESULT_VARIABLE PROC_RESULT
                    OUTPUT_VARIABLE COMMAND_OUTPUT
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
                if(PROC_RESULT EQUAL "0" AND NOT COMMAND_OUTPUT STREQUAL "")
                    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "${COMMAND_OUTPUT}")
                endif()
            endif()
        endif()
    endif()
endmacro()

macro(rocm_create_package)
    set(options LDCONFIG PTH HEADER_ONLY)
    set(oneValueArgs NAME DESCRIPTION SECTION MAINTAINER LDCONFIG_DIR PREFIX)
    set(multiValueArgs DEPENDS COMPONENTS)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    string(TOLOWER ${PARSE_NAME} _rocm_cpack_package_name)
    set(CPACK_PACKAGE_NAME ${_rocm_cpack_package_name})
    set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${PARSE_DESCRIPTION})
    set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH})
    set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
    set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
    set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
    if(NOT CMAKE_HOST_WIN32)
        set(CPACK_SET_DESTDIR
            ON
            CACHE BOOL "Boolean toggle to make CPack use DESTDIR mechanism when packaging")
        if(${CPACK_SET_DESTDIR})
            set(CPACK_PACKAGING_INSTALL_PREFIX "")
        endif()
    endif()

    rocm_get_patch_version(ROCM_VERSION_NUM)
    if(ROCM_VERSION_NUM)
        set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}.${ROCM_VERSION_NUM}")
    endif()

    set(CPACK_DEBIAN_PACKAGE_MAINTAINER ${PARSE_MAINTAINER})
    set(CPACK_DEBIAN_PACKAGE_SECTION "devel")
    set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")

    set(CPACK_NSIS_MODIFY_PATH On)
    set(CPACK_NSIS_PACKAGE_NAME ${PARSE_NAME})

    set(CPACK_RPM_PACKAGE_RELOCATABLE Off)
    set(CPACK_RPM_PACKAGE_AUTOREQPROV
        Off
        CACHE BOOL "turns off rpm autoreqprov field; packages explicity list dependencies")
    set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")

    if(DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
        set(DEBIAN_VERSION $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
    elseif(PROJECT_VERSION_TWEAK)
        # Sanitize tweak version for debian
        string(REGEX REPLACE "[^A-Za-z0-9.+~]" "~" DEBIAN_VERSION ${PROJECT_VERSION_TWEAK})
    endif()

    if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
        set(RPM_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE})
    elseif(PROJECT_VERSION_TWEAK)
        # Sanitize tweak version for rpm
        string(REPLACE "-" "_" RPM_RELEASE ${PROJECT_VERSION_TWEAK})
    endif()

    if (ROCM_USE_DEV_COMPONENT)
        list(APPEND PARSE_COMPONENTS devel)
        set(CPACK_DEBIAN_DEVEL_PACKAGE_NAME "${CPACK_PACKAGE_NAME}-dev")
        rocm_join_if_set(", " CPACK_DEBIAN_RUNTIME_PACKAGE_RECOMMENDS
            "${CPACK_PACKAGE_NAME}-dev (>=${CPACK_PACKAGE_VERSION})")

        rocm_find_program_version(rpmbuild GREATER_EQUAL 4.12.0)
        if(rpmbuild_VERSION_OK)
            rocm_join_if_set(", " CPACK_RPM_RUNTIME_PACKAGE_SUGGESTS
                "${CPACK_PACKAGE_NAME}-devel >= ${CPACK_PACKAGE_VERSION}"
            )
        endif()
        if(PARSE_HEADER_ONLY)
            set(CPACK_DEBIAN_DEVEL_PACKAGE_PROVIDES "${CPACK_PACKAGE_NAME} (= ${CPACK_PACKAGE_VERSION})")
            set(CPACK_RPM_DEVEL_PACKAGE_PROVIDES "${CPACK_PACKAGE_NAME} = ${CPACK_PACKAGE_VERSION}")
        else()
            rocm_package_add_dependencies(COMPONENT devel DEPENDS "${CPACK_PACKAGE_NAME} >= ${CPACK_PACKAGE_VERSION}")
        endif()
    endif()

    if(ROCM_PACKAGE_COMPONENTS)
        list(APPEND PARSE_COMPONENTS ${ROCM_PACKAGE_COMPONENTS})
    endif()

    rocm_set_cpack_gen()      # Set CPACK_GENERATOR if not already set
    if(CPACK_GENERATOR MATCHES ".*RPM.*")
        # '%{?dist}' breaks manual builds on debian systems due to empty Provides
        execute_process(
            COMMAND rpm --eval %{?dist}
            RESULT_VARIABLE PROC_RESULT
            OUTPUT_VARIABLE EVAL_RESULT
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(PROC_RESULT EQUAL "0" AND NOT EVAL_RESULT STREQUAL "")
            string(APPEND RPM_RELEASE "%{?dist}")
        endif()
    endif()
    set(CPACK_DEBIAN_PACKAGE_RELEASE ${DEBIAN_VERSION})
    set(CPACK_RPM_PACKAGE_RELEASE ${RPM_RELEASE})

    if(PARSE_DEPENDS)
        rocm_package_add_dependencies(DEPENDS ${PARSE_DEPENDS})
    endif()

    rocm_package_add_rocm_core_dependency()

    set(LIB_DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
    if(PARSE_PREFIX)
        set(LIB_DIR ${CMAKE_INSTALL_PREFIX}/${PARSE_PREFIX}/${CMAKE_INSTALL_LIBDIR})
    endif()

    file(WRITE ${PROJECT_BINARY_DIR}/debian/postinst "")
    file(WRITE ${PROJECT_BINARY_DIR}/debian/prerm "")
    set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${PROJECT_BINARY_DIR}/debian/postinst;${PROJECT_BINARY_DIR}/debian/prerm")
    set(CPACK_RPM_POST_INSTALL_SCRIPT_FILE "${PROJECT_BINARY_DIR}/debian/postinst")
    set(CPACK_RPM_PRE_UNINSTALL_SCRIPT_FILE "${PROJECT_BINARY_DIR}/debian/prerm")

    if(PARSE_LDCONFIG AND NOT ${ROCM_DISABLE_LDCONFIG})
        set(LDCONFIG_DIR ${LIB_DIR})
        if(PARSE_LDCONFIG_DIR)
            set(LDCONFIG_DIR ${PARSE_LDCONFIG_DIR})
        endif()
        file(
            APPEND ${PROJECT_BINARY_DIR}/debian/postinst
            "
            echo \"${LDCONFIG_DIR}\" > /etc/ld.so.conf.d/${PARSE_NAME}.conf
            ldconfig
        ")

        file(
            APPEND ${PROJECT_BINARY_DIR}/debian/prerm
            "
            rm /etc/ld.so.conf.d/${PARSE_NAME}.conf
            ldconfig
        ")
    endif()

    if(PARSE_PTH)
        rocm_parse_python_syspath(${LIB_DIR} ${PARSE_NAME})
    endif()
    rocm_setup_license(${PARSE_HEADER_ONLY})
    if(PARSE_COMPONENTS)
        rocm_set_comp_cpackvar(PARSE_HEADER_ONLY "${PARSE_COMPONENTS}")
    endif()
    include(CPack)
    set(ROCM_PACKAGE_CREATED TRUE CACHE INTERNAL "Track whether rocm_create_package has been called.")
endmacro()

macro(rocm_setup_license HEADER_ONLY)
    if(NOT CPACK_RESOURCE_FILE_LICENSE)
        file(GLOB _license_files LIST_DIRECTORIES FALSE "${CMAKE_SOURCE_DIR}/LICENSE*")
        set(_detected_license_files)
        foreach(_license_file IN LISTS _license_files)
            if(_license_file MATCHES "LICENSE(\\.(md|txt))?$")
                list(APPEND _detected_license_files "${_license_file}")
            endif()
        endforeach()
        list(LENGTH _detected_license_files _num_licenses)
        if(_num_licenses GREATER 1)
            message(AUTHOR_WARNING
                "rocm-cmake warning: Multiple license files found, "
                "please specify one using CPACK_RESOURCE_FILE_LICENSE."
            )
        elseif(_num_licenses EQUAL 0)
            message(AUTHOR_WARNING
                "rocm-cmake warning: Could not find a license file, "
                "please specify one using CPACK_RESOURCE_FILE_LICENSE."
            )
        else()
            list(GET _detected_license_files 0 CPACK_RESOURCE_FILE_LICENSE)
            message(STATUS "rocm-cmake: Set license file to ${CPACK_RESOURCE_FILE_LICENSE}.")
        endif()
    endif()

    if(CPACK_RESOURCE_FILE_LICENSE)
        if(ROCM_USE_DEV_COMPONENT AND ${HEADER_ONLY})
            install(
                FILES ${CPACK_RESOURCE_FILE_LICENSE}
                DESTINATION share/doc/${_rocm_cpack_package_name}
                COMPONENT devel
            )
        else()
            install(
                FILES ${CPACK_RESOURCE_FILE_LICENSE}
                DESTINATION share/doc/${_rocm_cpack_package_name}
            )
        endif()
    endif()
endmacro()

macro(rocm_set_comp_cpackvar HEADER_ONLY components)
    # Setting component specific variables
    set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)

    if(NOT ${HEADER_ONLY})
        set(CPACK_RPM_MAIN_COMPONENT "runtime")
        set(CPACK_RPM_RUNTIME_PACKAGE_NAME "${CPACK_PACKAGE_NAME}")
        list(APPEND CPACK_COMPONENTS_ALL runtime)
        set(CPACK_DEBIAN_RUNTIME_FILE_NAME
           "${CPACK_PACKAGE_NAME}_${CPACK_PACKAGE_VERSION}-${DEBIAN_VERSION}_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}.deb")
        set(CPACK_DEBIAN_RUNTIME_PACKAGE_NAME "${CPACK_PACKAGE_NAME}")
    endif()

    foreach(COMPONENT ${components})
        list(APPEND CPACK_COMPONENTS_ALL "${COMPONENT}")
        string(TOUPPER "${COMPONENT}" COMPONENT_UC)
        set(CPACK_RPM_${COMPONENT_UC}_FILE_NAME "RPM-DEFAULT")
        set(CPACK_DEBIAN_${COMPONENT_UC}_FILE_NAME "DEB-DEFAULT")
        if(NOT DEFINED CPACK_DEBIAN_${COMPONENT_UC}_PACKAGE_NAME
            OR CPACK_DEBIAN_${COMPONENT_UC}_PACKAGE_NAME STREQUAL "")
            if(NOT DEFINED CPACK_DEBIAN_PACKAGE_NAME OR CPACK_DEBIAN_PACKAGE_NAME STREQUAL "")
                set(CPACK_DEBIAN_PACKAGE_NAME "${CPACK_PACKAGE_NAME}")
            endif()
            string(TOLOWER "${CPACK_DEBIAN_PACKAGE_NAME}-${COMPONENT}" CPACK_DEBIAN_${COMPONENT_UC}_PACKAGE_NAME)
        else()
            string(REGEX REPLACE "<PACKAGE_NAME>" "${CPACK_PACKAGE_NAME}"
                CPACK_DEBIAN_${COMPONENT_UC}_PACKAGE_NAME "${CPACK_DEBIAN_${COMPONENT_UC}_PACKAGE_NAME}")
        endif()
        if(NOT DEFINED CPACK_RPM_${COMPONENT_UC}_PACKAGE_NAME
            OR CPACK_RPM_${COMPONENT_UC}_PACKAGE_NAME STREQUAL "")
            if(NOT DEFINED CPACK_RPM_PACKAGE_NAME OR CPACK_RPM_PACKAGE_NAME STREQUAL "")
                set(CPACK_RPM_PACKAGE_NAME "${CPACK_PACKAGE_NAME}")
            endif()
            string(TOLOWER "${CPACK_RPM_PACKAGE_NAME}-${COMPONENT}" CPACK_RPM_${COMPONENT_UC}_PACKAGE_NAME)
        else()
            string(REGEX REPLACE "<PACKAGE_NAME>" "${CPACK_PACKAGE_NAME}"
                CPACK_RPM_${COMPONENT_UC}_PACKAGE_NAME "${CPACK_RPM_${COMPONENT_UC}_PACKAGE_NAME}")
        endif()
    endforeach()
    if(ROCM_PACKAGE_COMPONENT_DEPENDENCIES)
        foreach(COMP_DEP IN LISTS ROCM_PACKAGE_COMPONENT_DEPENDENCIES)
            string(REGEX REPLACE "^(.*)->.*$" "\\1" _downstream "${COMP_DEP}")
            string(REGEX REPLACE "^.*->(.*)$" "\\1" _upstream "${COMP_DEP}")
            string(TOUPPER "${_upstream}" _upstream_uc)
            rocm_package_add_rpm_dependencies(COMPONENT "${_downstream}"
                DEPENDS "${CPACK_RPM_${_upstream_uc}_PACKAGE_NAME} >= ${CPACK_PACKAGE_VERSION}")
            rocm_package_add_deb_dependencies(COMPONENT "${_downstream}"
                DEPENDS "${CPACK_DEBIAN_${_upstream_uc}_PACKAGE_NAME} >= ${CPACK_PACKAGE_VERSION}")
        endforeach()
    endif()
endmacro()

macro(rocm_package_setup_component COMPONENT_NAME)
    set(options)
    set(oneValueArgs PACKAGE_NAME LIBRARY_NAME PARENT)
    set(multiValueArgs DEPENDS)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    list(APPEND ROCM_PACKAGE_COMPONENTS ${COMPONENT_NAME})

    if(NOT DEFINED PARSE_PACKAGE_NAME)
        string(TOLOWER "${COMPONENT_NAME}" PARSE_PACKAGE_NAME)
        if(NOT BUILD_SHARED_LIBS)
            set(PARSE_PACKAGE_NAME "${PARSE_PACKAGE_NAME}-static")
        endif()
    endif()

    if(NOT DEFINED PARSE_LIBRARY_NAME)
        set(PARSE_LIBRARY_NAME "<PACKAGE_NAME>")
    endif()

    string(TOUPPER "${COMPONENT_NAME}" COMPONENT_GNAME)

    set(CPACK_DEBIAN_${COMPONENT_GNAME}_PACKAGE_NAME "${PARSE_LIBRARY_NAME}-${PARSE_PACKAGE_NAME}")
    set(CPACK_RPM_${COMPONENT_GNAME}_PACKAGE_NAME "${PARSE_LIBRARY_NAME}-${PARSE_PACKAGE_NAME}")

    if(DEFINED PARSE_PARENT)
        list(APPEND ROCM_PACKAGE_COMPONENT_DEPENDENCIES "${PARSE_PARENT}->${COMPONENT_NAME}")
    endif()

    if(DEFINED PARSE_DEPENDS)
        cmake_parse_arguments(DEPENDS "" "" "COMMON;DEB;RPM;COMPONENT" ${PARSE_DEPENDS})
        rocm_package_add_deb_dependencies(COMPONENT ${COMPONENT_NAME} DEPENDS ${DEPENDS_COMMON} ${DEPENDS_DEB})
        rocm_package_add_rpm_dependencies(COMPONENT ${COMPONENT_NAME} DEPENDS ${DEPENDS_COMMON} ${DEPENDS_RPM})
        foreach(DEP_COMP IN LISTS DEPENDS_COMPONENT)
            list(APPEND ROCM_PACKAGE_COMPONENT_DEPENDENCIES "${COMPONENT_NAME}->${DEP_COMP}")
        endforeach()
    endif()
endmacro()
