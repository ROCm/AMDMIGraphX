# ########################################################################
# Copyright 2016-2021 Advanced Micro Devices, Inc.
# ########################################################################

include(ROCMInstallSymlinks)
include(ROCMUtilities)

macro(rocm_package_setup_client_component COMPONENT_NAME)
    set(options)
    set(oneValueArgs PACKAGE_NAME LIBRARY_NAME)
    set(multiValueArgs DEPENDS)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(BUILD_SHARED_LIBS)
        if(DEFINED PARSE_DEPENDS)
            cmake_parse_arguments(DEPENDS "" "" "COMMON;RPM;DEB;COMPONENT" ${PARSE_DEPENDS})
            set(_DEPENDS_ARG
                DEPENDS
                    COMMON "${DEPENDS_COMMON}"
                    RPM "${DEPENDS_RPM}"
                    DEB "${DEPENDS_DEB}"
                    COMPONENT "${DEPENDS_COMPONENT}" runtime
            )
        else()
            set(_DEPENDS_ARG DEPENDS COMPONENT runtime)
        endif()
    elseif(DEFINED PARSE_DEPENDS)
        set(_DEPENDS_ARG DEPENDS "${PARSE_DEPENDS}")
    endif()

    if(DEFINED PARSE_PACKAGE_NAME)
        set(_PACKAGE_NAME_ARG "PACKAGE_NAME;${PARSE_PACKAGE_NAME}")
    endif()

    if(DEFINED LIBRARY_NAME)
        set(_LIBRARY_NAME_ARG "LIBRARY_NAME;${PARSE_LIBRARY_NAME}")
    endif()

    rocm_package_setup_component(
        ${COMPONENT_NAME}
        ${_PACKAGE_NAME_ARG}
        ${_LIBRARY_NAME_ARG}
        PARENT clients
        ${_DEPENDS_ARG}
    )
endmacro()
