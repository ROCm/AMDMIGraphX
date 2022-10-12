# ######################################################################################################################
# Copyright (C) 2017 Advanced Micro Devices, Inc.
# ######################################################################################################################

set(ROCM_ENABLE_GH_ANNOTATIONS
    Off
    CACHE BOOL "")

if(NOT TARGET analyze)
    add_custom_target(analyze)
endif()

function(rocm_mark_as_analyzer)
    add_dependencies(analyze ${ARGN})
endfunction()
