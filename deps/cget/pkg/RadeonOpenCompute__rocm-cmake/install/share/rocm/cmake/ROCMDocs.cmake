# ######################################################################################################################
# Copyright (C) 2021 Advanced Micro Devices, Inc.
# ######################################################################################################################

if(NOT TARGET doc)
    add_custom_target(doc)
endif()

function(rocm_mark_as_doc)
    add_dependencies(doc ${ARGN})
endfunction()

function(rocm_clean_doc_output DIR)
    set_property(
        DIRECTORY
        APPEND
        PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${DIR})
endfunction()
