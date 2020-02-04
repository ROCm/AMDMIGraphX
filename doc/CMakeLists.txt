
include(DoxygenDoc)

set(DOXYGEN_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/doxygen/)
add_doxygen_doc(
    OUTPUT_DIRECTORY ${DOXYGEN_OUTPUT}
    INPUT 
        ${PROJECT_SOURCE_DIR}/src
    INCLUDE_PATH
        ${PROJECT_SOURCE_DIR}/src/include
        ${PROJECT_SOURCE_DIR}/src/targets/cpu/include
        ${PROJECT_SOURCE_DIR}/src/targets/gpu/include
    STRIP_FROM_INC_PATH
        ${PROJECT_SOURCE_DIR}/src/include
        ${PROJECT_SOURCE_DIR}/src/targets/cpu/include
        ${PROJECT_SOURCE_DIR}/src/targets/gpu/include
    SEARCH_INCLUDES YES
    MACRO_EXPANSION YES
    RECURSIVE YES
    GENERATE_XML YES
    GENERATE_LATEX YES
    USE_PDFLATEX YES
    CALL_GRAPH YES
    CALLER_GRAPH YES
    BUILTIN_STL_SUPPORT YES
    PROJECT_NAME MIGraphX
    SORT_MEMBERS_CTORS_1ST YES
    SOURCE_BROWSER YES
    GENERATE_TREEVIEW YES
    REFERENCED_BY_RELATION YES
    REFERENCES_RELATION YES
    REFERENCES_LINK_SOURCE YES
    EXTRACT_ALL YES
    ENUM_VALUES_PER_LINE 1
    FULL_PATH_NAMES YES
    PREDEFINED DOXYGEN
)
add_custom_target(remove_inline_ns 
    sed -i "s/MIGRAPHX_INLINE_NS:://g" *.xml 
    WORKING_DIRECTORY ${DOXYGEN_OUTPUT}/xml)
add_dependencies(remove_inline_ns doxygen)

include(SphinxDoc)
add_sphinx_doc(src 
    BUILDER html 
    OUTPUT_DIR html 
    VARS 
        breathe_projects.proj=${DOXYGEN_OUTPUT}/xml
        breathe_default_project=proj
    DEPENDS doxygen remove_inline_ns
)

find_package(LATEX)
if(LATEX_FOUND)
    add_sphinx_doc(src 
        BUILDER latex
        OUTPUT_DIR pdf
        VARS 
            breathe_projects.proj=${DOXYGEN_OUTPUT}/xml
            breathe_default_project=proj
        DEPENDS doxygen
    )
else()
    message("Latex builder not found. Latex builder is required only for building the PDF documentation for MIGraph and is not necessary for building the library, or any other components. To build PDF documentation run make in ${CMAKE_CURRENT_SOURCE_DIR}/pdf, once a latex builder is installed.")
endif()

