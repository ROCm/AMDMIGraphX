
function(register_op TARGET_NAME)
    set(options)
    set(oneValueArgs HEADER)
    set(multiValueArgs OPERATORS INCLUDES)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    string(MAKE_C_IDENTIFIER "${PARSE_HEADER}" BASE_NAME)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ops)
    set(FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/ops/${BASE_NAME}.cpp)
    file(WRITE "${FILE_NAME}" "")
    foreach(INCLUDE ${PARSE_INCLUDES})
        file(APPEND "${FILE_NAME}" "
#include <${INCLUDE}>
")
    endforeach()
    file(APPEND "${FILE_NAME}" "
#include <migraphx/register_op.hpp>
#include <${PARSE_HEADER}>
")


        file(APPEND "${FILE_NAME}" "
namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
")
    foreach(OPERATOR ${PARSE_OPERATORS})
        file(APPEND "${FILE_NAME}" "
MIGRAPHX_REGISTER_OP(${OPERATOR})
")
    endforeach()
    file(APPEND "${FILE_NAME}" "
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
")
    target_sources(${TARGET_NAME} PRIVATE ${FILE_NAME})
endfunction()
