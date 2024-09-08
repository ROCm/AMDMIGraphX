
include(CheckCSourceCompiles)
include(CheckFunctionExists)
include(CheckLibraryExists)
include(CheckSymbolExists)
include(CheckIncludeFiles)
include(CheckStructHasMember)
include(CheckTypeSize)
include(CheckCSourceRuns)
include(TestBigEndian)

function(patch_file FILE MATCH REPLACEMENT)
    message(STATUS "Patch: ${FILE}")
    file(READ ${FILE} CONTENT)
    string(REPLACE 
        "${MATCH}"
        "${REPLACEMENT}" 
        OUTPUT_CONTENT "${CONTENT}")
    file(WRITE ${FILE} "${OUTPUT_CONTENT}")
endfunction()

macro(append_prefix PREFIX _LIST_)
    foreach(ELEM ${ARGN})
        list(APPEND ${_LIST_} ${PREFIX}${ELEM})
    endforeach()
endmacro()

function(read_lines FILE VAR)
    file(READ ${FILE} LINES)
    # Replace semicolons with "<semi>" to avoid CMake messing with them.
    string(REPLACE ";" "<semi>" LINES "${LINES}")
    # Split into lines keeping newlines to avoid foreach skipping empty ones.
    string(REGEX MATCHALL "[^\n]*\n" LINES "${LINES}")
    set(${VAR} ${LINES} PARENT_SCOPE)
endfunction()

function(ac_config_header INPUT OUTPUT)
    read_lines(${INPUT} CONFIG)
    list(LENGTH CONFIG length)
    math(EXPR length "${length} - 1")
    foreach (i RANGE ${length})
        list(GET CONFIG ${i} line)
        if (line MATCHES "^#( *)undef (.*)\n")
            set(space "${CMAKE_MATCH_1}")
            set(var ${CMAKE_MATCH_2})
            if (NOT DEFINED ${var} OR (var MATCHES ^HAVE AND NOT ${var}))
                set(line "/* #${space}undef ${var} */\n")
            else ()
                if (var MATCHES ^PACKAGE OR "${var}" STREQUAL "VERSION")
                    set(value \"${${var}}\")
                elseif(var MATCHES ^HAVE)
                    if(${var})
                        set(value 1)
                    else()
                        set(value 0)
                    endif()
                else()
                    set(value ${${var}})
                endif()
                set(line "#${space}define ${var} ${value}\n")
            endif ()
        endif ()
        string(REPLACE "<semi>" ";" line "${line}")
        set(CONFIG_OUT "${CONFIG_OUT}${line}")
    endforeach ()
    file(WRITE ${OUTPUT}
"/* config.h.  Generated from config.h.in by configure.  */
/* config.h.in.  Generated from configure.ac by autoheader.  */

${CONFIG_OUT}")
endfunction()

function(parse_makefile_var MAKEFILE VAR)
    file(STRINGS ${MAKEFILE} lines)
    foreach (line ${lines})
        string(REGEX MATCHALL "[^ \t]+" words ${line})
        list(LENGTH words words_len)
        if(words_len GREATER 2)
            list(GET words 0 var_name)
            list(GET words 1 assign)
            if(assign STREQUAL "=" AND var_name STREQUAL ${VAR})
                list(REMOVE_AT words 0 1)
                set(${VAR} ${words} PARENT_SCOPE)
            endif()
        endif()
    endforeach()
endfunction()

# set(_ac_config_vars)

# macro(set_config_var VAR)
#     set(${VAR} ${ARGN}) # Should we cache as internal?
#     list(APPEND _ac_config_vars ${VAR})
# endmacro()

macro(ac_check_headers)
    foreach (header ${ARGN})
        string(TOUPPER HAVE_${header} var)
        string(REGEX REPLACE "\\.|/" "_" var ${var})
        if(NOT DEFINED ${var})
            check_include_files(${header} ${var})
        endif()
    endforeach()
endmacro()

macro(ac_check_includes_list INCLUDES)
    set(${INCLUDES})
    foreach (header ${ARGN})
        string(TOUPPER HAVE_${header} var)
        string(REGEX REPLACE "\\.|/" "_" var ${var})
        if(NOT DEFINED ${var})
            check_include_files(${header} ${var})
        endif()
        if(${var})
            list(APPEND ${INCLUDES} ${header})
        endif()
    endforeach()
endmacro()

macro(ac_includes_default_list)
    if(NOT DEFINED AC_INCLUDES_DEFAULT)
        ac_check_includes_list(_AC_INCLUDES_DEFAULT_LIST sys/types.h sys/stat.h stdlib.h stddef.h stdlib.h memory.h string.h strings.h inttypes.h stdint.h unistd.h)
        set(AC_INCLUDES_DEFAULT "#include<stdio.h>\n")
        foreach(INCLUDE ${_AC_INCLUDES_DEFAULT_LIST})
            set(AC_INCLUDES_DEFAULT "${AC_INCLUDES_DEFAULT}\n#include <${INCLUDE}>\n")
        endforeach()
        ac_header_stdc()
    endif()
    if(${ARGC} GREATER 0)
        set(${INCLUDES} stdio.h ${_AC_INCLUDES_DEFAULT_LIST})
    endif()
endmacro()

macro(ac_check_funcs)
    foreach (fun ${ARGN})
        string(TOUPPER HAVE_${fun} var)
        string(REGEX REPLACE "\\.|/" "_" var ${var})
        if(NOT DEFINED ${var})
            check_function_exists(${fun} ${var})
        endif()
    endforeach()
endmacro()

macro(ac_check_sizeof TYPE)
    string(TOUPPER SIZEOF_${TYPE} var)
    string(REGEX REPLACE "\\.|/" "_" var ${var})
    string(REPLACE "*" "P" var ${var})
    if(NOT DEFINED ${var})
        check_type_size(${TYPE} ${var})
    endif()
endmacro()

macro(ac_c_bigendian)
    test_big_endian(WORDS_BIGENDIAN)
    if(NOT WORDS_BIGENDIAN)
        set(WORDS_LITTLEENDIAN 1)
    endif()
endmacro()

# Check for inline.
macro(ac_c_inline)
    foreach (keyword inline __inline__ __inline)
        check_c_source_compiles("
        static ${keyword} void foo() { return 0; }
        int main() {}" C_HAS_${keyword})
        if (C_HAS_${keyword})
            set(C_INLINE ${keyword})
            break ()
        endif ()
    endforeach ()
    if (C_INLINE)
        # Check for GNU-style extern inline.
        check_c_source_compiles("
        extern ${C_INLINE} double foo(double x);
        extern ${C_INLINE} double foo(double x) { return x + 1.0; }
        double foo(double x) { return x + 1.0; }
        int main() { foo(1.0); }" C_EXTERN_INLINE)
        if (C_EXTERN_INLINE)
            set(HAVE_INLINE 1)
        else ()
        # Check for C99-style inline.
        check_c_source_compiles("
            extern inline void* foo() { foo(); return &foo; }
            int main() { return foo() != 0; }" C_C99INLINE)
        if (C_C99INLINE)
            set(HAVE_INLINE 1)
            set(HAVE_C99_INLINE 1)
        endif ()
        endif ()
        endif ()
        if (C_INLINE AND NOT C_HAS_inline)
            set(inline ${C_INLINE})
    endif ()
endmacro()

macro(ac_c_const)
    check_c_source_compiles(
        "int main(int argc, char **argv){const int r = 0;return r;}"
    HAS_CONST_SUPPORT)
    if (NOT HAS_CONST_SUPPORT)
        set(const "")
    endif()
endmacro()

macro(ac_sys_largefile)
    check_c_source_runs("
#include <sys/types.h>
#define BIG_OFF_T (((off_t)1<<62)-1+((off_t)1<<62))
int main (int argc, char **argv) {
    int big_off_t=((BIG_OFF_T%2147483629==721) &&
                   (BIG_OFF_T%2147483647==1));
    return big_off ? 0 : 1;
}
" HAVE_LARGE_FILE_SUPPORT)

    # Check if it makes sense to define _LARGE_FILES or _FILE_OFFSET_BITS
    if (NOT HAVE_LARGE_FILE_SUPPORT)
  
        set (_LARGE_FILE_EXTRA_SRC "
#include <sys/types.h>
int main (int argc, char **argv) {
  return sizeof(off_t) == 8 ? 0 : 1;
}
")
        check_c_source_runs("#define _LARGE_FILES\n${_LARGE_FILE_EXTRA_SRC}" 
            HAVE_USEFUL_D_LARGE_FILES)
        if (NOT HAVE_USEFUL_D_LARGE_FILES)
            if (NOT DEFINED HAVE_USEFUL_D_FILE_OFFSET_BITS)
              set(SHOW_LARGE_FILE_WARNING TRUE)
            endif()
            check_c_source_runs("#define _FILE_OFFSET_BITS 64\n${_LARGE_FILE_EXTRA_SRC}"
                HAVE_USEFUL_D_FILE_OFFSET_BITS)
            if(HAVE_USEFUL_D_FILE_OFFSET_BITS)
                set(_FILE_OFFSET_BITS 64)
            elseif(SHOW_LARGE_FILE_WARNING)
                message(WARNING "No 64 bit file support through off_t available.")
            endif()
        else()
            set(_LARGE_FILES 1)
        endif()
    endif()
endmacro()

macro(ac_header_dirent)
    ac_check_headers(dirent.h sys/ndir.h sys/dir.h ndir.h)
endmacro()

macro(ac_header_resolv)
    ac_check_headers(sys/types.h netinet/in.h arpa/nameser.h netdb.h resolv.h)
endmacro()

macro(ac_header_stdbool)
    ac_check_headers(stdbool.h)
    check_type_size(_Bool HAVE__BOOL)
endmacro()

macro(ac_header_stdc)
    if (NOT DEFINED STDC_HEADERS)
        ac_check_headers(stdlib.h stdarg.h stddef.h string.h float.h)
        check_symbol_exists(memchr string.h HAVE_MEMCHR)
        check_symbol_exists(free stdlib.h HAVE_FREE)
        check_c_source_compiles("
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <float.h>

int main(int argc, char **argv)
{
  void *ptr;
  free((void*)1);
  ptr = memchr((void*)1, 0, 0);

  return (int)ptr;
}

        " STDC_HEADERS)
    endif()
endmacro()

macro(ac_header_time)
    check_c_source_compiles("
# include <sys/time.h>
# include <time.h>
int main {}
    " TIME_WITH_SYS_TIME)
    ac_check_headers(sys/time.h)
endmacro()

macro(ac_struct_tm)
    check_c_source_compiles(
        "
#include <sys/time.h>
int main(int argc, char **argv) { struct tm x; return 0; }"
        TM_IN_SYS_TIME
    )
endmacro()

macro(ac_type_off_t)
    ac_check_sizeof(off_t)
    if (NOT SIZEOF_OFF_T)
        set(off_t "long int")
    endif()
    set(off_t ${off_t})
endmacro()


macro(ac_type_size_t)
    ac_check_sizeof(size_t)
    if (NOT SIZEOF_SIZE_T)
        set(size_t "unsigned int")
    endif()
    set(size_t ${size_t})
endmacro()

function(ac_parse_includes OUTPUT_VAR)
    if(${ARGC} GREATER 1)
        string(REGEX MATCHALL "include[ \t]+<[a-zA-Z0-9_./]+>" INCLUDES ${ARGN})
        string(REGEX REPLACE "<|>" "" INCLUDES ${INCLUDES})
        string(REGEX REPLACE "include[ \t]+" ";" INCLUDES ${INCLUDES})
        list(REMOVE_AT INCLUDES 0)
        set(INCLUDE_LIST ${INCLUDES})
    else()
        set(INCLUDE_LIST)
    endif()
    ac_check_includes_list(OUT_INCLUDES ${INCLUDE_LIST})
    set(${OUTPUT_VAR} ${OUT_INCLUDES} PARENT_SCOPE)
endfunction()

function(ac_parse_struct_member ELEM STRUCT MEMBER)
    string(REPLACE "." ";" ELEM_LIST "${ELEM}")
    list(GET ELEM_LIST 0 _STRUCT)
    string(STRIP "${_STRUCT}" _STRUCT)
    list(REMOVE_AT ELEM_LIST 0)
    string(REPLACE ";" "." _MEMBER "${ELEM_LIST}")
    set(${STRUCT} ${_STRUCT} PARENT_SCOPE)
    set(${MEMBER} ${_MEMBER} PARENT_SCOPE)
endfunction()

macro(_ac_check_member STRUCT MEMBER)
    string(TOUPPER HAVE_${STRUCT}_${MEMBER} var)
    string(REPLACE "." "_" var ${var})
    string(REPLACE " " "_" var ${var})
    if(NOT DEFINED ${var})
        if(${ARGC} GREATER 2)
            set(HEADERS ${ARGN})
        else()
            ac_includes_default_list(HEADERS)
        endif()
        check_struct_has_member(${STRUCT} ${MEMBER} "${HEADERS}" ${var})
    endif()
endmacro()

macro(ac_check_member STRUCT_MEMBER)
    ac_parse_struct_member("${STRUCT_MEMBER}" STRUCT MEMBER)
    ac_parse_includes(INCLUDES ${ARGN})
    _ac_check_member(${STRUCT} ${MEMBER} ${INCLUDES})
endmacro()

macro(ac_check_members STRUCT_MEMBERS)
    string(REPLACE "," ";" STRUCT_MEMBER_LIST "${STRUCT_MEMBERS}")
    foreach(STRUCT_MEMBER ${STRUCT_MEMBER_LIST})
        ac_check_member(${STRUCT_MEMBER} ${ARGN})
    endforeach()
endmacro()

function(ac_config_file)
    set(prefix ${CMAKE_INSTALL_PREFIX})
    set(exec_prefix ${CMAKE_INSTALL_PREFIX})
    set(libdir ${CMAKE_INSTALL_PREFIX}/lib)
    set(includedir ${CMAKE_INSTALL_PREFIX}/include)
    configure_file(${ARGN})
endfunction()

macro(_ac_check_type TYPE)
    string(TOUPPER "HAVE_${TYPE}" VAR)
    if(${ARGC} GREATER 1)
        set(CMAKE_EXTRA_INCLUDE_FILES ${ARGN})
    else()
        ac_includes_default_list(DEFAULT_INCLUDES)
        set(CMAKE_EXTRA_INCLUDE_FILES ${DEFAULT_INCLUDES})
    endif()
    check_type_size(${TYPE} ${VAR})
    set(CMAKE_EXTRA_INCLUDE_FILES)
endmacro()

macro(ac_check_type TYPE)
    ac_parse_includes(INCLUDES ${ARGN})
    _ac_check_type(${TYPE} ${INCLUDES})
endmacro()

macro(ac_check_types prelude)
    ac_parse_includes(INCLUDES ${prelude})
    foreach(typename ${ARGN})
        _ac_check_type(${typename} ${INCLUDES})
    endforeach()
endmacro()

macro(_ac_check_decl SYMBOL)
    string(TOUPPER "HAVE_DECL_${SYMBOL}" VAR)
    if(${ARGC} GREATER 1)
        set(INCLUDES ${ARGN})
    else()
        ac_includes_default_list(DEFAULT_INCLUDES)
        set(INCLUDES ${DEFAULT_INCLUDES})
    endif()
    check_symbol_exists(${SYMBOL} "${INCLUDES}" ${VAR})
endmacro()

macro(ac_check_decl TYPE)
    ac_parse_includes(INCLUDES ${ARGN})
    _ac_check_decl(${TYPE} ${INCLUDES})
endmacro()

macro(ac_check_decls prelude)
    ac_parse_includes(INCLUDES ${prelude})
    foreach(typename ${ARGN})
        _ac_check_decl(${typename} ${INCLUDES})
    endforeach()
endmacro()

macro(ac_set_lib_var LIB VAR)
    if(VAR STREQUAL "")
        string(TOUPPER "HAVE_${LIB}" ${UPPER_VAR})
        set(${UPPER_VAR} ${ARGN})
    else()
        set(${VAR} ${ARGN})
    endif()
endmacro()

macro(ac_check_lib VAR LIB FUN)
    find_library(ac_cv_lib_${LIB} ${LIB})
    set(CMAKE_REQUIRED_LIBRARIES ${ARGN})
    check_library_exists(${LIB} ${FUN} _LIB_LOCATION _LIB_FUNCTION_FOUND)
    if(_LIB_FUNCTION_FOUND)
        ac_set_lib_var(${LIB} "${VAR}" 1)
        set(LIBS "${LIBS} -L${_LIB_LOCATION} -l${LIB}")
    endif()
    set(CMAKE_REQUIRED_LIBRARIES)
endmacro()

macro(ac_search_libs VAR FUN LIBS)
    string(TOUPPER "HAVE_${FUN}" FUN_VAR)
    check_function_exists(${FUN} ${FUN_VAR})
    if(${FUN_VAR})
        ac_set_lib_var(${LIB} "${VAR}" 1)
    else()
        foreach(LIB ${LIBS})
            ac_check_lib(ac_cv_search_${FUN}_${LIB} ${LIB} ${FUN} ${ARGN})
            if(${ac_cv_search_${FUN}_${LIB}})
                ac_set_lib_var(${LIB} "${VAR}" 1)
                break()
            endif()
        endforeach()
    endif()
endmacro()

macro(ac_source_compiles SRC VAR)
    check_c_source_compiles("${SRC}" ac_cv_source_compiles_${VAR})
    if(ac_cv_source_compiles_${VAR})
        set(${VAR} 1)
    else()
        set(${VAR} 0)
    endif()
endmacro()

macro(ac_source_runs SRC VAR)
    check_c_source_runs("${SRC}" ac_cv_source_runs_${VAR})
    if(ac_cv_source_runs_${VAR})
        set(${VAR} 1)
    else()
        set(${VAR} 0)
    endif()
endmacro()


set(AC_LIBSRCS)
macro(ac_libobj FUN)
    check_function_exists(${FUN} ac_libobj_have_${FUN})
    if(ac_libobj_have_${FUN})
        list(APPEND AC_LIBSRS ${ARGN}${FUN}.c)
    endif()
endmacro()

macro(ac_replace_funcs PREFIX)
    ac_check_funcs(${ARGN})
    foreach(FUN ${ARGN})
        ac_libobj(${FUN} ${PREFIX})
    endforeach()
endmacro()

macro(ac_init NAME VER)
    set(PACKAGE ${NAME})
    set(PACKAGE_NAME ${PACKAGE})
    set(PACKAGE_TARNAME ${PACKAGE})
    set(PACKAGE_BUGREPORT "")
    set(PACKAGE_URL "")
    set(PACKAGE_VERSION ${VER})
    set(VERSION ${VER})
endmacro()
