# Copyright (C) 2007-2009 LuaDist.
# Submitted by David Manura
# Redistribution and use of this file is allowed according to the terms of the MIT license.
# For details see the COPYRIGHT file distributed with LuaDist.
# Please note that the package source code is licensed under its own license.

PROJECT(bzip2 C)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
find_package(cget-recipe-utils)
# Where to install module parts:
set(INSTALL_BIN bin CACHE PATH "Where to install binaries to.")
set(INSTALL_LIB lib CACHE PATH "Where to install libraries to.")
set(INSTALL_INC include CACHE PATH "Where to install headers to.")
set(INSTALL_ETC etc CACHE PATH "Where to store configuration files")
set(INSTALL_DATA share/${PROJECT_NAME} CACHE PATH "Directory the package can store documentation, tests or other data in.")
set(INSTALL_DOC ${INSTALL_DATA}/doc CACHE PATH "Recommended directory to install documentation into.")
set(INSTALL_EXAMPLE ${INSTALL_DATA}/example CACHE PATH "Recommended directory to install examples into.")
set(INSTALL_TEST ${INSTALL_DATA}/test CACHE PATH "Recommended directory to install tests into.")
set(INSTALL_FOO ${INSTALL_DATA}/etc CACHE PATH "Where to install additional files")


# In MSVC, prevent warnings that can occur when using standard libraries.
if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif(MSVC)

ADD_DEFINITIONS(-D_FILE_OFFSET_BITS=64)

if(MINGW)
    patch_file(bzip2.c "sys\\stat.h" "sys/stat.h")
endif()

# Library
SET(BZIP2_SRCS blocksort.c huffman.c crctable.c randtable.c
               compress.c decompress.c bzlib.c )

ADD_LIBRARY(bz2 ${BZIP2_SRCS})

ADD_EXECUTABLE(bzip2 bzip2.c)
TARGET_LINK_LIBRARIES(bzip2 bz2)

ADD_EXECUTABLE(bzip2recover bzip2recover.c)

FILE(WRITE bzegrep.1 ".so man1/bzgrep.1")
FILE(WRITE bzfgrep.1 ".so man1/bzgrep.1")
FILE(WRITE bzless.1 ".so man1/bzmore.1")
FILE(WRITE bzcmp.1 ".so man1/bzdiff.1")

INCLUDE(CTest)
ADD_TEST(test ${CMAKE_COMMAND} -P test.cmake)

INSTALL(TARGETS bzip2 bzip2recover bz2 RUNTIME DESTINATION ${INSTALL_BIN} LIBRARY DESTINATION ${INSTALL_LIB} ARCHIVE DESTINATION ${INSTALL_LIB})
INSTALL(FILES bzlib.h DESTINATION ${INSTALL_INC})
INSTALL(FILES README LICENSE manual.html bzip2.1 bzgrep.1 bzmore.1 bzdiff.1
              bzegrep.1 bzfgrep.1 bzless.1 bzcmp.1
              DESTINATION ${INSTALL_DOC})
INSTALL(PROGRAMS bzgrep bzmore bzdiff DESTINATION ${INSTALL_BIN}) #~2DO: windows versions?

#~2DO: improve with symbolic links
INSTALL(PROGRAMS $<TARGET_FILE:bzip2> DESTINATION ${INSTALL_BIN} RENAME bunzip2)
INSTALL(PROGRAMS $<TARGET_FILE:bzip2> DESTINATION ${INSTALL_BIN} RENAME bzcat)
INSTALL(PROGRAMS bzgrep DESTINATION ${INSTALL_BIN} RENAME bzegrep)
INSTALL(PROGRAMS bzgrep DESTINATION ${INSTALL_BIN} RENAME bzfgrep)
INSTALL(PROGRAMS bzmore DESTINATION ${INSTALL_BIN} RENAME bzless)
INSTALL(PROGRAMS bzdiff DESTINATION ${INSTALL_BIN} RENAME bzcmp)

#~2DO? build manual.ps and manual.pdf

