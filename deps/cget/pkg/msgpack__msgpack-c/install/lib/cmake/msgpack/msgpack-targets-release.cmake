#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "msgpackc" for configuration "Release"
set_property(TARGET msgpackc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(msgpackc PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libmsgpackc.so.2.0.0"
  IMPORTED_SONAME_RELEASE "libmsgpackc.so.2"
  )

list(APPEND _IMPORT_CHECK_TARGETS msgpackc )
list(APPEND _IMPORT_CHECK_FILES_FOR_msgpackc "${_IMPORT_PREFIX}/lib/libmsgpackc.so.2.0.0" )

# Import target "msgpackc-static" for configuration "Release"
set_property(TARGET msgpackc-static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(msgpackc-static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libmsgpackc.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS msgpackc-static )
list(APPEND _IMPORT_CHECK_FILES_FOR_msgpackc-static "${_IMPORT_PREFIX}/lib/libmsgpackc.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
