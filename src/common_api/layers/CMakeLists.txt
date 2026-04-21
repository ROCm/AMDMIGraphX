file(GLOB LAYER_SRC *.cpp)
target_sources(commonapi PRIVATE ${LAYER_SRC})
target_include_directories(commonapi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
