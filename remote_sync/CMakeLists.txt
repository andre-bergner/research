cmake_minimum_required( VERSION 3.2 )


find_package( boost )

add_executable( remote_sync "stuart_landau_net.cpp" )

target_include_directories( remote_sync PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../libs
    ${CMAKE_CURRENT_SOURCE_DIR}/../libs/dynamical_system
)

set_property(TARGET remote_sync PROPERTY CXX_STANDARD 14)

if( CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang" )
  target_compile_options( remote_sync PRIVATE -fcolor-diagnostics )
endif()


find_program(PYTHON3 python3 PATHS ENV PATH)
find_program(PYTHON2 python2 PATHS ENV PATH)


if (NOT PYTHON3)
   message (FATAL_ERROR "python3 not found!")
endif()

if (NOT PYTHON2)
   message (FATAL_ERROR "python2 not found!")
endif()

#add_custom_target( run
#   COMMAND ${PYTHON4_PATH} -c 'print(123)'
#   WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
#)

#add_custom_target( run
#   COMMAND ${PYTHON3_PATH} ${CMAKE_SOURCE_DIR}/cov/live_cover.py $<TARGET_FILE:${PROJECT_NAME}> ${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR}
#   WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
#)
