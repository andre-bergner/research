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
if (PYTHON3)
   message ("Using python3.")
   set(PYTHON ${PYTHON3})
endif()

if (NOT PYTHON)
   find_program(PYTHON2 python2 PATHS ENV PATH)
   if (PYTHON2)
      message ("Using python2.")
      set(PYTHON ${PYTHON2})
   endif()
endif()


if (NOT PYTHON)
   message (FATAL_ERROR "No python found! Searched for python2 and python3.")
endif()


add_custom_target( run
   COMMAND ${PYTHON3} ${CMAKE_SOURCE_DIR}/sync_study.py
   WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
)
