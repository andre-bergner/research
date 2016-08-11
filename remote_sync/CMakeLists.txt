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
