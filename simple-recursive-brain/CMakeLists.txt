cmake_minimum_required(VERSION 3.12) # Ensure a minimum version of CMake

set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

project(SRB VERSION 1.0 LANGUAGES CXX) # Project name and version

# Set C++ standard/compile options
set(CMAKE_CXX_STANDARD 20) # Use C++20 standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set debug and release flags globally
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Find OpenMP package
find_package(OpenMP REQUIRED)

# Collect all source files except test-run.cpp
set(SOURCE_FILES
    src/brain.cpp
    src/edge.cpp
    src/node.cpp
    src/util.cpp
)

# Debug version of the core library
add_library(core_debug ${SOURCE_FILES})
target_compile_options(core_debug PRIVATE -g -ggdb)
target_include_directories(core_debug PUBLIC "${PROJECT_SOURCE_DIR}/src")
if(OpenMP_CXX_FOUND)
    target_link_libraries(core_debug PUBLIC OpenMP::OpenMP_CXX)
endif()

# Release version of the core library
add_library(core_release ${SOURCE_FILES})
target_compile_options(core_release PRIVATE -O3)
target_include_directories(core_release PUBLIC "${PROJECT_SOURCE_DIR}/src")
if(OpenMP_CXX_FOUND)
    target_link_libraries(core_release PUBLIC OpenMP::OpenMP_CXX)
endif()

# Define the executable that uses test-run.cpp and links against the library
add_executable(SRBDebug src/test-run.cpp)
target_link_libraries(SRBDebug PRIVATE core_debug)
set_target_properties(SRBDebug PROPERTIES OUTPUT_NAME "SRBDebug")

# Release build (optimized)
add_executable(SRBOptimized src/test-run.cpp)
target_link_libraries(SRBOptimized PRIVATE core_release)
set_target_properties(SRBOptimized PROPERTIES OUTPUT_NAME "SRB")

# Optionally, link OpenMP to the executables if they use OpenMP features directly
if(OpenMP_CXX_FOUND)
    target_link_libraries(SRBDebug PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(SRBOptimized PRIVATE OpenMP::OpenMP_CXX)
endif()
