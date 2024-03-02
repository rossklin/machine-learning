cmake_minimum_required(VERSION 3.12) # Ensure a minimum version of CMake

project(SRB VERSION 1.0 LANGUAGES CXX) # Project name and version

# Set C++ standard/compile options
set(CMAKE_CXX_STANDARD 20) # Use C++20 standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Collect all source files except test-run.cpp
set(SOURCE_FILES
    src/brain.cpp
    src/edge.cpp
    src/node.cpp
    src/util.cpp
)

# Define a library from the collected source files
add_library(core ${SOURCE_FILES})

# Include directories for the compiler
target_include_directories(core PUBLIC "${PROJECT_SOURCE_DIR}/src")

# Define the executable that uses test-run.cpp and links against the library
add_executable(SRB src/test-run.cpp)
target_link_libraries(SRB PRIVATE core)

# Debug build (with debug info and no optimizations)
set_target_properties(SRB PROPERTIES COMPILE_FLAGS "-g" OUTPUT_NAME "SRBDebug")

# Release build (optimized)
add_executable(MyProjectOptimized src/test-run.cpp)
target_link_libraries(MyProjectOptimized PRIVATE core)
set_target_properties(MyProjectOptimized PROPERTIES COMPILE_FLAGS "-O3" OUTPUT_NAME "SRB")
