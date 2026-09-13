cmake_minimum_required(VERSION 3.18)
project(tracker_ops CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")

find_package(OpenCV REQUIRED)

# Вказуємо шукати спочатку переданий Python3_EXECUTABLE
if(NOT Python3_EXECUTABLE)
    set(Python3_EXECUTABLE "${CMAKE_SOURCE_DIR}/.venv/bin/python")
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Отримуємо шлях до pybind11 через вказаний Python3_EXECUTABLE
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_FOUND
)

if(NOT PYBIND11_FOUND EQUAL 0)
    message(FATAL_ERROR "pybind11 not found in ${Python3_EXECUTABLE}. Run: ${Python3_EXECUTABLE} -m pip install pybind11")
endif()

find_package(pybind11 CONFIG REQUIRED)

file(GLOB_RECURSE SOURCES 
    "src/*.cpp"
    "bindings/*.cpp"
)

pybind11_add_module(tracker_ops ${SOURCES})
target_include_directories(tracker_ops PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(tracker_ops PRIVATE ${OpenCV_LIBS})
install(TARGETS tracker_ops LIBRARY DESTINATION .)
