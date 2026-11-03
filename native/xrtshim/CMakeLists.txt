# xrtshim for Windows (MSVC), against the xrt_coreutil.dll AMD's NPU driver installs.
#
#   .\make-importlib.ps1
#   cmake -B build -DXRT_LIBRARY=xrt_coreutil.lib
#   cmake --build build --config Release
#
# make-importlib.ps1 makes xrt_coreutil.lib from the DLL. The XRT headers are in include/
# (XRT 2.21, Apache-2.0); XRT_INCLUDE_DIR takes others. On Linux, build-linux.sh is simpler.
cmake_minimum_required(VERSION 3.16)
project(xrtshim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(XRT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE PATH "the directory holding xrt/xrt_bo.h")
set(XRT_LIBRARY "" CACHE FILEPATH "xrt_coreutil's import library (Windows) or shared library")
if(NOT XRT_LIBRARY)
  message(FATAL_ERROR "set XRT_LIBRARY (make-importlib.ps1 makes xrt_coreutil.lib)")
endif()

add_library(xrtshim SHARED xrtshim.cpp)
target_include_directories(xrtshim SYSTEM PRIVATE ${XRT_INCLUDE_DIR})
target_link_libraries(xrtshim PRIVATE ${XRT_LIBRARY})
if(MSVC)
  target_compile_options(xrtshim PRIVATE /EHsc /W3)
  # xrt/detail/xclbin.h uses Windows' own uuid functions.
  target_link_libraries(xrtshim PRIVATE rpcrt4)
else()
  set_target_properties(xrtshim PROPERTIES CXX_VISIBILITY_PRESET hidden)
endif()
