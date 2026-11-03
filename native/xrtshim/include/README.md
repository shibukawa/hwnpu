# XRT headers

The headers xrtshim includes, from XRT 2.21.75 (Ubuntu's `libxrt-dev` 2.21.75+dfsg-4), and
`xrt/detail/windows/` from XRT's sources (github.com/Xilinx/XRT,
`src/runtime_src/core/include/xrt/detail/windows`), which the package leaves out. They are
here so that xrtshim builds on Windows without XRT's sources. XRT is licensed under the Apache
License 2.0; each header keeps its notice.
