// xrtshim: a C ABI over XRT's C++ API. See xrtshim.h.

#include "xrtshim.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_ext.h"
#include "xrt/experimental/xrt_system.h"
#include "xrt/experimental/xrt_version.h"
#include "xrt/experimental/xrt_xclbin.h"

struct xs_device {
  xrt::device dev;
};

struct xs_bo {
  xrt::bo bo;
};

struct xs_ctx {
  xrt::hw_context ctx;
  bool elf = false;
};

struct xs_kernel {
  xrt::kernel k;
};

struct xs_run {
  xrt::run run;
  xrt::bo scratchpad; // empty until xs_run_scratchpad finds one
};

namespace {

char *error(const std::string &msg) {
  char *p = static_cast<char *>(std::malloc(msg.size() + 1));
  if (p != nullptr) {
    std::memcpy(p, msg.c_str(), msg.size() + 1);
  }
  return p;
}

// guard runs f, turning an exception into an error message.
template <typename F> char *guard(const char *what, F f) {
  try {
    f();
    return nullptr;
  } catch (const std::exception &e) {
    return error(std::string(what) + ": " + e.what());
  } catch (...) {
    return error(std::string(what) + ": unknown exception");
  }
}

std::string version() {
  return std::to_string(xrt::version::major()) + "." + std::to_string(xrt::version::minor()) + "." +
         std::to_string(xrt::version::patch());
}

} // namespace

extern "C" {

int xs_abi(void) { return XS_ABI; }

void xs_free_error(char *err) { std::free(err); }

char *xs_device_count(uint32_t *count) {
  return guard("enumerate devices", [&] { *count = static_cast<uint32_t>(xrt::system::enumerate_devices()); });
}

char *xs_device_open(uint32_t index, xs_device **out) {
  *out = nullptr;
  return guard("open device", [&] { *out = new xs_device{xrt::device(index)}; });
}

void xs_device_free(xs_device *d) { delete d; }

char *xs_device_info(xs_device *d, int key, char *buf, size_t n, size_t *need) {
  return guard("device info", [&] {
    std::string v;
    switch (key) {
    case XS_INFO_NAME:
      v = d->dev.get_info<xrt::info::device::name>();
      break;
    case XS_INFO_BDF:
      v = d->dev.get_info<xrt::info::device::bdf>();
      break;
    case XS_INFO_PLATFORM:
      v = d->dev.get_info<xrt::info::device::platform>();
      break;
    case XS_INFO_ELECTRICAL:
      v = d->dev.get_info<xrt::info::device::electrical>();
      break;
    case XS_INFO_XRT_VERSION:
      v = version();
      break;
    default:
      throw std::runtime_error("unknown key " + std::to_string(key));
    }
    *need = v.size() + 1;
    if (n > 0) {
      size_t m = v.size() < n - 1 ? v.size() : n - 1;
      std::memcpy(buf, v.data(), m);
      buf[m] = 0;
    }
  });
}

char *xs_bo_alloc(xs_device *d, size_t size, int kind, int group, xs_bo **out) {
  *out = nullptr;
  return guard("allocate buffer", [&] {
    auto flags = kind == XS_BO_CACHEABLE ? xrt::bo::flags::cacheable : xrt::bo::flags::host_only;
    *out = new xs_bo{xrt::bo(d->dev, size, flags, static_cast<xrt::memory_group>(group))};
  });
}

void xs_bo_free(xs_bo *b) { delete b; }

char *xs_bo_map(xs_bo *b, void **ptr) {
  *ptr = nullptr;
  return guard("map buffer", [&] { *ptr = b->bo.map(); });
}

uint64_t xs_bo_address(xs_bo *b) {
  try {
    return b->bo.address();
  } catch (...) {
    return 0;
  }
}

char *xs_bo_sync(xs_bo *b, int dir, size_t off, size_t n) {
  return guard("sync buffer", [&] {
    b->bo.sync(dir == XS_SYNC_FROM_DEVICE ? XCL_BO_SYNC_BO_FROM_DEVICE : XCL_BO_SYNC_BO_TO_DEVICE, n, off);
  });
}

char *xs_ctx_xclbin(xs_device *d, const void *axlf, size_t n, xs_ctx **out) {
  *out = nullptr;
  return guard("load xclbin", [&] {
    xrt::xclbin x(std::vector<char>(static_cast<const char *>(axlf), static_cast<const char *>(axlf) + n));
    auto uuid = d->dev.register_xclbin(x);
    *out = new xs_ctx{xrt::hw_context(d->dev, uuid), false};
  });
}

char *xs_ctx_elf(xs_device *d, const void *data, size_t n, xs_ctx **out) {
  *out = nullptr;
  return guard("load ELF", [&] {
    xrt::elf e(data, n);
    auto *c = new xs_ctx{xrt::hw_context(d->dev, e), true};
    *out = c;
  });
}

void xs_ctx_free(xs_ctx *c) { delete c; }

char *xs_kernel_open(xs_ctx *c, const char *name, xs_kernel **out) {
  *out = nullptr;
  return guard("open kernel", [&] {
    if (c->elf) {
      *out = new xs_kernel{xrt::ext::kernel(c->ctx, std::string(name))};
    } else {
      *out = new xs_kernel{xrt::kernel(c->ctx, std::string(name))};
    }
  });
}

void xs_kernel_free(xs_kernel *k) { delete k; }

int xs_kernel_group_id(xs_kernel *k, int arg) {
  try {
    return k->k.group_id(arg);
  } catch (...) {
    return -1;
  }
}

char *xs_run_new(xs_kernel *k, xs_run **out) {
  *out = nullptr;
  return guard("create run", [&] { *out = new xs_run{xrt::run(k->k), xrt::bo()}; });
}

void xs_run_free(xs_run *r) { delete r; }

char *xs_run_set_bo(xs_run *r, int index, xs_bo *b) {
  return guard("set argument", [&] { r->run.set_arg(index, b->bo); });
}

char *xs_run_set_scalar(xs_run *r, int index, const void *value, size_t n) {
  return guard("set argument", [&] { r->run.set_arg(index, value, n); });
}

char *xs_run_scratchpad(xs_run *r, void **ptr, size_t *n) {
  *ptr = nullptr;
  *n = 0;
  return guard("ctrl scratchpad", [&] {
    if (!r->scratchpad) {
      r->scratchpad = r->run.get_ctrl_scratchpad_bo();
    }
    if (r->scratchpad) {
      *ptr = r->scratchpad.map();
      *n = r->scratchpad.size();
    }
  });
}

char *xs_run_start(xs_run *r) {
  return guard("start run", [&] {
    if (r->scratchpad) {
      r->scratchpad.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    r->run.start();
  });
}

char *xs_run_wait(xs_run *r, uint32_t timeout_ms, int *state) {
  *state = 0;
  return guard("wait run", [&] { *state = static_cast<int>(r->run.wait(std::chrono::milliseconds(timeout_ms))); });
}

int xs_run_state(xs_run *r) {
  try {
    return static_cast<int>(r->run.state());
  } catch (...) {
    return XS_STATE_ERROR;
  }
}

} // extern "C"
