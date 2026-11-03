/*
 * xrtshim: a C ABI over XRT's C++ API, for hwnpu (Go without cgo) to reach AMD's NPU where
 * the only way in is XRT: Windows, whose NPU driver ships xrt_coreutil.dll. It builds on Linux
 * too, against the XRT packages, which is how it is tested (internal/backend/amdxrt).
 *
 * Every function that can fail returns NULL on success, or an error message the caller frees
 * with xs_free_error. Objects are opaque and freed by their xs_*_free.
 */
#ifndef HWNPU_XRTSHIM_H
#define HWNPU_XRTSHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#define XS_API __declspec(dllexport)
#else
#define XS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* The ABI version: hwnpu refuses a shim of another. */
#define XS_ABI 1

typedef struct xs_device xs_device;
typedef struct xs_bo xs_bo;
typedef struct xs_ctx xs_ctx;
typedef struct xs_kernel xs_kernel;
typedef struct xs_run xs_run;

/* Buffer kinds of xs_bo_alloc. */
enum {
  XS_BO_HOST = 0,      /* xrt::bo::flags::host_only: data the NPU reads and writes */
  XS_BO_CACHEABLE = 1, /* xrt::bo::flags::cacheable: instruction streams */
};

/* Directions of xs_bo_sync. */
enum {
  XS_SYNC_TO_DEVICE = 0,
  XS_SYNC_FROM_DEVICE = 1,
};

/* Keys of xs_device_info. */
enum {
  XS_INFO_NAME = 0,
  XS_INFO_BDF = 1,
  XS_INFO_PLATFORM = 2,   /* JSON */
  XS_INFO_ELECTRICAL = 3, /* JSON */
  XS_INFO_XRT_VERSION = 4,
};

/* Command states of xs_run_wait and xs_run_state (XRT's ert_cmd_state). */
enum {
  XS_STATE_NEW = 1,
  XS_STATE_QUEUED = 2,
  XS_STATE_COMPLETED = 4,
  XS_STATE_ERROR = 5,
  XS_STATE_ABORT = 6,
  XS_STATE_SUBMITTED = 7,
  XS_STATE_TIMEOUT = 8,
  XS_STATE_NORESPONSE = 9,
};

XS_API int xs_abi(void);
XS_API void xs_free_error(char *err);

XS_API char *xs_device_count(uint32_t *count);
XS_API char *xs_device_open(uint32_t index, xs_device **out);
XS_API void xs_device_free(xs_device *d);
/* Writes the value, NUL terminated, into buf of n bytes; *need is its length with the NUL. */
XS_API char *xs_device_info(xs_device *d, int key, char *buf, size_t n, size_t *need);

/* group: the memory group (xs_kernel_group_id of the argument the buffer is for); 0 for a
 * full ELF's buffers. */
XS_API char *xs_bo_alloc(xs_device *d, size_t size, int kind, int group, xs_bo **out);
XS_API void xs_bo_free(xs_bo *b);
XS_API char *xs_bo_map(xs_bo *b, void **ptr);
XS_API uint64_t xs_bo_address(xs_bo *b);
XS_API char *xs_bo_sync(xs_bo *b, int dir, size_t off, size_t n);

/* A hardware context on an xclbin (an AIE2 design: NPU1, or an NPU2 xclbin). */
XS_API char *xs_ctx_xclbin(xs_device *d, const void *axlf, size_t n, xs_ctx **out);
/* A hardware context on a full ELF (AIE2P: NPU2). */
XS_API char *xs_ctx_elf(xs_device *d, const void *elf, size_t n, xs_ctx **out);
XS_API void xs_ctx_free(xs_ctx *c);

/* A kernel by name: an xclbin's kernel, or a full ELF's "kernel:instance". */
XS_API char *xs_kernel_open(xs_ctx *c, const char *name, xs_kernel **out);
XS_API void xs_kernel_free(xs_kernel *k);
XS_API int xs_kernel_group_id(xs_kernel *k, int arg);

XS_API char *xs_run_new(xs_kernel *k, xs_run **out);
XS_API void xs_run_free(xs_run *r);
XS_API char *xs_run_set_bo(xs_run *r, int index, xs_bo *b);
XS_API char *xs_run_set_scalar(xs_run *r, int index, const void *value, size_t n);
/* A full ELF's run: the ctrl scratchpad its control code reads per-call words from, mapped
 * (NULL and 0 when it has none). It is synced to the device by xs_run_start. */
XS_API char *xs_run_scratchpad(xs_run *r, void **ptr, size_t *n);
XS_API char *xs_run_start(xs_run *r);
/* Waits up to timeout_ms (0: forever) and writes the state: XS_STATE_TIMEOUT if not done. */
XS_API char *xs_run_wait(xs_run *r, uint32_t timeout_ms, int *state);
XS_API int xs_run_state(xs_run *r);

#ifdef __cplusplus
}
#endif

#endif
