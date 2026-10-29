/*
 * zemock: a tiny stand-in for the Level Zero loader + Intel NPU driver, used to test hwnpu's
 * Level Zero backend on machines without an Intel NPU. It is NOT a model of the real driver's
 * performance or numerics; it only reproduces the API contract hwnpu relies on:
 *
 *  - one driver (NPU driver UUID) with one VPU device and the graph extension "ZE_extension_graph" 1.12
 *  - native blobs: "ZEMOCKB1" | u32 n | f32 scale | f32 bias, computing y = relu(scale * x + bias)
 *    on fp32 vectors of shape (1, n); they need pfnAppendGraphInitialize (COMMAND_LIST_INITIALIZE)
 *  - OpenVINO IR (NGRAPH_LITE) using the OpenVINO NPU plugin layout; only the layer types
 *    Parameter, Const, Multiply, Add, ReLU and Result are "supported". The first two Const layers
 *    are read from the weights as scale and bias. IR graphs need pfnGraphInitialize (INITIALIZE)
 *  - argument values are captured when the graph is appended to a command list
 *  - build logs through pfnBuildLogGetString(NULL, ...), supported layers as "<a><b>"
 *  - two command queue groups (0: copy only, 1: compute); queues and lists must use group 1
 *  - "ZE_extension_command_queue_npu" 1.1; queue priority, turbo and workload type are recorded
 *
 * Build: cc -shared -fPIC -o libzemock.{so,dylib} zemock.c
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef uint32_t ze_result_t;
#define OK 0u
#define NOT_READY 1u
#define ERR_INVALID_ARGUMENT 0x78000004u
#define ERR_INVALID_NULL_POINTER 0x78000007u
#define ERR_UNSUPPORTED_FEATURE 0x78000003u
#define ERR_INVALID_NATIVE_BINARY 0x7800000fu
#define ERR_UNINITIALIZED 0x78000001u
#define MAKE_VERSION(a, b) (((uint32_t)(a) << 16) | (b))

typedef struct { uint32_t stype; void *pNext; uint8_t uuid[16]; uint32_t driverVersion; } driver_props;
typedef struct { char name[256]; uint32_t version; } ext_props;
typedef struct {
    uint32_t stype; void *pNext; uint32_t type, vendorId, deviceId, flags, subdeviceId, coreClockRate;
    uint64_t maxMemAllocSize; uint32_t a, b, c, d, e, f, g; uint64_t timerResolution; uint32_t h, i;
    uint8_t uuid[16]; char name[256];
} device_props;
typedef struct { uint32_t stype; void *pNext; uint32_t ext; uint16_t major, minor; uint32_t formats, opset; } device_graph_props;
typedef struct { uint32_t stype; void *pNext; uint32_t format; size_t inputSize; const uint8_t *pInput; const char *pBuildFlags; uint32_t flags; } graph_desc2;
typedef struct { uint32_t stype; void *pNext; uint32_t numGraphArgs; uint32_t initStageRequired; } graph_props2;
typedef struct {
    uint32_t stype; void *pNext; char name[256]; uint32_t type; uint32_t dims[5];
    uint32_t networkPrecision, networkLayout, devicePrecision, deviceLayout; float quantReverseScale;
    uint8_t quantZeroPoint; uint32_t dims_count; char debug_friendly_name[256];
    char associated_tensor_names[32][256]; uint32_t associated_tensor_names_count;
} arg_props3;

static int g_drv, g_dev, g_ctx;
static char g_log[1024];
static char g_last_flags[4096];
static char g_last_rtinfo_ok;

typedef struct { uint32_t n; float scale, bias; int is_ir; int initialized; void *args[2]; } graph;
typedef struct { graph *g; int init; void *args[2]; } command;
typedef struct { command cmds[8]; int count; int closed; } cmdlist;
typedef struct { int signaled; } fence;
typedef struct { char layers[2048]; } query;

static int32_t g_queue_priority = -1, g_queue_options = -1, g_workload = -1;
static int32_t g_creates[3];
static uint32_t g_last_graph_flags;

/* exported for tests */
const char *zemock_last_build_flags(void) { return g_last_flags; }
int zemock_last_rtinfo_ok(void) { return g_last_rtinfo_ok; }
int32_t zemock_last_queue_priority(void) { return g_queue_priority; }
int32_t zemock_last_queue_options(void) { return g_queue_options; }
int32_t zemock_last_workload(void) { return g_workload; }
int32_t zemock_creates(uint32_t format) { return format < 3 ? g_creates[format] : -1; }
uint32_t zemock_last_graph_flags(void) { return g_last_graph_flags; }

static const uint8_t npu_uuid[16] = {0x01, 0x7d, 0xe9, 0x31, 0x6b, 0x4d, 0x4f, 0xd4, 0xaa, 0x9b, 0x5b, 0xed, 0x77, 0xfc, 0x8e, 0x89};

ze_result_t zeInit(uint32_t flags) { (void)flags; return OK; }

static ze_result_t list_one(uint32_t *count, void **out, void *h) {
    if (!count) return ERR_INVALID_NULL_POINTER;
    if (out && *count >= 1) out[0] = h;
    *count = 1;
    return OK;
}
ze_result_t zeDriverGet(uint32_t *count, void **drivers) { return list_one(count, drivers, &g_drv); }
ze_result_t zeDeviceGet(void *drv, uint32_t *count, void **devices) { (void)drv; return list_one(count, devices, &g_dev); }
ze_result_t zeDriverGetApiVersion(void *drv, uint32_t *v) { (void)drv; *v = MAKE_VERSION(1, 13); return OK; }

ze_result_t zeDriverGetProperties(void *drv, driver_props *p) {
    (void)drv;
    memcpy(p->uuid, npu_uuid, 16);
    p->driverVersion = 1234;
    return OK;
}

ze_result_t zeDriverGetExtensionProperties(void *drv, uint32_t *count, ext_props *props) {
    (void)drv;
    if (props && *count >= 3) {
        strcpy(props[0].name, "ZE_extension_unrelated");
        props[0].version = MAKE_VERSION(3, 0);
        strcpy(props[1].name, "ZE_extension_graph");
        props[1].version = MAKE_VERSION(1, 12);
        strcpy(props[2].name, "ZE_extension_command_queue_npu");
        props[2].version = MAKE_VERSION(1, 1);
    }
    *count = 3;
    return OK;
}

ze_result_t zeDeviceGetProperties(void *dev, device_props *p) {
    (void)dev;
    p->type = 5; /* ZE_DEVICE_TYPE_VPU */
    p->vendorId = 0x8086;
    p->deviceId = 0x7d1d;
    strcpy(p->name, "Mock Intel NPU");
    return OK;
}

ze_result_t zeContextCreate(void *drv, const void *desc, void **ctx) { (void)drv; (void)desc; *ctx = &g_ctx; return OK; }
ze_result_t zeContextDestroy(void *ctx) { (void)ctx; return OK; }

ze_result_t zeMemAllocHost(void *ctx, const void *desc, size_t size, size_t align, void **p) {
    (void)ctx; (void)desc;
    size_t rounded = (size + align - 1) / align * align;
    *p = aligned_alloc(align, rounded);
    if (*p) memset(*p, 0, rounded);
    return *p ? OK : ERR_INVALID_ARGUMENT;
}
ze_result_t zeMemFree(void *ctx, void *p) { (void)ctx; free(p); return OK; }

typedef struct { uint32_t stype; void *pNext; uint32_t flags; size_t maxFill; uint32_t numQueues; } queue_group_props;
typedef struct { uint32_t stype; const void *pNext; uint32_t ordinal, index, flags, mode, priority; } queue_desc;
typedef struct { uint32_t stype; const void *pNext; uint32_t ordinal, flags; } list_desc;
typedef struct { uint32_t stype; const void *pNext; } ext_head;
typedef struct { uint32_t stype; const void *pNext; uint8_t turbo; } queue_npu_ext;
typedef struct { uint32_t stype; const void *pNext; uint32_t options; } queue_npu_ext2;

ze_result_t zeDeviceGetCommandQueueGroupProperties(void *dev, uint32_t *count, queue_group_props *p) {
    (void)dev;
    if (p && *count >= 2) {
        p[0].flags = 0x2; /* copy */
        p[0].numQueues = 1;
        p[1].flags = 0x1; /* compute */
        p[1].numQueues = 1;
    }
    *count = 2;
    return OK;
}

ze_result_t zeCommandQueueCreate(void *ctx, void *dev, const queue_desc *desc, void **q) {
    (void)ctx; (void)dev;
    if (desc->ordinal != 1) return ERR_INVALID_ARGUMENT;
    g_queue_priority = (int32_t)desc->priority;
    g_queue_options = 0;
    g_workload = -1;
    for (const ext_head *e = desc->pNext; e; e = e->pNext) {
        if (e->stype == 0x1) g_queue_options = ((const queue_npu_ext *)e)->turbo ? 1 : 0;
        if (e->stype == 0x2) g_queue_options = (int32_t)((const queue_npu_ext2 *)e)->options;
    }
    *q = malloc(1);
    return OK;
}

static ze_result_t pfnSetWorkloadType(void *q, uint32_t workload) { (void)q; g_workload = (int32_t)workload; return OK; }
static void *queue_table[1] = {pfnSetWorkloadType};
ze_result_t zeCommandQueueDestroy(void *q) { free(q); return OK; }
ze_result_t zeCommandListCreate(void *ctx, void *dev, const list_desc *desc, cmdlist **l) {
    (void)ctx; (void)dev;
    if (desc->ordinal != 1) return ERR_INVALID_ARGUMENT;
    *l = calloc(1, sizeof(cmdlist));
    return OK;
}
ze_result_t zeCommandListDestroy(cmdlist *l) { free(l); return OK; }
ze_result_t zeCommandListClose(cmdlist *l) { l->closed = 1; return OK; }
ze_result_t zeCommandListReset(cmdlist *l) { l->count = 0; l->closed = 0; return OK; }
ze_result_t zeFenceCreate(void *q, const void *desc, fence **f) { (void)q; (void)desc; *f = calloc(1, sizeof(fence)); return OK; }
ze_result_t zeFenceDestroy(fence *f) { free(f); return OK; }
ze_result_t zeFenceReset(fence *f) { f->signaled = 0; return OK; }
ze_result_t zeFenceQueryStatus(fence *f) { return f->signaled ? OK : NOT_READY; }
ze_result_t zeFenceHostSynchronize(fence *f, uint64_t timeout) { (void)timeout; return f->signaled ? OK : NOT_READY; }

ze_result_t zeCommandQueueExecuteCommandLists(void *q, uint32_t n, cmdlist **lists, fence *f) {
    (void)q;
    for (uint32_t i = 0; i < n; i++) {
        cmdlist *l = lists[i];
        if (!l->closed) return ERR_INVALID_ARGUMENT;
        for (int c = 0; c < l->count; c++) {
            command *cmd = &l->cmds[c];
            graph *g = cmd->g;
            if (cmd->init) { g->initialized = 1; continue; }
            if (!g->initialized) return ERR_UNINITIALIZED;
            const float *x = cmd->args[0];
            float *y = cmd->args[1];
            if (!x || !y) return ERR_INVALID_NULL_POINTER;
            for (uint32_t k = 0; k < g->n; k++) {
                float v = g->scale * x[k] + g->bias;
                y[k] = v > 0 ? v : 0;
            }
        }
    }
    if (f) f->signaled = 1;
    return OK;
}

/* ---- graph extension ---- */

static const char *supported_types[] = {"Parameter", "Const", "Multiply", "Add", "ReLU", "Result", NULL};
static int is_supported(const char *type) {
    for (int i = 0; supported_types[i]; i++)
        if (strcmp(type, supported_types[i]) == 0) return 1;
    return 0;
}

static int attr(const char *tag, const char *tag_end, const char *key, char *out, size_t cap) {
    char pat[64];
    snprintf(pat, sizeof pat, " %s=\"", key);
    const char *p = strstr(tag, pat);
    if (!p || p > tag_end) return 0;
    p += strlen(pat);
    const char *e = strchr(p, '"');
    size_t len = (size_t)(e - p) < cap - 1 ? (size_t)(e - p) : cap - 1;
    memcpy(out, p, len);
    out[len] = 0;
    return 1;
}

/* parse the NGRAPH_LITE buffer; fills g (when not NULL) and the supported layer list */
static ze_result_t parse_ir(const uint8_t *in, size_t size, graph *g, char *layers, size_t layers_cap) {
    if (size < 16) { strcpy(g_log, "input too small"); return ERR_INVALID_ARGUMENT; }
    uint16_t major, minor; uint32_t count; uint64_t xml_len, bin_len;
    memcpy(&major, in, 2); memcpy(&minor, in + 2, 2); memcpy(&count, in + 4, 4); memcpy(&xml_len, in + 8, 8);
    if (major != 7 || minor != 3) { snprintf(g_log, sizeof g_log, "compiler version mismatch %u.%u", major, minor); return ERR_INVALID_ARGUMENT; }
    if (count != 2 || 16 + xml_len + 8 > size) { strcpy(g_log, "bad serialized IR header"); return ERR_INVALID_ARGUMENT; }
    memcpy(&bin_len, in + 16 + xml_len, 8);
    if (16 + xml_len + 8 + bin_len != size) { strcpy(g_log, "bad serialized IR size"); return ERR_INVALID_ARGUMENT; }
    char *xml = malloc(xml_len + 1);
    memcpy(xml, in + 16, xml_len);
    xml[xml_len] = 0;
    const uint8_t *bin = in + 16 + xml_len + 8;

    g_last_rtinfo_ok = strstr(xml, "<is_new_api value=\"true\"") && strstr(xml, "<use_indices_for_io_metadata value=\"true\"");
    ze_result_t r = OK;
    int consts = 0;
    if (layers) layers[0] = 0;
    for (const char *p = strstr(xml, "<layer "); p; p = strstr(p + 1, "<layer ")) {
        const char *end = strchr(p, '>');
        char type[64] = "", name[128] = "", buf[64];
        attr(p, end, "type", type, sizeof type);
        attr(p, end, "name", name, sizeof name);
        if (!is_supported(type)) {
            if (r == OK) snprintf(g_log, sizeof g_log, "unsupported operation %s (%s)", type, name);
            r = ERR_UNSUPPORTED_FEATURE;
            continue;
        }
        if (layers && strlen(layers) + strlen(name) + 3 < layers_cap) {
            strcat(layers, "<"); strcat(layers, name); strcat(layers, ">");
        }
        const char *data = strstr(p, "<data ");
        const char *data_end = data ? strchr(data, '>') : NULL;
        if (g && strcmp(type, "Parameter") == 0 && data && attr(data, data_end, "shape", buf, sizeof buf)) {
            const char *comma = strrchr(buf, ',');
            g->n = (uint32_t)atoi(comma ? comma + 1 : buf);
        }
        if (g && strcmp(type, "Const") == 0 && data && attr(data, data_end, "offset", buf, sizeof buf)) {
            size_t off = (size_t)atoll(buf);
            float v = 0;
            if (off + 4 <= bin_len) memcpy(&v, bin + off, 4);
            if (consts == 0) g->scale = v; else if (consts == 1) g->bias = v;
            consts++;
        }
    }
    free(xml);
    return r;
}

static ze_result_t pfnCreate2(void *ctx, void *dev, const graph_desc2 *d, graph **out) {
    (void)ctx; (void)dev;
    g_log[0] = 0;
    snprintf(g_last_flags, sizeof g_last_flags, "%s", d->pBuildFlags ? d->pBuildFlags : "");
    g_last_graph_flags = d->flags;
    /* like real drivers before graph extension 1.16: ze_graph_desc_2_t carries the stype of ze_graph_desc_t */
    if (d->stype != 0x2) { snprintf(g_log, sizeof g_log, "bad stype %#x", d->stype); return ERR_INVALID_ARGUMENT; }
    if (d->format < 3) g_creates[d->format]++;
    graph *g = calloc(1, sizeof(graph));
    ze_result_t r = OK;
    if (d->format == 1) { /* native */
        if (d->inputSize != 20 || memcmp(d->pInput, "ZEMOCKB1", 8) != 0) {
            strcpy(g_log, "not a zemock blob");
            r = ERR_INVALID_NATIVE_BINARY;
        } else {
            memcpy(&g->n, d->pInput + 8, 4);
            memcpy(&g->scale, d->pInput + 12, 4);
            memcpy(&g->bias, d->pInput + 16, 4);
        }
    } else if (d->format == 2) {
        if (!strstr(g_last_flags, "--inputs_precisions=\"0:FP32\"") || !strstr(g_last_flags, "--outputs_layouts=\"0:NC\"")) {
            snprintf(g_log, sizeof g_log, "unexpected build flags: %s", g_last_flags);
            r = ERR_INVALID_ARGUMENT;
        } else {
            r = parse_ir(d->pInput, d->inputSize, g, NULL, 0);
            g->is_ir = 1;
        }
    } else {
        r = ERR_INVALID_ARGUMENT;
    }
    if (r != OK) { free(g); return r; }
    *out = g;
    return OK;
}

static ze_result_t pfnDestroy(graph *g) { free(g); return OK; }

static ze_result_t pfnGetProperties2(graph *g, graph_props2 *p) {
    if (p->stype != 0x3) return ERR_INVALID_ARGUMENT;
    p->numGraphArgs = 2;
    p->initStageRequired = g->is_ir ? 0x2 : 0x1;
    return OK;
}
static ze_result_t pfnGetProperties(graph *g, graph_props2 *p) { (void)g; p->numGraphArgs = 2; return OK; }
static ze_result_t pfnGraphInitialize(graph *g) { g->initialized = 1; return OK; }

static ze_result_t pfnGetArgumentProperties3(graph *g, uint32_t i, arg_props3 *p) {
    if (i > 1 || p->stype != 0x4) return ERR_INVALID_ARGUMENT;
    strcpy(p->name, i == 0 ? "x" : "y");
    p->type = i; /* 0 input, 1 output */
    p->dims[0] = 1;
    p->dims[1] = g->n;
    p->dims_count = 2;
    p->networkPrecision = p->devicePrecision = 0x01; /* FP32 */
    return OK;
}

static ze_result_t pfnSetArgumentValue(graph *g, uint32_t i, const void *v) {
    if (i > 1) return ERR_INVALID_ARGUMENT;
    g->args[i] = (void *)v;
    return OK;
}

static ze_result_t append(cmdlist *l, graph *g, int init) {
    if (l->closed || l->count == 8) return ERR_INVALID_ARGUMENT;
    command *c = &l->cmds[l->count++];
    c->g = g;
    c->init = init;
    c->args[0] = g->args[0];
    c->args[1] = g->args[1];
    return OK;
}
static ze_result_t pfnAppendGraphInitialize(cmdlist *l, graph *g, void *sig, uint32_t n, void **w) { (void)sig; (void)n; (void)w; return append(l, g, 1); }
static ze_result_t pfnAppendGraphExecute(cmdlist *l, graph *g, void *prof, void *sig, uint32_t n, void **w) { (void)prof; (void)sig; (void)n; (void)w; return append(l, g, 0); }

static ze_result_t pfnGetNativeBinary(graph *g, size_t *size, uint8_t *out) {
    *size = 20;
    if (out) { memcpy(out, "ZEMOCKB1", 8); memcpy(out + 8, &g->n, 4); memcpy(out + 12, &g->scale, 4); memcpy(out + 16, &g->bias, 4); }
    return OK;
}

static ze_result_t pfnDeviceGetGraphProperties(void *dev, device_graph_props *p) {
    (void)dev;
    p->ext = MAKE_VERSION(1, 12);
    p->major = 7;
    p->minor = 3;
    p->formats = 0x1 | 0x2;
    p->opset = 13;
    return OK;
}

static ze_result_t pfnQueryNetworkCreate2(void *ctx, void *dev, const graph_desc2 *d, query **q) {
    (void)ctx; (void)dev;
    if (d->stype != 0x2) return ERR_INVALID_ARGUMENT;
    *q = calloc(1, sizeof(query));
    parse_ir(d->pInput, d->inputSize, NULL, (*q)->layers, sizeof (*q)->layers);
    return OK;
}
static ze_result_t pfnQueryNetworkDestroy(query *q) { free(q); return OK; }
static ze_result_t pfnQueryNetworkGetSupportedLayers(query *q, size_t *size, char *out) {
    if (out) memcpy(out, q->layers, *size);
    *size = strlen(q->layers);
    return OK;
}

static ze_result_t pfnBuildLogGetString(void *g, uint32_t *size, char *out) {
    (void)g;
    if (out) {
        memcpy(out, g_log, *size);
    } else {
        *size = g_log[0] ? (uint32_t)strlen(g_log) + 1 : 0;
    }
    return OK;
}

static void *graph_table[29] = {
    [1] = pfnDestroy, [2] = pfnGetProperties, [4] = pfnSetArgumentValue, [5] = pfnAppendGraphInitialize,
    [6] = pfnAppendGraphExecute, [7] = pfnGetNativeBinary, [8] = pfnDeviceGetGraphProperties,
    [11] = pfnGetArgumentProperties3, [13] = pfnQueryNetworkDestroy, [14] = pfnQueryNetworkGetSupportedLayers,
    [15] = pfnBuildLogGetString, [16] = pfnCreate2, [17] = pfnQueryNetworkCreate2,
    [21] = pfnGetProperties2, [22] = pfnGraphInitialize,
};

ze_result_t zeDriverGetExtensionFunctionAddress(void *drv, const char *name, void **fn) {
    (void)drv;
    if (strcmp(name, "ZE_extension_graph") == 0) { *fn = graph_table; return OK; }
    if (strcmp(name, "ZE_extension_command_queue_npu") == 0) { *fn = queue_table; return OK; }
    return ERR_INVALID_ARGUMENT;
}
