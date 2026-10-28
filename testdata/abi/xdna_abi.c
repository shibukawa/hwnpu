/*
 * Prints ioctl numbers, struct layouts and ERT header encodings used by the AMD backend, as JSON.
 * Go tests in internal/xdna, internal/xclbin and internal/ert compare against xdna_abi.json.
 *
 * Headers: include/uapi/drm/{amdxdna_accel,drm,drm_mode}.h from Linux, and
 * src/runtime_src/core/include/xrt/detail/{xclbin,ert}.h from XRT. xclbin.h needs <uuid/uuid.h>;
 * a stub defining uuid_t is enough. The raw kernel headers keep __user annotations, so define it
 * away. Build on Linux:
 *   cc -D__user= -I <dir with the headers> -I <dir with uuid/uuid.h> xdna_abi.c -o abi && ./abi > xdna_abi.json
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include "amdxdna_accel.h"
#include "xclbin.h"
#include "ert.h"

static int first_struct = 1, first_field;

#define BEGIN(T) \
    printf("%s  \"%s\": {\"size\": %zu, \"fields\": {", first_struct ? "" : ",\n", #T, sizeof(struct T)); \
    first_struct = 0; first_field = 1;
#define F(T, m) printf("%s\"%s\": %zu", first_field ? "" : ", ", #m, offsetof(struct T, m)); first_field = 0;
#define END() printf("}}");
#define IOCTL(name) printf("%s\"%s\": %lu", first_field ? "" : ", ", #name, (unsigned long)(name)); first_field = 0;

static unsigned header(int state, int extra, int count, int opcode, int type) {
    struct ert_start_kernel_cmd c = {0};
    c.state = state;
    c.extra_cu_masks = extra;
    c.count = count;
    c.opcode = opcode;
    c.type = type;
    return c.header;
}

int main(void) {
    printf("{\n");
    printf("  \"ioctls\": {"); first_field = 1;
    IOCTL(DRM_IOCTL_AMDXDNA_CREATE_HWCTX) IOCTL(DRM_IOCTL_AMDXDNA_DESTROY_HWCTX) IOCTL(DRM_IOCTL_AMDXDNA_CONFIG_HWCTX)
    IOCTL(DRM_IOCTL_AMDXDNA_CREATE_BO) IOCTL(DRM_IOCTL_AMDXDNA_GET_BO_INFO) IOCTL(DRM_IOCTL_AMDXDNA_SYNC_BO)
    IOCTL(DRM_IOCTL_AMDXDNA_EXEC_CMD) IOCTL(DRM_IOCTL_AMDXDNA_GET_INFO) IOCTL(DRM_IOCTL_AMDXDNA_SET_STATE)
    IOCTL(DRM_IOCTL_AMDXDNA_WAIT_CMD) IOCTL(DRM_IOCTL_GEM_CLOSE) IOCTL(DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT)
    printf("},\n  \"ert_headers\": [");
    printf("%u, %u, %u, %u]", header(1, 0, 5, ERT_START_CU, ERT_CU), header(4, 3, 2047, 31, 15),
           header(ERT_CMD_STATE_COMPLETED, 0, 1, ERT_START_NPU, ERT_CU), header(0, 0, 0, 0, 0));
    first_struct = 0;
    BEGIN(amdxdna_qos_info) F(amdxdna_qos_info, gops) F(amdxdna_qos_info, priority) END()
    BEGIN(amdxdna_drm_create_hwctx)
        F(amdxdna_drm_create_hwctx, ext) F(amdxdna_drm_create_hwctx, ext_flags) F(amdxdna_drm_create_hwctx, qos_p)
        F(amdxdna_drm_create_hwctx, umq_bo) F(amdxdna_drm_create_hwctx, log_buf_bo) F(amdxdna_drm_create_hwctx, max_opc)
        F(amdxdna_drm_create_hwctx, num_tiles) F(amdxdna_drm_create_hwctx, mem_size) F(amdxdna_drm_create_hwctx, umq_doorbell)
        F(amdxdna_drm_create_hwctx, handle) F(amdxdna_drm_create_hwctx, syncobj_handle)
    END()
    BEGIN(amdxdna_drm_destroy_hwctx) F(amdxdna_drm_destroy_hwctx, handle) F(amdxdna_drm_destroy_hwctx, pad) END()
    BEGIN(amdxdna_cu_config) F(amdxdna_cu_config, cu_bo) F(amdxdna_cu_config, cu_func) F(amdxdna_cu_config, pad) END()
    BEGIN(amdxdna_hwctx_param_config_cu) F(amdxdna_hwctx_param_config_cu, num_cus) F(amdxdna_hwctx_param_config_cu, pad) F(amdxdna_hwctx_param_config_cu, cu_configs) END()
    BEGIN(amdxdna_drm_config_hwctx)
        F(amdxdna_drm_config_hwctx, handle) F(amdxdna_drm_config_hwctx, param_type) F(amdxdna_drm_config_hwctx, param_val)
        F(amdxdna_drm_config_hwctx, param_val_size) F(amdxdna_drm_config_hwctx, pad)
    END()
    BEGIN(amdxdna_drm_create_bo) F(amdxdna_drm_create_bo, flags) F(amdxdna_drm_create_bo, vaddr) F(amdxdna_drm_create_bo, size) F(amdxdna_drm_create_bo, type) F(amdxdna_drm_create_bo, handle) END()
    BEGIN(amdxdna_drm_get_bo_info)
        F(amdxdna_drm_get_bo_info, ext) F(amdxdna_drm_get_bo_info, ext_flags) F(amdxdna_drm_get_bo_info, handle) F(amdxdna_drm_get_bo_info, pad)
        F(amdxdna_drm_get_bo_info, map_offset) F(amdxdna_drm_get_bo_info, vaddr) F(amdxdna_drm_get_bo_info, xdna_addr)
    END()
    BEGIN(amdxdna_drm_sync_bo) F(amdxdna_drm_sync_bo, handle) F(amdxdna_drm_sync_bo, direction) F(amdxdna_drm_sync_bo, offset) F(amdxdna_drm_sync_bo, size) END()
    BEGIN(amdxdna_drm_exec_cmd)
        F(amdxdna_drm_exec_cmd, ext) F(amdxdna_drm_exec_cmd, ext_flags) F(amdxdna_drm_exec_cmd, hwctx) F(amdxdna_drm_exec_cmd, type)
        F(amdxdna_drm_exec_cmd, cmd_handles) F(amdxdna_drm_exec_cmd, args) F(amdxdna_drm_exec_cmd, cmd_count) F(amdxdna_drm_exec_cmd, arg_count) F(amdxdna_drm_exec_cmd, seq)
    END()
    BEGIN(amdxdna_drm_query_aie_metadata)
        F(amdxdna_drm_query_aie_metadata, col_size) F(amdxdna_drm_query_aie_metadata, cols) F(amdxdna_drm_query_aie_metadata, rows)
        F(amdxdna_drm_query_aie_metadata, version) F(amdxdna_drm_query_aie_metadata, core) F(amdxdna_drm_query_aie_metadata, mem) F(amdxdna_drm_query_aie_metadata, shim)
    END()
    BEGIN(amdxdna_drm_query_sensor)
        F(amdxdna_drm_query_sensor, label) F(amdxdna_drm_query_sensor, input) F(amdxdna_drm_query_sensor, max) F(amdxdna_drm_query_sensor, average)
        F(amdxdna_drm_query_sensor, highest) F(amdxdna_drm_query_sensor, status) F(amdxdna_drm_query_sensor, units) F(amdxdna_drm_query_sensor, unitm)
        F(amdxdna_drm_query_sensor, type) F(amdxdna_drm_query_sensor, pad)
    END()
    BEGIN(amdxdna_drm_query_firmware_version) F(amdxdna_drm_query_firmware_version, major) F(amdxdna_drm_query_firmware_version, build) END()
    BEGIN(amdxdna_drm_get_info) F(amdxdna_drm_get_info, param) F(amdxdna_drm_get_info, buffer_size) F(amdxdna_drm_get_info, buffer) END()
    BEGIN(drm_gem_close) F(drm_gem_close, handle) F(drm_gem_close, pad) END()
    BEGIN(drm_syncobj_timeline_wait)
        F(drm_syncobj_timeline_wait, handles) F(drm_syncobj_timeline_wait, points) F(drm_syncobj_timeline_wait, timeout_nsec)
        F(drm_syncobj_timeline_wait, count_handles) F(drm_syncobj_timeline_wait, flags) F(drm_syncobj_timeline_wait, first_signaled)
        F(drm_syncobj_timeline_wait, pad) F(drm_syncobj_timeline_wait, deadline_nsec)
    END()
    BEGIN(axlf) F(axlf, m_magic) F(axlf, m_signature_length) F(axlf, m_keyBlock) F(axlf, m_uniqueId) F(axlf, m_header) F(axlf, m_sections) END()
    BEGIN(axlf_header)
        F(axlf_header, m_length) F(axlf_header, m_versionPatch) F(axlf_header, m_versionMajor) F(axlf_header, m_versionMinor)
        F(axlf_header, m_mode) F(axlf_header, m_interface_uuid) F(axlf_header, m_platformVBNV) F(axlf_header, uuid)
        F(axlf_header, m_debug_bin) F(axlf_header, m_numSections)
    END()
    BEGIN(axlf_section_header) F(axlf_section_header, m_sectionKind) F(axlf_section_header, m_sectionName) F(axlf_section_header, m_sectionOffset) F(axlf_section_header, m_sectionSize) END()
    BEGIN(ip_layout) F(ip_layout, m_count) F(ip_layout, m_ip_data) END()
    BEGIN(ip_data) F(ip_data, m_type) F(ip_data, properties) F(ip_data, m_base_address) F(ip_data, m_name) END()
    BEGIN(array_offset) F(array_offset, size) F(array_offset, offset) END()
    BEGIN(aie_partition_info) F(aie_partition_info, column_width) F(aie_partition_info, start_columns) END()
    BEGIN(aie_pdi) F(aie_pdi, uuid) F(aie_pdi, pdi_image) F(aie_pdi, cdo_groups) END()
    BEGIN(cdo_group) F(cdo_group, mpo_name) F(cdo_group, cdo_type) F(cdo_group, pdi_id) F(cdo_group, dpu_kernel_ids) F(cdo_group, pre_cdo_groups) END()
    BEGIN(aie_partition)
        F(aie_partition, schema_version) F(aie_partition, mpo_name) F(aie_partition, operations_per_cycle) F(aie_partition, info)
        F(aie_partition, aie_pdi) F(aie_partition, kernel_commit_id)
    END()
    BEGIN(ert_start_kernel_cmd) F(ert_start_kernel_cmd, header) F(ert_start_kernel_cmd, cu_mask) F(ert_start_kernel_cmd, data) END()
    printf("\n}\n");
    /* ps_kernel bit positions inside ip_data.properties */
    return 0;
}
