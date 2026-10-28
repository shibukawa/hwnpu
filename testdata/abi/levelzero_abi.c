/*
 * Prints sizes and member offsets of the Level Zero structs bound by internal/ze, as JSON.
 * The Go test internal/ze/abi_test.go compares them with unsafe.Sizeof/Offsetof.
 *
 * Regenerate (headers from oneapi-src/level-zero and intel/level-zero-npu-extensions):
 *   cc -I <level-zero>/include -I <level-zero-npu-extensions> levelzero_abi.c -o abi && ./abi > levelzero_abi.json
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <ze_api.h>
#include <ze_graph_ext.h>
#include <ze_command_queue_npu_ext.h>

static int first_struct = 1, first_field;

#define BEGIN(T) \
    printf("%s  \"%s\": {\"size\": %zu, \"fields\": {", first_struct ? "" : ",\n", #T, sizeof(T)); \
    first_struct = 0; first_field = 1;
#define F(T, m) \
    printf("%s\"%s\": %zu", first_field ? "" : ", ", #m, offsetof(T, m)); first_field = 0;
#define END() printf("}}");

int main(void) {
    printf("{\n");
    BEGIN(ze_driver_properties_t) F(ze_driver_properties_t, stype) F(ze_driver_properties_t, pNext) F(ze_driver_properties_t, uuid) F(ze_driver_properties_t, driverVersion) END()
    BEGIN(ze_driver_extension_properties_t) F(ze_driver_extension_properties_t, name) F(ze_driver_extension_properties_t, version) END()
    BEGIN(ze_device_properties_t)
        F(ze_device_properties_t, stype) F(ze_device_properties_t, pNext) F(ze_device_properties_t, type)
        F(ze_device_properties_t, vendorId) F(ze_device_properties_t, deviceId) F(ze_device_properties_t, flags)
        F(ze_device_properties_t, subdeviceId) F(ze_device_properties_t, coreClockRate) F(ze_device_properties_t, maxMemAllocSize)
        F(ze_device_properties_t, maxHardwareContexts) F(ze_device_properties_t, maxCommandQueuePriority)
        F(ze_device_properties_t, numThreadsPerEU) F(ze_device_properties_t, physicalEUSimdWidth)
        F(ze_device_properties_t, numEUsPerSubslice) F(ze_device_properties_t, numSubslicesPerSlice)
        F(ze_device_properties_t, numSlices) F(ze_device_properties_t, timerResolution)
        F(ze_device_properties_t, timestampValidBits) F(ze_device_properties_t, kernelTimestampValidBits)
        F(ze_device_properties_t, uuid) F(ze_device_properties_t, name)
    END()
    BEGIN(ze_context_desc_t) F(ze_context_desc_t, stype) F(ze_context_desc_t, pNext) F(ze_context_desc_t, flags) END()
    BEGIN(ze_host_mem_alloc_desc_t) F(ze_host_mem_alloc_desc_t, stype) F(ze_host_mem_alloc_desc_t, pNext) F(ze_host_mem_alloc_desc_t, flags) END()
    BEGIN(ze_fence_desc_t) F(ze_fence_desc_t, stype) F(ze_fence_desc_t, pNext) F(ze_fence_desc_t, flags) END()
    BEGIN(ze_command_queue_desc_t)
        F(ze_command_queue_desc_t, stype) F(ze_command_queue_desc_t, pNext) F(ze_command_queue_desc_t, ordinal)
        F(ze_command_queue_desc_t, index) F(ze_command_queue_desc_t, flags) F(ze_command_queue_desc_t, mode)
        F(ze_command_queue_desc_t, priority)
    END()
    BEGIN(ze_command_list_desc_t) F(ze_command_list_desc_t, stype) F(ze_command_list_desc_t, pNext) F(ze_command_list_desc_t, commandQueueGroupOrdinal) F(ze_command_list_desc_t, flags) END()
    BEGIN(ze_graph_compiler_version_info_t) F(ze_graph_compiler_version_info_t, major) F(ze_graph_compiler_version_info_t, minor) END()
    BEGIN(ze_device_graph_properties_t)
        F(ze_device_graph_properties_t, stype) F(ze_device_graph_properties_t, pNext) F(ze_device_graph_properties_t, graphExtensionVersion)
        F(ze_device_graph_properties_t, compilerVersion) F(ze_device_graph_properties_t, graphFormatsSupported)
        F(ze_device_graph_properties_t, maxOVOpsetVersionSupported)
    END()
    BEGIN(ze_graph_desc_2_t)
        F(ze_graph_desc_2_t, stype) F(ze_graph_desc_2_t, pNext) F(ze_graph_desc_2_t, format) F(ze_graph_desc_2_t, inputSize)
        F(ze_graph_desc_2_t, pInput) F(ze_graph_desc_2_t, pBuildFlags) F(ze_graph_desc_2_t, flags)
    END()
    BEGIN(ze_graph_properties_t) F(ze_graph_properties_t, stype) F(ze_graph_properties_t, pNext) F(ze_graph_properties_t, numGraphArgs) END()
    BEGIN(ze_graph_properties_2_t) F(ze_graph_properties_2_t, stype) F(ze_graph_properties_2_t, pNext) F(ze_graph_properties_2_t, numGraphArgs) F(ze_graph_properties_2_t, initStageRequired) END()
    BEGIN(ze_graph_argument_properties_3_t)
        F(ze_graph_argument_properties_3_t, stype) F(ze_graph_argument_properties_3_t, pNext) F(ze_graph_argument_properties_3_t, name)
        F(ze_graph_argument_properties_3_t, type) F(ze_graph_argument_properties_3_t, dims)
        F(ze_graph_argument_properties_3_t, networkPrecision) F(ze_graph_argument_properties_3_t, networkLayout)
        F(ze_graph_argument_properties_3_t, devicePrecision) F(ze_graph_argument_properties_3_t, deviceLayout)
        F(ze_graph_argument_properties_3_t, quantReverseScale) F(ze_graph_argument_properties_3_t, quantZeroPoint)
        F(ze_graph_argument_properties_3_t, dims_count) F(ze_graph_argument_properties_3_t, debug_friendly_name)
        F(ze_graph_argument_properties_3_t, associated_tensor_names) F(ze_graph_argument_properties_3_t, associated_tensor_names_count)
    END()
    BEGIN(ze_graph_dditable_ext_t)
        F(ze_graph_dditable_ext_t, pfnCreate) F(ze_graph_dditable_ext_t, pfnDestroy) F(ze_graph_dditable_ext_t, pfnGetProperties)
        F(ze_graph_dditable_ext_t, pfnSetArgumentValue) F(ze_graph_dditable_ext_t, pfnAppendGraphInitialize)
        F(ze_graph_dditable_ext_t, pfnAppendGraphExecute) F(ze_graph_dditable_ext_t, pfnGetNativeBinary)
        F(ze_graph_dditable_ext_t, pfnDeviceGetGraphProperties) F(ze_graph_dditable_ext_t, pfnGetArgumentProperties3)
        F(ze_graph_dditable_ext_t, pfnQueryNetworkDestroy) F(ze_graph_dditable_ext_t, pfnQueryNetworkGetSupportedLayers)
        F(ze_graph_dditable_ext_t, pfnBuildLogGetString) F(ze_graph_dditable_ext_t, pfnCreate2)
        F(ze_graph_dditable_ext_t, pfnQueryNetworkCreate2) F(ze_graph_dditable_ext_t, pfnGetProperties2)
        F(ze_graph_dditable_ext_t, pfnGraphInitialize) F(ze_graph_dditable_ext_t, pfnCreate3)
        F(ze_graph_dditable_ext_t, pfnSetArgumentValue2) F(ze_graph_dditable_ext_t, pfnEvict)
        F(ze_graph_dditable_ext_t, pfnGetArgumentNames)
    END()
    BEGIN(ze_command_queue_group_properties_t)
        F(ze_command_queue_group_properties_t, stype) F(ze_command_queue_group_properties_t, pNext) F(ze_command_queue_group_properties_t, flags)
        F(ze_command_queue_group_properties_t, maxMemoryFillPatternSize) F(ze_command_queue_group_properties_t, numQueues)
    END()
    BEGIN(ze_command_queue_desc_npu_ext_t) F(ze_command_queue_desc_npu_ext_t, stype) F(ze_command_queue_desc_npu_ext_t, pNext) F(ze_command_queue_desc_npu_ext_t, turbo) END()
    BEGIN(ze_command_queue_desc_npu_ext_2_t) F(ze_command_queue_desc_npu_ext_2_t, stype) F(ze_command_queue_desc_npu_ext_2_t, pNext) F(ze_command_queue_desc_npu_ext_2_t, options) END()
    printf("\n}\n");
    return 0;
}
