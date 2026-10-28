package xdna

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type cStruct struct {
	Size   uintptr            `json:"size"`
	Fields map[string]uintptr `json:"fields"`
}

func golden(t *testing.T) (map[string]uintptr, map[string]cStruct) {
	t.Helper()
	raw, err := os.ReadFile("../../testdata/abi/xdna_abi.json")
	if err != nil {
		t.Fatal(err)
	}
	var all map[string]json.RawMessage
	if err := json.Unmarshal(raw, &all); err != nil {
		t.Fatal(err)
	}
	var ioctls map[string]uintptr
	json.Unmarshal(all["ioctls"], &ioctls)
	structs := map[string]cStruct{}
	for k, v := range all {
		var s cStruct
		if json.Unmarshal(v, &s) == nil && s.Size != 0 {
			structs[k] = s
		}
	}
	return ioctls, structs
}

// Acceptance requirement:cgo-free/abi-golden (AMD).
func TestIoctlNumbers(t *testing.T) {
	ioctls, _ := golden(t)
	for name, got := range map[string]uintptr{
		"DRM_IOCTL_AMDXDNA_CREATE_HWCTX": IoctlCreateHwctx, "DRM_IOCTL_AMDXDNA_DESTROY_HWCTX": IoctlDestroyHwctx,
		"DRM_IOCTL_AMDXDNA_CONFIG_HWCTX": IoctlConfigHwctx, "DRM_IOCTL_AMDXDNA_CREATE_BO": IoctlCreateBO,
		"DRM_IOCTL_AMDXDNA_GET_BO_INFO": IoctlGetBOInfo, "DRM_IOCTL_AMDXDNA_SYNC_BO": IoctlSyncBO,
		"DRM_IOCTL_AMDXDNA_EXEC_CMD": IoctlExecCmd, "DRM_IOCTL_AMDXDNA_GET_INFO": IoctlGetInfo,
		"DRM_IOCTL_GEM_CLOSE": IoctlGemClose, "DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT": IoctlSyncobjWait,
	} {
		if want, ok := ioctls[name]; !ok || got != want {
			t.Errorf("%s: Go %#x, C %#x", name, got, want)
		}
	}
}

func TestStructLayouts(t *testing.T) {
	_, structs := golden(t)
	cases := []struct {
		c      string
		v      any
		fields map[string]string
	}{
		{"amdxdna_qos_info", QoSInfo{}, map[string]string{"gops": "GOPS", "priority": "Priority"}},
		{"amdxdna_drm_create_hwctx", CreateHwctx{}, map[string]string{"ext": "Ext", "ext_flags": "ExtFlags", "qos_p": "QoS", "umq_bo": "UMQBO",
			"log_buf_bo": "LogBufBO", "max_opc": "MaxOPC", "num_tiles": "NumTiles", "mem_size": "MemSize", "umq_doorbell": "UMQDoorbell",
			"handle": "Handle", "syncobj_handle": "SyncobjHandle"}},
		{"amdxdna_drm_destroy_hwctx", DestroyHwctx{}, map[string]string{"handle": "Handle", "pad": "Pad"}},
		{"amdxdna_cu_config", CUConfig{}, map[string]string{"cu_bo": "CUBO", "cu_func": "CUFunc", "pad": "Pad"}},
		{"amdxdna_hwctx_param_config_cu", ConfigCUHeader{}, map[string]string{"num_cus": "NumCUs", "pad": "Pad", "cu_configs": ""}},
		{"amdxdna_drm_config_hwctx", ConfigHwctx{}, map[string]string{"handle": "Handle", "param_type": "ParamType", "param_val": "ParamVal", "param_val_size": "ParamValSize", "pad": "Pad"}},
		{"amdxdna_drm_create_bo", CreateBO{}, map[string]string{"flags": "Flags", "vaddr": "VAddr", "size": "Size", "type": "Type", "handle": "Handle"}},
		{"amdxdna_drm_get_bo_info", GetBOInfo{}, map[string]string{"ext": "Ext", "ext_flags": "ExtFlags", "handle": "Handle", "pad": "Pad", "map_offset": "MapOffset", "vaddr": "VAddr", "xdna_addr": "XdnaAddr"}},
		{"amdxdna_drm_sync_bo", SyncBO{}, map[string]string{"handle": "Handle", "direction": "Direction", "offset": "Offset", "size": "Size"}},
		{"amdxdna_drm_exec_cmd", ExecCmd{}, map[string]string{"ext": "Ext", "ext_flags": "ExtFlags", "hwctx": "Hwctx", "type": "Type", "cmd_handles": "CmdHandles", "args": "Args", "cmd_count": "CmdCount", "arg_count": "ArgCount", "seq": "Seq"}},
		{"amdxdna_drm_query_aie_metadata", AIEMetadata{}, map[string]string{"col_size": "ColSize", "cols": "Cols", "rows": "Rows", "version": "VersionMajor", "core": "Core", "mem": "Mem", "shim": "Shim"}},
		{"amdxdna_drm_query_sensor", Sensor{}, map[string]string{"label": "Label", "input": "Input", "max": "Max", "average": "Average", "highest": "Highest", "status": "Status", "units": "Units", "unitm": "UnitM", "type": "Type", "pad": "Pad"}},
		{"amdxdna_drm_query_firmware_version", FirmwareVersion{}, map[string]string{"major": "Major", "build": "Build"}},
		{"amdxdna_drm_get_info", GetInfo{}, map[string]string{"param": "Param", "buffer_size": "BufferSize", "buffer": "Buffer"}},
		{"drm_gem_close", GemClose{}, map[string]string{"handle": "Handle", "pad": "Pad"}},
		{"drm_syncobj_timeline_wait", SyncobjTimelineWait{}, map[string]string{"handles": "Handles", "points": "Points", "timeout_nsec": "TimeoutNsec", "count_handles": "CountHandles", "flags": "Flags", "first_signaled": "FirstSignaled", "pad": "Pad", "deadline_nsec": "DeadlineNsec"}},
	}
	for _, c := range cases {
		g, ok := structs[c.c]
		if !ok {
			t.Errorf("%s missing from golden file", c.c)
			continue
		}
		typ := reflect.TypeOf(c.v)
		if typ.Size() != g.Size {
			t.Errorf("%s: Go size %d, C size %d", c.c, typ.Size(), g.Size)
		}
		for cName, off := range g.Fields {
			goName, ok := c.fields[cName]
			if !ok {
				t.Errorf("%s.%s has no Go mapping", c.c, cName)
				continue
			}
			if goName == "" { // flexible array member: starts right after the header
				if typ.Size() != off {
					t.Errorf("%s.%s: Go header size %d, C offset %d", c.c, cName, typ.Size(), off)
				}
				continue
			}
			f, ok := typ.FieldByName(goName)
			if !ok {
				t.Errorf("%s: Go field %s not found", c.c, goName)
				continue
			}
			if f.Offset != off {
				t.Errorf("%s.%s: Go offset %d, C offset %d", c.c, cName, f.Offset, off)
			}
		}
	}
}
