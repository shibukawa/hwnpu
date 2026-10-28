package ze

import (
	"structs"
	"unsafe"

	"github.com/shibukawa/hwnpu/internal/dl"
)

// QueueExtName is ZE_COMMAND_QUEUE_NPU_EXT_NAME.
const QueueExtName = "ZE_extension_command_queue_npu"

// NPU command queue extension constants.
const (
	stypeQueueDescNPUExt  = 0x1
	stypeQueueDescNPUExt2 = 0x2
	QueueOptionTurbo      = 1 << 0
	WorkloadDefault       = 0x0
	WorkloadBackground    = 0x1
)

// QueueDescNPUExt is ze_command_queue_desc_npu_ext_t (version 1.0).
type QueueDescNPUExt struct {
	_     structs.HostLayout
	SType uint32
	PNext uintptr
	Turbo bool
}

// QueueDescNPUExt2 is ze_command_queue_desc_npu_ext_2_t (version 1.1).
type QueueDescNPUExt2 struct {
	_       structs.HostLayout
	SType   uint32
	PNext   uintptr
	Options uint32
}

// QueueExt is the bound NPU command queue extension.
type QueueExt struct {
	Name    string
	Version uint32

	pfnSetWorkloadType func(queue Handle, workload uint32) Result
}

// LoadQueueExt binds the NPU command queue extension, or returns nil when the driver lacks it.
func (a *API) LoadQueueExt(driver Handle) (*QueueExt, error) {
	props, err := a.Extensions(driver)
	if err != nil {
		return nil, err
	}
	var version uint32
	for _, p := range props {
		if CString(p.Name[:]) == QueueExtName && p.Version>>16 == 1 {
			version = min(p.Version, MakeVersion(1, 1))
		}
	}
	if version == 0 {
		return nil, nil
	}
	table, err := a.ExtensionFunction(driver, QueueExtName)
	if err != nil {
		return nil, err
	}
	q := &QueueExt{Name: QueueExtName, Version: version}
	if fn := *(*uintptr)(table); fn != 0 {
		if err := dl.BindAddr(&q.pfnSetWorkloadType, fn); err != nil {
			return nil, err
		}
	}
	return q, nil
}

// Desc returns an extension descriptor requesting turbo, to chain into ze_command_queue_desc_t.
// The returned value must stay alive until the queue is created.
func (q *QueueExt) Desc(turbo bool) (pNext uintptr, keep any) {
	if q.Version >= MakeVersion(1, 1) {
		d := &QueueDescNPUExt2{SType: stypeQueueDescNPUExt2}
		if turbo {
			d.Options = QueueOptionTurbo
		}
		return uintptr(unsafe.Pointer(d)), d
	}
	d := &QueueDescNPUExt{SType: stypeQueueDescNPUExt, Turbo: turbo}
	return uintptr(unsafe.Pointer(d)), d
}

// SetWorkloadType marks a queue's work as default or background.
func (q *QueueExt) SetWorkloadType(queue Handle, workload uint32) error {
	if q.pfnSetWorkloadType == nil {
		return nil
	}
	return check("pfnSetWorkloadType", q.pfnSetWorkloadType(queue, workload))
}
