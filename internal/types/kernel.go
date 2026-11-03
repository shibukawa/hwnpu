package types

// BufferKind selects the memory a buffer is allocated from.
type BufferKind uint8

const (
	// BufferDefault is host memory shared with the device.
	BufferDefault BufferKind = iota
	// BufferInstructions is memory for NPU instruction streams (control code). On AMD NPUs it
	// comes from the device heap, as XRT does for "cacheable" buffers; elsewhere it is ordinary.
	BufferInstructions
)

// ProgramSource is the input of KernelRunner.Load.
type ProgramSource struct {
	Xclbin []byte // xclbin contents
	ELF    []byte // or a full ELF (AIE2P), as IRON and aiecc write for Strix Point and later
	Path   string // or a path to an .xclbin or full ELF file (told apart by its contents)
}

// KernelArg describes one kernel argument.
type KernelArg struct {
	Index  int
	Name   string
	Type   string // C type from the metadata, e.g. "uint64_t" or "char *"
	Offset int    // byte offset in the command's register map
	Size   int    // byte size in the register map
	Buffer bool   // the argument is a device buffer address
}

// KernelInfo describes a kernel of a loaded program.
type KernelInfo struct {
	Name string
	Args []KernelArg
	CUs  int // compute units instantiating the kernel
}
