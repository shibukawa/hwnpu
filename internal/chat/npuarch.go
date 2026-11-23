package chat

// NPUArch is this Mac's architecture of the Neural Engine (h15g for an M3), "" on another machine
// or where Core AI is not available.
func NPUArch() string { return hubArch() }
