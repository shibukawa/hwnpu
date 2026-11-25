package chat

import (
	"sync"

	"github.com/shibukawa/bonsaiai/internal/backend/coreai"
)

// hubArch is this Mac's architecture of the Neural Engine (h14g), the target of the prebuilt
// programs it reads from the conversions on Hugging Face.
var hubArch = sync.OnceValue(coreai.Arch)
