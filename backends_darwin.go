package hwnpu

import (
	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/backend/coreml"
)

func init() { backend.Register(coreml.Provider{}) }
