package mlpackage

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu/internal/mlspec"
	"google.golang.org/protobuf/proto"
)

// ReadModel reads the model specification (description and program) of an .mlpackage.
func ReadModel(dir string) (*mlspec.Model, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "Data", "com.apple.CoreML", "model.mlmodel"))
	if err != nil {
		return nil, err
	}
	var m mlspec.Model
	if err := proto.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("mlpackage: %s: %w", dir, err)
	}
	return &m, nil
}
