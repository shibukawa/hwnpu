package mlpackage

import (
	"crypto/rand"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu/internal/milspec"
	"github.com/shibukawa/hwnpu/internal/mlspec"
	"google.golang.org/protobuf/proto"
)

// Feature describes one model input or output: a multi array of fp16, fp32 or int32.
type Feature struct {
	Name  string
	DType milspec.DataType
	Shape []int
}

// Package is an .mlpackage under construction.
type Package struct {
	Dir   string
	Blobs *BlobWriter
	// Metadata goes into the model description's user defined metadata.
	Metadata map[string]string
	// Description is the model's short description.
	Description string
}

// Create makes the package directory and opens its weight file. The directory is replaced.
func Create(dir string) (*Package, error) {
	if err := os.RemoveAll(dir); err != nil {
		return nil, err
	}
	if err := os.MkdirAll(filepath.Join(dir, "Data", "com.apple.CoreML", "weights"), 0o755); err != nil {
		return nil, err
	}
	w, err := NewBlobWriter(filepath.Join(dir, "Data", "com.apple.CoreML", "weights", "weight.bin"))
	if err != nil {
		return nil, err
	}
	return &Package{Dir: dir, Blobs: w, Metadata: map[string]string{}}, nil
}

func featureType(f Feature) (*mlspec.FeatureType, error) {
	var dt mlspec.ArrayFeatureType_ArrayDataType
	switch f.DType {
	case Float16:
		dt = mlspec.ArrayFeatureType_FLOAT16
	case Float32:
		dt = mlspec.ArrayFeatureType_FLOAT32
	case Int32:
		dt = mlspec.ArrayFeatureType_INT32
	default:
		return nil, fmt.Errorf("mlpackage: feature %s: unsupported type %v", f.Name, f.DType)
	}
	shape := make([]int64, len(f.Shape))
	for i, d := range f.Shape {
		shape[i] = int64(d)
	}
	return &mlspec.FeatureType{Type: &mlspec.FeatureType_MultiArrayType{MultiArrayType: &mlspec.ArrayFeatureType{Shape: shape, DataType: dt}}}, nil
}

// Finish writes the model file and manifest and closes the weight file. specVersion is the Core ML
// specification version (9 for the CoreML8 opset, iOS 18 / macOS 15).
func (p *Package) Finish(prog *milspec.Program, specVersion int32, inputs, outputs []Feature) error {
	if err := p.Blobs.Close(); err != nil {
		return err
	}
	desc := &mlspec.ModelDescription{Metadata: &mlspec.Metadata{ShortDescription: p.Description, UserDefined: p.Metadata}}
	for _, f := range inputs {
		t, err := featureType(f)
		if err != nil {
			return err
		}
		desc.Input = append(desc.Input, &mlspec.FeatureDescription{Name: f.Name, Type: t})
	}
	for _, f := range outputs {
		t, err := featureType(f)
		if err != nil {
			return err
		}
		desc.Output = append(desc.Output, &mlspec.FeatureDescription{Name: f.Name, Type: t})
	}
	m := &mlspec.Model{SpecificationVersion: specVersion, Description: desc, Type: &mlspec.Model_MlProgram{MlProgram: prog}}
	raw, err := proto.Marshal(m)
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(p.Dir, "Data", "com.apple.CoreML", "model.mlmodel"), raw, 0o644); err != nil {
		return err
	}
	modelID, weightsID := uuid(), uuid()
	manifest := map[string]any{
		"fileFormatVersion": "1.0.0",
		"itemInfoEntries": map[string]any{
			modelID:   map[string]string{"author": "com.apple.CoreML", "description": "CoreML Model Specification", "name": "model.mlmodel", "path": "com.apple.CoreML/model.mlmodel"},
			weightsID: map[string]string{"author": "com.apple.CoreML", "description": "CoreML Model Weights", "name": "weights", "path": "com.apple.CoreML/weights"},
		},
		"rootModelIdentifier": modelID,
	}
	js, _ := json.MarshalIndent(manifest, "", "    ")
	return os.WriteFile(filepath.Join(p.Dir, "Manifest.json"), js, 0o644)
}

func uuid() string {
	var b [16]byte
	rand.Read(b[:])
	b[6] = b[6]&0x0f | 0x40
	b[8] = b[8]&0x3f | 0x80
	return fmt.Sprintf("%X-%X-%X-%X-%X", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}
