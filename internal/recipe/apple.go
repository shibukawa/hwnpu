package recipe

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"go/parser"
	"go/printer"
	"go/token"
	"maps"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"
)

// Convert are the values a model's Core AI conversion is made with (model-recipes.yaml
// models.NAME.apple.convert): what its code took as constants before, by their names in the
// code's terms (text_group, head_chunk). A value is a number, a string or a list of them.
type Convert map[string]any

// AppleVersion is the version of what model's recipe makes for Core AI (its apple.version, 0
// when it has none): the sources keep its exports as vN (converted/MODEL/coreai/EXPORT/v1), and
// an export made here of another version is made again.
func (f *File) AppleVersion(model string) int {
	if m, ok := f.Models[model]; ok && m.Apple != nil {
		return m.Apple.Version
	}
	return 0
}

// ConvertOf is model's apple.convert (empty when it has none).
func (f *File) ConvertOf(model string) Convert {
	if m, ok := f.Models[model]; ok && m.Apple != nil && m.Apple.Convert != nil {
		return m.Apple.Convert
	}
	return Convert{}
}

// AppleCode are the Go packages model's Core AI exports are written by (directories of the
// module): the file's apple.code (what every conversion writes with), then the model's.
func (f *File) AppleCode(model string) []string {
	out := slices.Clone(f.Apple.Code)
	if m, ok := f.Models[model]; ok && m.Apple != nil {
		for _, c := range m.Apple.Code {
			if !slices.Contains(out, c) {
				out = append(out, c)
			}
		}
	}
	return out
}

// AppleFingerprint is the SHA-256 (hex) of model's apple.convert and finish: what changes its
// exports' programs and files. Not its prebuild rules, archs and jobs (which programs are prebuilt, not what they
// are) or its rewrites (in the names of the exports they change).
func (f *File) AppleFingerprint(model string) string {
	b, _ := json.Marshal(f.ConvertOf(model)) // a map's keys in order
	if fin := f.AppleFinish(model); len(fin) > 0 {
		b, _ = json.Marshal(struct {
			Convert Convert
			Finish  []Step
			Steps   map[string]int
		}{f.ConvertOf(model), fin, finishVersions(fin)})
	}
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

// CodeHash is the SHA-256 (hex) of the Go files (not the tests) of the packages dirs of the
// module at root, without their comments and as gofmt prints them: a change of what the code
// does, not of how it is told. Each package is its directory alone (not those below it); a
// name ending in .go is that file alone (of a package that does more than convert), and !FILE
// leaves a file out (one of a package's that only a run reads, not the conversion).
func CodeHash(root string, dirs []string) (string, error) {
	h := sha256.New()
	skip := map[string]bool{}
	for _, d := range dirs {
		if f, ok := strings.CutPrefix(d, "!"); ok {
			skip[filepath.Join(root, filepath.FromSlash(f))] = true
		}
	}
	for _, d := range dirs {
		if strings.HasPrefix(d, "!") {
			continue
		}
		var names []string
		if strings.HasSuffix(d, ".go") {
			// one file of a package
			names = []string{filepath.Join(root, filepath.FromSlash(d))}
			if _, err := os.Stat(names[0]); err != nil {
				return "", fmt.Errorf("recipe: %s: %w", d, err)
			}
			d = path.Dir(d)
		} else {
			var err error
			if names, err = filepath.Glob(filepath.Join(root, filepath.FromSlash(d), "*.go")); err != nil {
				return "", err
			}
		}
		if len(names) == 0 {
			return "", fmt.Errorf("recipe: %s has no Go files", d)
		}
		slices.Sort(names)
		for _, name := range names {
			if strings.HasSuffix(name, "_test.go") || skip[name] {
				continue
			}
			src, err := os.ReadFile(name)
			if err != nil {
				return "", err
			}
			fset := token.NewFileSet()
			file, err := parser.ParseFile(fset, name, src, parser.SkipObjectResolution)
			if err != nil {
				return "", err
			}
			file.Comments = nil
			var buf bytes.Buffer
			if err := (&printer.Config{Mode: printer.UseSpaces | printer.TabIndent, Tabwidth: 8}).Fprint(&buf, fset, file); err != nil {
				return "", err
			}
			fmt.Fprintf(h, "%s %d\n", path.Join(d, filepath.Base(name)), buf.Len())
			h.Write(buf.Bytes())
		}
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func (c Convert) check(at string) error {
	for _, k := range slices.Sorted(maps.Keys(c)) {
		switch v := c[k].(type) {
		case string, uint64, int64, float64, bool:
		case []any:
			for i, e := range v {
				switch e.(type) {
				case string, uint64, int64, float64:
				default:
					return fmt.Errorf("%s.%s[%d]: %v (a number or a string)", at, k, i, e)
				}
			}
		default:
			return fmt.Errorf("%s.%s: %v (a number, a string or a list of them)", at, k, v)
		}
	}
	return nil
}

// Int is the whole number key of c; an error names model and key when it is missing or not one.
func (c Convert) Int(model, key string) (int, error) {
	v, ok := c[key]
	if !ok {
		return 0, fmt.Errorf("recipe: model-recipes.yaml models.%s.apple.convert has no %s", model, key)
	}
	n, ok := toInt(v)
	if !ok {
		return 0, fmt.Errorf("recipe: model-recipes.yaml models.%s.apple.convert.%s: %v is not a whole number", model, key, v)
	}
	return n, nil
}

// Ints is the list of whole numbers key of c.
func (c Convert) Ints(model, key string) ([]int, error) {
	v, ok := c[key]
	if !ok {
		return nil, fmt.Errorf("recipe: model-recipes.yaml models.%s.apple.convert has no %s", model, key)
	}
	list, ok := v.([]any)
	if !ok {
		return nil, fmt.Errorf("recipe: model-recipes.yaml models.%s.apple.convert.%s: %v is not a list", model, key, v)
	}
	out := make([]int, len(list))
	for i, e := range list {
		if out[i], ok = toInt(e); !ok {
			return nil, fmt.Errorf("recipe: model-recipes.yaml models.%s.apple.convert.%s[%d]: %v is not a whole number", model, key, i, e)
		}
	}
	return out, nil
}

// Strings is the list of strings key of c.
func (c Convert) Strings(model, key string) ([]string, error) {
	v, ok := c[key]
	if !ok {
		return nil, fmt.Errorf("recipe: model-recipes.yaml models.%s.apple.convert has no %s", model, key)
	}
	list, ok := v.([]any)
	if !ok {
		return nil, fmt.Errorf("recipe: model-recipes.yaml models.%s.apple.convert.%s: %v is not a list", model, key, v)
	}
	out := make([]string, len(list))
	for i, e := range list {
		if out[i], ok = e.(string); !ok {
			return nil, fmt.Errorf("recipe: model-recipes.yaml models.%s.apple.convert.%s[%d]: %v is not a string", model, key, i, e)
		}
	}
	return out, nil
}

// Float is the number key of c.
func (c Convert) Float(model, key string) (float64, error) {
	v, ok := c[key]
	if !ok {
		return 0, fmt.Errorf("recipe: model-recipes.yaml models.%s.apple.convert has no %s", model, key)
	}
	switch n := v.(type) {
	case float64:
		return n, nil
	case uint64:
		return float64(n), nil
	case int64:
		return float64(n), nil
	}
	return 0, fmt.Errorf("recipe: model-recipes.yaml models.%s.apple.convert.%s: %v is not a number", model, key, v)
}

// String is the string key of c.
func (c Convert) String(model, key string) (string, error) {
	v, ok := c[key]
	if !ok {
		return "", fmt.Errorf("recipe: model-recipes.yaml models.%s.apple.convert has no %s", model, key)
	}
	s, ok := v.(string)
	if !ok {
		return "", fmt.Errorf("recipe: model-recipes.yaml models.%s.apple.convert.%s: %v is not a string", model, key, v)
	}
	return s, nil
}

func toInt(v any) (int, bool) {
	switch n := v.(type) {
	case uint64:
		return int(n), true
	case int64:
		return int(n), true
	case float64:
		if n == float64(int(n)) {
			return int(n), true
		}
	}
	return 0, false
}
