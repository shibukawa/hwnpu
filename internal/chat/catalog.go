package chat

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu/internal/llm"
)

// Kind is what a model is for.
type Kind string

const (
	KindChat  Kind = "chat"      // generates text: the chat completions API
	KindEmbed Kind = "embedding" // turns a text into a vector: the embeddings API
)

// Model is a model hwnpu-chat runs.
type Model struct {
	Name  string // what -use and the API call it
	Kind  Kind
	Repo  string  // the Hugging Face repository it is downloaded from
	Bytes float64 // the size of the download (0: unknown)
	About string
}

// Catalog lists the models hwnpu-chat knows, by name.
var Catalog = []Model{
	{Name: "LFM3-1.7B", Kind: KindChat, Repo: "LiquidAI/LFM3-1.7B", Bytes: 3.5e9,
		About: "Liquid AI's 1.7B model: chat with thinking and tool calls, for terminals and agents such as OpenCode"},
	{Name: "plamo-embedding-1b", Kind: KindEmbed, Repo: "pfnet/plamo-embedding-1b", Bytes: 2.1e9,
		About: "Preferred Networks' Japanese sentence embeddings (2048 dimensions): search queries and documents, clustering"},
}

// Lookup finds a model of the catalog by its name or its repository, ignoring case.
func Lookup(name string) (Model, bool) {
	for _, m := range Catalog {
		if strings.EqualFold(name, m.Name) || strings.EqualFold(name, m.Repo) {
			return m, true
		}
	}
	return Model{}, false
}

// ModelsDir is where models are kept when not told: hwnpu/models in the user cache directory.
// Each model has a directory of its name there.
func ModelsDir() string {
	d, err := os.UserCacheDir()
	if err != nil {
		d = os.TempDir()
	}
	return filepath.Join(d, "hwnpu", "models")
}

// Resolve returns the model a name of -use means: one of the catalog, or a Hugging Face
// repository (OWNER/NAME) of an architecture hwnpu-chat runs, whose config.json it downloads to
// root/NAME to find out what the model is for.
func Resolve(ctx context.Context, name, root string, log io.Writer) (Model, error) {
	if m, ok := Lookup(name); ok {
		return m, nil
	}
	if strings.Count(name, "/") != 1 {
		return Model{}, fmt.Errorf("chat: no model %q (-list shows them; a Hugging Face repository is written OWNER/NAME)", name)
	}
	m := Model{Name: filepath.Base(name), Repo: name}
	dir := filepath.Join(root, m.Name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return m, err
	}
	path := filepath.Join(dir, "config.json")
	if err := fetch(ctx, name, "config.json", path, log); err != nil {
		return m, err
	}
	a, model, err := llm.LoadConfig(path, "")
	if err != nil {
		return m, err
	}
	switch _, cached := model.(llm.Cached); {
	case a.Name == "plamo-bi":
		m.Kind = KindEmbed
	case cached:
		m.Kind = KindChat
	default:
		return m, fmt.Errorf("chat: %s is a %s, which hwnpu-chat does not run", name, a.Title)
	}
	return m, nil
}

// Downloaded reports whether the checkpoint of the model is in dir.
func Downloaded(dir string) bool {
	_, err := os.Stat(filepath.Join(dir, "model.safetensors"))
	return err == nil
}
