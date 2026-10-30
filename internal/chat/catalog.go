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
	Name     string // what -use and the API call it
	Kind     Kind
	Features []string // what it can do besides its kind, from Features
	Repo     string   // the Hugging Face repository it is downloaded from
	Bytes    float64  // the size of the download (0: unknown)
	About    string
}

// Features explains the features a model of the catalog can have. The names are those of
// Ollama's capabilities where it has one.
var Features = []struct{ Name, About string }{
	{"tools", "calls the tools a request offers (tools, tool_calls), so that agents such as OpenCode can drive it"},
	{"thinking", "thinks before it answers (reasoning_content), which -think and reasoning_effort turn on and off"},
	{"query", "embeds search queries and the documents searched differently (input_type)"},
	{"vision", "reads the images of user messages (image_url; /image in the terminal)"},
	{"audio", "reads the audio of user messages (input_audio, WAV; /audio in the terminal)"},
}

// Catalog lists the models hwnpu-chat knows, by name.
var Catalog = []Model{
	{Name: "LFM3-1.7B", Kind: KindChat, Features: []string{"tools", "thinking"}, Repo: "LiquidAI/LFM3-1.7B", Bytes: 3.5e9,
		About: "Liquid AI's 1.7B model, text only: small and fast enough to run behind an agent such as OpenCode"},
	{Name: "LFM3-VL-2B", Kind: KindChat, Features: []string{"tools", "thinking", "vision"}, Repo: "LiquidAI/LFM3-VL-2B", Bytes: 3.8e9,
		About: "Liquid AI's LFM3 with a vision encoder: its text model in 8 bit tables, images of 512x512"},
	{Name: "LFM3-Audio-2B", Kind: KindChat, Features: []string{"tools", "thinking", "audio"}, Repo: "LiquidAI/LFM3-Audio-2B", Bytes: 4.0e9,
		About: "Liquid AI's LFM3 with an audio encoder: its text model in 8 bit tables, clips up to 30 seconds; answers are text (speech: hwnpu-tts)"},
	{Name: "LFM3-Omni-2B", Kind: KindChat, Features: []string{"tools", "thinking", "vision", "audio"}, Repo: "LiquidAI/LFM3-Omni-2B", Bytes: 4.3e9,
		About: "Liquid AI's LFM3 with both encoders: images of 512x512 and clips up to 30 seconds, answers in text"},
	{Name: "gemma-4-E2B-it", Kind: KindChat, Features: []string{"tools", "thinking"}, Repo: "google/gemma-4-E2B-it", Bytes: 10.2e9,
		About: "Google's Gemma 4 E2B, instruction tuned: its text model in 6 bit tables (the download has the vision and audio parts too, which are not converted)"},
	{Name: "gemma-4-E4B-it", Kind: KindChat, Features: []string{"tools", "thinking"}, Repo: "google/gemma-4-E4B-it", Bytes: 16.0e9,
		About: "Google's Gemma 4 E4B, instruction tuned: larger and slower than E2B, its text model in 6 bit tables (vision and audio not converted)"},
	{Name: "plamo-embedding-1b", Kind: KindEmbed, Features: []string{"query"}, Repo: "pfnet/plamo-embedding-1b", Bytes: 2.1e9,
		About: "Preferred Networks' Japanese sentence embeddings (2048 dimensions): search, clustering, classification"},
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
// root/NAME to find out what the model is for. What else such a model can do is not known.
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
