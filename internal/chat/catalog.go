package chat

import (
	"context"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"

	"github.com/shibukawa/hwnpu/internal/llm"
)

// Kind is what a model is for.
type Kind string

const (
	KindChat       Kind = "chat"       // generates text: the chat completions API
	KindEmbed      Kind = "embedding"  // turns a text into a vector: the embeddings API
	KindTranscribe Kind = "transcribe" // speech to text: hwnpu-transcribe
	KindSpeech     Kind = "speech"     // text to speech: hwnpu-tts
	KindDecision   Kind = "decision"   // answers about a state: hwnpu-systemone
)

// Model is a model of the definitions (models.yaml).
type Model struct {
	Name     string // what -use and the API call it
	Kind     Kind
	Features []string // what it can do besides its kind, from Features
	Repo     string   // the Hugging Face repository it is downloaded from
	Revision string   // the repository's revision to download ("": main)
	// Arch is the structure the definitions name (llm.Lookup), "" for the one config.json says.
	Arch string
	// Files are the definition files that wrote the model (ModelFiles).
	Files []string
	Bytes float64 // the size of the download (0: unknown)
	About string
	// NoThink is the model's own default of answering without thinking first, which -think
	// given on the command line, reasoning_effort or enable_thinking override (Qwen3.5's 0.8B
	// and 2B: their chat templates think only when asked, and the 0.8B thinks to the budget on
	// simple questions, docs/design.md 16.99).
	NoThink bool
	// Family and Tag name the model as FAMILY:TAG (plamo3translate:2b). The family alone picks
	// the tag for the machine's memory: the one of the largest Machine that the machine has.
	Family, Tag string
	// Machine is the least memory of a machine, in GB, that the model is recommended for when its
	// family is named without a tag (MODEL.md's columns: 8, 16, 24, 32, 48, 64).
	Machine int
	// Memory is what the model takes running, in GB: its NPU buffers and the process after a short
	// conversation (MODEL.md, "at start").
	Memory float64
}

// DirName is the model's directory in the models' directory: its repository's name, so that
// models of one repository (LFM3-Audio-2B and its speech) share the download.
func (m Model) DirName() string {
	if m.Repo == "" {
		return m.Name
	}
	return path.Base(m.Repo)
}

// Source is the repository as fetch takes it: OWNER/NAME, or OWNER/NAME@REVISION when pinned.
func (m Model) Source() string {
	if m.Revision == "" {
		return m.Repo
	}
	return m.Repo + "@" + m.Revision
}

// Features explains the features a model of the catalog can have. The names are those of
// Ollama's capabilities where it has one.
var Features = []struct{ Name, About string }{
	{"tools", "calls the tools a request offers (tools, tool_calls), so that agents such as OpenCode can drive it"},
	{"thinking", "thinks before it answers (reasoning_content), which -think and reasoning_effort turn on and off"},
	{"query", "embeds search queries and the documents searched differently (input_type)"},
	{"vision", "reads the images of user messages (image_url; /image in the terminal)"},
	{"audio", "reads the audio of user messages (input_audio, WAV; /audio in the terminal)"},
	{"translation", "translates the text of each user message (chat_template_kwargs target_lang, source_lang, form, context, glossary; -to in the terminal)"},
}

// Catalog lists the models hwnpu knows, by name, in the order of the definition files
// (models.yaml, LoadModels).
var Catalog []Model

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

// Resolve returns the model a name of -use means: one of the catalog by its name; a family or a
// category of the catalog, with a tag or picked for the machine's memory (Pick: plamo3translate,
// plamo3translate:2b, translate); or a Hugging Face repository (OWNER/NAME) of an architecture
// hwnpu-chat runs, whose config.json it downloads to root/NAME to find out what the model is for.
// What else such a model can do is not known. Only the model picked is downloaded.
func Resolve(ctx context.Context, name, root string, log io.Writer) (Model, error) {
	m, ok, err := lookupFor(name, log)
	if err != nil {
		return Model{}, err
	}
	if ok {
		return m, kindIs(m, KindChat, KindEmbed)
	}
	if strings.Count(name, "/") != 1 {
		return Model{}, fmt.Errorf("chat: no model %q (-list shows the models, their families and the uses that pick one; a Hugging Face repository is written OWNER/NAME)", name)
	}
	m = Model{Name: filepath.Base(name), Repo: name}
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

// lookupFor finds a model of the definitions a name means: by its name (or repository), or as
// a family or a use (Pick, logged), each through the aliases. ok is false for none.
func lookupFor(name string, log io.Writer) (Model, bool, error) {
	if m, ok := Lookup(name); ok {
		return Alias(m, log), true, nil
	}
	m, why, ok, err := Pick(name, MachineMemory())
	if err != nil || !ok {
		return Model{}, false, err
	}
	if log != nil {
		fmt.Fprintf(log, "%s: %s, %s (about %.1f GB running)\n", name, m.Name, why, m.Memory)
	}
	return m, true, nil
}

// kindIs is nil when m is of one of kinds, else an error that names the command that runs it.
func kindIs(m Model, ks ...Kind) error {
	if slices.Contains(ks, m.Kind) {
		return nil
	}
	return fmt.Errorf("chat: %s is a %s model, which %s runs", m.Name, m.Kind, kinds[m.Kind])
}

// ResolveFor returns the model a name means for the command that runs models of kind: one of
// the definitions by its name, a use or a family (with a tag or picked for the machine's
// memory), or a Hugging Face repository OWNER/NAME, taken to be of that kind.
func ResolveFor(name string, kind Kind, log io.Writer) (Model, error) {
	m, ok, err := lookupFor(name, log)
	if err != nil {
		return Model{}, err
	}
	if ok {
		return m, kindIs(m, kind)
	}
	if strings.Count(name, "/") != 1 {
		return Model{}, fmt.Errorf("chat: no model %q (-list shows the models, their families and the uses that pick one; a Hugging Face repository is written OWNER/NAME)", name)
	}
	repo, rev, _ := strings.Cut(name, "@")
	return Model{Name: path.Base(repo), Kind: kind, Repo: repo, Revision: rev}, nil
}

// Downloaded reports whether the model is in dir: its tables (TablesFile, written once its
// programs are converted, which then run without the checkpoint), model.safetensors, or the
// index of a checkpoint in several files and all the files it names.
func Downloaded(dir string) bool {
	if exists(filepath.Join(dir, TablesFile)) || exists(filepath.Join(dir, "model.safetensors")) {
		return true
	}
	files, err := shards(filepath.Join(dir, "model.safetensors.index.json"))
	if err != nil {
		return false
	}
	for _, f := range files {
		if _, err := os.Stat(filepath.Join(dir, f)); err != nil {
			return false
		}
	}
	return true
}

// PromptStore is the directory of the prompt store (Config.PromptDir) for the -prompt-cache
// flag: dir as given, prompts in the compile cache when empty, none for "off".
func PromptStore(dir, cacheDir string) string {
	switch dir {
	case "off":
		return ""
	case "":
		if cacheDir == "" {
			return ""
		}
		return filepath.Join(cacheDir, "prompts")
	}
	return dir
}
