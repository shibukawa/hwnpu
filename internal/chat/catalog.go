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
	KindChat       Kind = "chat"        // generates text: the chat completions API
	KindEmbed      Kind = "embedding"   // turns a text into a vector: the embeddings API
	KindTranscribe Kind = "transcribe"  // speech to text: hwnpu-audio transcribe
	KindSpeech     Kind = "speech"      // text to speech: hwnpu-audio tts
	KindDecision   Kind = "decision"    // answers about a state: hwnpu-chat systemone
	KindImage      Kind = "image"       // draws a picture from a prompt: hwnpu-image generate
	KindDepth      Kind = "depth"       // the depth of a picture: hwnpu-image depth
	KindFace       Kind = "face-detect" // the faces in a picture: hwnpu-image face-detect
	KindPitch      Kind = "pitch"       // the pitch of a voice or an instrument: hwnpu-audio pitch
	KindPose       Kind = "pose"        // people's keypoints in an image: hwnpu-image pose
	KindDetect     Kind = "detect"      // the objects in an image, boxes and labels: hwnpu-image detect
	KindOCR        Kind = "ocr"         // the text in an image, lines and boxes: hwnpu-image ocr
	KindLabel      Kind = "label"       // labels of an image and embeddings of images and texts: hwnpu-image label, embed
	KindSegment    Kind = "segment"     // the mask of a thing a point or a box picks: hwnpu-image segment
	KindNotes      Kind = "notes"       // the notes of audio, as MIDI: hwnpu-audio notes
	KindMusic      Kind = "music"       // music from a style: hwnpu-audio generate
	KindSFX        Kind = "sfx"         // sound effects from a text: hwnpu-audio generate -m sfx
)

// Model is a model of the definitions (models.yaml).
type Model struct {
	Name     string // what -use and the API call it
	Kind     Kind
	Features []string // what it can do besides its kind, from Features
	Repo     string   // the Hugging Face repository it is downloaded from
	Revision string   // the repository's revision to download ("": main)
	// SourceName is the source of sources the model comes from, "" for hf (Hugging Face, or the
	// sources of Search); Repo is then the model's folder in it.
	SourceName string
	// SHA256 pins the repository's files by name.
	SHA256 map[string]string
	// Arch is the structure the definitions name (llm.Lookup), "" for the one config.json says.
	Arch string
	// Files are the definition files that wrote the model (ModelFiles).
	Files []string
	Bytes float64 // the size of the download (0: unknown), for a vendor BytesBy does not name (BytesFor)
	// BytesBy is the size of the download by the NPU's vendor (PerVendor).
	BytesBy PerVendor
	About   string
	// NoThink is the model's own default of answering without thinking first, which -think
	// given on the command line, reasoning_effort or enable_thinking override (Qwen3.5's 0.8B
	// and 2B: their chat templates think only when asked, and the 0.8B thinks to the budget on
	// simple questions, docs/design.md 16.99).
	NoThink bool
	// Device is where the model runs by the vendor of the machine's NPU (Vendors: "amd",
	// "apple", "intel"), Devices: "npu" or "cpu"; a vendor it leaves out runs it on the NPU
	// (DeviceFor).
	Device map[string]string
	// Family and Tag name the model as FAMILY:TAG (plamo3translate:2b). The family alone picks
	// the tag for the machine's memory: the one of the largest Machine that the machine has.
	Family, Tag string
	// Machine is the least memory of a machine, in GB, that the model is recommended for when its
	// family is named without a tag (MODEL.md's columns: 8, 16, 24, 32, 48, 64).
	Machine int
	// Memory is what the model takes running, in GB: its NPU buffers and the process after a short
	// conversation (MODEL.md, "at start"), for a vendor MemoryBy does not name. MemoryBy has it by
	// the NPU's vendor (MemoryFor).
	Memory   float64
	MemoryBy PerVendor
	// Run is how hwnpu-chat runs a chat model when its flags do not say (models.yaml's prefill,
	// format, context and paged), on a machine of Machine GB.
	Run Run
	// Variants change Memory and Run on machines of more memory, by the least memory (GB) of each
	// (models.yaml's variants): On is the model as a machine runs it. Lookup and Catalog have it
	// as it is; Resolve, ResolveFor and Pick give it On the machine.
	Variants map[int]Variant
	// AMDExport is how tools/amd/iron_export_lm.py exports the model for AMD's NPU when its
	// export is not here: the exporter's model and options (PrepareAMD); none for a model AMD's
	// NPU has no form of.
	AMDExport []string
	// File is the model's checkpoint file in a repository that holds several models (Anima's
	// diffusion models: anima-turbo-v1.1.safetensors), "" when the repository is the one model.
	File string
	// With are the other repositories the model reads, by role (an OCR's detector): OWNER/NAME
	// or OWNER/NAME@REVISION as fetch takes them. Their files' sha256 are SHA256's ROLE/FILE.
	With map[string]string
	// NotOn are the chips (Chips) a family or a category does not pick the model on: it is too
	// slow there (PLaMo 3 Translate 8B on an M1, over the 2.8 GiB its Neural Engine runs at full
	// speed). Its name and FAMILY:TAG still run it.
	NotOn []string
	// License is the weights' license (models.yaml's license): its id, where it is, and whether
	// the user accepts it before the first download or run (AcceptLicense).
	License License
}

// DirName is the model's directory in the models' directory: its repository's name, so that
// models of one repository (LFM3-Audio-2B and its speech) share the download.
func (m Model) DirName() string {
	if m.Repo == "" {
		return m.Name
	}
	return path.Base(m.Repo)
}

// Source is the repository as fetch takes it: OWNER/NAME, or OWNER/NAME@REVISION when pinned,
// from hf (or the sources of Search), or SOURCE:NAME from a source of sources.
func (m Model) Source() string {
	switch {
	case m.SourceName != "":
		return m.SourceName + ":" + m.Repo
	case m.Revision != "":
		return m.Repo + "@" + m.Revision
	}
	return m.Repo
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
	m, ok = repoModel(name)
	if !ok {
		return Model{}, errNoModel(name)
	}
	dir := filepath.Join(root, m.DirName())
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return m, err
	}
	path := filepath.Join(dir, "config.json")
	if err := fetch(ctx, m.Source(), "config.json", path, log); err != nil {
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
// a family or a use (Pick, logged), each through the aliases, On the machine. ok is false for
// none.
func lookupFor(name string, log io.Writer) (Model, bool, error) {
	if m, ok := Lookup(name); ok {
		return Alias(m, log).On(MachineMemory()), true, nil
	}
	m, why, ok, err := Pick(name, MachineMemory())
	if err != nil || !ok {
		return Model{}, false, err
	}
	if log != nil {
		fmt.Fprintf(log, "%s: %s, %s (about %.1f GB running)\n", name, m.Name, why, m.MemoryFor(HereVendor()))
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
	return ResolveForKinds(name, log, kind)
}

// ResolveForKinds is ResolveFor for a command that runs models of several kinds (hwnpu-audio:
// music and sound effects); a repository is taken to be of the first.
func ResolveForKinds(name string, log io.Writer, kinds ...Kind) (Model, error) {
	kind := kinds[0]
	m, ok, err := lookupFor(name, log)
	if err != nil {
		return Model{}, err
	}
	if ok {
		return m, kindIs(m, kinds...)
	}
	m, ok = repoModel(name)
	if !ok {
		return Model{}, errNoModel(name)
	}
	m.Kind = kind
	return m, nil
}

// NoModelError is the error of a name that is no model, family or use of the definitions nor a
// repository.
type NoModelError struct{ Name string }

func (e *NoModelError) Error() string {
	return fmt.Sprintf("chat: no model %q (hwnpu-chat list shows the models, their families and the uses that pick one; a Hugging Face repository is written OWNER/NAME, a model of a source of the definitions SOURCE:NAME)", e.Name)
}

func errNoModel(name string) error { return &NoModelError{Name: name} }

// repoModel is the model a name of a repository means: OWNER/NAME or OWNER/NAME@REVISION of
// Hugging Face (or the sources of Search), or SOURCE:NAME, a model of a source of the
// definitions. ok is false for neither.
func repoModel(name string) (Model, bool) {
	if src, rest := splitSource(name); src != "" {
		if !safeName(rest) || strings.Contains(rest, "@") {
			return Model{}, false
		}
		return Model{Name: path.Base(rest), Repo: rest, SourceName: src}, true
	}
	repo, rev, _ := strings.Cut(name, "@")
	if strings.Count(repo, "/") != 1 || !safeName(repo) || strings.HasSuffix(repo, "/") {
		return Model{}, false
	}
	return Model{Name: path.Base(repo), Repo: repo, Revision: rev}, true
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
