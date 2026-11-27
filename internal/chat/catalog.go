package chat

import (
	"cmp"
	"context"
	"fmt"
	"github.com/shibukawa/hwnpu/internal/cache"
	"io"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"

	"github.com/shibukawa/hwnpu/internal/llm"
)

// Features explains the features a model of the catalog can have. The names are those of
// Ollama's capabilities where it has one.
var Features = []struct{ Name, About string }{
	{"tools", "calls the tools a request offers (tools, tool_calls), so that agents such as OpenCode can drive it"},
	{"thinking", "thinks before it answers (reasoning_content), which -think and reasoning_effort turn on and off"},
	{"query", "embeds search queries and the documents searched differently (input_type)"},
	{"vision", "reads the images of user messages (image_url; /image in the terminal)"},
	{"audio", "reads the audio of user messages (input_audio, WAV; /audio in the terminal)"},
	{"edit", "edits or composes from reference images (hwnpu-image generate --image)"},
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

// ExpertsUsage is the file of an application's counts of a mixture of experts' experts in the
// model's directory dir (experts.Placement.Usage): experts-usage/KEY.bin.
func ExpertsUsage(dir, key string) string {
	key = strings.Map(func(r rune) rune {
		if r == '/' || r == '\\' || r == ':' || r < ' ' {
			return '_'
		}
		return r
	}, cmp.Or(key, "default"))
	return filepath.Join(dir, "experts-usage", key+".bin")
}

// ModelDir is the directory of model m's converted programs: ModelsDir/NAME (Model.DirName).
// Its files (config.json, the checkpoint) are in RepoDir.
func ModelDir(m Model) string {
	d := modelDir(m)
	cache.Touch(d) // what asks for it uses it (LRU)
	return d
}

func modelDir(m Model) string { return filepath.Join(ModelsDir(), m.DirName()) }

// ModelsDir is where the models' converted programs are kept: models under cache.Root.
// Each model has a directory of its name there.
func ModelsDir() string {
	return cache.Sub("models")
}

// Resolve returns the model a name of -use means: one of the catalog by its name; a family or a
// category of the catalog, with a tag or picked for the machine's memory (Pick: plamo3translate,
// plamo3translate:2b, translate); or a Hugging Face repository (OWNER/NAME) of an architecture
// hwnpu-chat runs, whose config.json it downloads (RepoDir) to find out what the model is for.
// What else such a model can do is not known. Only the model picked is downloaded.
func Resolve(ctx context.Context, name string, log io.Writer) (Model, error) {
	return ResolveOn(ctx, name, 0, log)
}

// ResolveOn is Resolve on a machine of machine GB (hwnpu-chat --memory): the model picked and
// run as such a machine would, whatever memory is chosen for the model (ChosenMemory); 0 for this
// machine's memory (MachineMemory) and the memory chosen.
func ResolveOn(ctx context.Context, name string, machine int, log io.Writer) (Model, error) {
	m, ok, err := lookupOn(name, machine, log)
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
	if err := Allowed(m); err != nil {
		return m, err // a repository's license is not known: a +license refuses it
	}
	path, err := fetch(ctx, m.Source(), "config.json", log)
	if err != nil {
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
// a family or a use (Pick, logged), each through the aliases, On the machine, or on the memory
// chosen for the model (ChosenMemory). ok is false for none.
func lookupFor(name string, log io.Writer) (Model, bool, error) { return lookupOn(name, 0, log) }

// LookupFor is the definition name picks on this machine (lookupFor, as AMDExport picks it):
// false for a name the definitions do not have.
func LookupFor(name string) (Model, bool) {
	m, ok, err := lookupFor(strings.TrimSpace(name), nil)
	return m, ok && err == nil
}

// LookupOn finds the model of the definitions a name means, as hwnpu's commands do (lookupOn),
// for a package that runs models of several kinds (ai): by its name, or as a family or a use
// picked for a machine of machine GB (0: this one, with the memory chosen for the model). Why
// it was picked goes to log. ok is false for none; a repository is not looked up.
func LookupOn(name string, machine int, log io.Writer) (Model, bool, error) {
	return lookupOn(name, machine, log)
}

// lookupOn is lookupFor on a machine of machine GB, 0 for this one's (MachineMemory). A machine
// given, or a tag of memory (lfm3:8g), runs the model as on it, whatever memory is chosen for it.
func lookupOn(name string, machine int, log io.Writer) (Model, bool, error) {
	given := machine > 0
	if !given {
		machine = MachineMemory()
	}
	if m, ok := Lookup(name); ok {
		m = Alias(m, log)
		m = m.On(runsAs(m, machine, given, log))
		return m, true, Allowed(m) // named: refused, not replaced
	}
	if _, tag, tagged := strings.Cut(name, ":"); tagged {
		if _, ok := MemoryTag(tag); ok {
			given = true
		}
	}
	m, why, ok, err := Pick(name, machine)
	if err != nil || !ok {
		return Model{}, false, err
	}
	if gb := runsAs(m, machine, given, log); gb != machine {
		m = byName(m.Name).On(gb)
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
	return m, Allowed(m)
}

// NoModelError is the error of a name that is no model, family or use of the definitions nor a
// repository.
type NoModelError struct{ Name string }

func (e *NoModelError) Error() string {
	return fmt.Sprintf("chat: no model %q (hwnpu-chat list shows the models, their families and the uses that pick one; a repository is written OWNER/NAME)", e.Name)
}

func errNoModel(name string) error { return &NoModelError{Name: name} }

// repoModel is the model a name of a repository means: OWNER/NAME or OWNER/NAME@REVISION (of
// the sources or Hugging Face). ok is false otherwise.
func repoModel(name string) (Model, bool) {
	repo, rev, _ := strings.Cut(name, "@")
	if strings.Count(repo, "/") != 1 || !safeName(repo) || strings.HasSuffix(repo, "/") {
		return Model{}, false
	}
	return Model{Name: path.Base(repo), Repo: repo, Revision: rev}, true
}

// Downloaded reports whether model m is here: its tables in its directory of programs
// (ModelDir, TablesFile: written once its programs are converted, which then run without the
// checkpoint), or its checkpoint (RepoDir): model.safetensors, or the index of a checkpoint in
// several files and all the files it names.
func Downloaded(m Model) bool {
	if exists(filepath.Join(modelDir(m), TablesFile)) {
		return true
	}
	if m.Repo == "" {
		return false
	}
	dir := RepoDir(m.Source())
	if exists(filepath.Join(dir, "model.safetensors")) {
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
