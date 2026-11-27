// Package catalog is the data of hwnpu's model definitions (models.yaml): the models, their
// kinds, licenses, memory and variants, and the machine's chip they are read for. It holds no
// engine: a client of hwnpud reads it (docs/design/hwnpud.md, "Clients without the engine"), and
// internal/chat, which reads the files and runs the models, names its types.
package catalog

import (
	"cmp"
	"fmt"
	"maps"
	"path"
	"slices"
	"strings"
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

// NPUSpec is a model's export for one NPU vendor (models.yaml's npu): the weight formats it is
// made in (Text for the text encoder, Dit for the DiT; "" for the defaults), the picture sizes
// that are the default on every generation of the vendor's NPU (Sizes: the aspect ratio's name,
// then width and height), the sizes a generation overrides (ByGen: by generation, a chip on
// Apple, xdna1 or xdna2 on AMD), and the edits the export holds programs for (Edits: by aspect
// ratio, the references of each as hwnpu-image names them: "1", "2@half", "1+2@half", ...).
type NPUSpec struct {
	Text, Dit string
	Sizes     map[string][2]int
	ByGen     map[string]map[string][2]int
	Edits     map[string][]string
	// EditsByGen are a generation's edits where they are not Edits' (npu's generation_edits), by
	// aspect ratio: an export of its own for that generation, as its other sizes make one.
	EditsByGen map[string]map[string][]string
}

// EditsOn are the edits of the aspect ratios on generation gen: its own where it has them, the
// default's otherwise ("" or an unknown generation: the default).
func (s NPUSpec) EditsOn(gen string) map[string][]string {
	out := make(map[string][]string, len(s.Edits))
	for k, v := range s.Edits {
		out[k] = v
	}
	for k, v := range s.EditsByGen[gen] {
		out[k] = v
	}
	return out
}

// SizesOn are the picture sizes of the aspect ratios on generation gen: its override where it has
// one, the default otherwise ("" or an unknown generation: the default).
func (s NPUSpec) SizesOn(gen string) map[string][2]int {
	out := make(map[string][2]int, len(s.Sizes))
	for k, v := range s.Sizes {
		out[k] = v
	}
	for k, v := range s.ByGen[gen] {
		out[k] = v
	}
	return out
}

// Model is a model of the definitions (models.yaml).
type Model struct {
	Name     string // what -use and the API call it
	Kind     Kind
	Features []string // what it can do besides its kind, from Features
	Repo     string   // the Hugging Face repository it is downloaded from
	Revision string   // the repository's revision to download ("": main)
	// Targets are the versions of the model's conversions by target (models.yaml's targets).
	Targets map[string]int
	// Develop marks a model in development (models.yaml's develop).
	Develop bool
	// Pins are the conversions models.yaml's exports pin: by target, then export.
	Pins map[string]map[string]Pin
	// Hub are the repositories of the model's conversions on Hugging Face (models.yaml's hub),
	// in the order they are read: those of the files read last first.
	Hub []HubRepo
	// SHA256 pins the repository's files by name.
	SHA256 map[string]string
	// Arch is the structure the definitions name (llm.Lookup), "" for the one config.json says.
	Arch string
	// Files are the definition files that wrote the model (ModelFiles).
	Files []string
	Bytes float64 // the size of the download (0: unknown), for a vendor BytesBy does not name (BytesFor)
	// BytesBy is the size of the download by the NPU's vendor (PerVendor).
	BytesBy PerVendor
	// DiskBy is what the model keeps in the cache to run, by the NPU's vendor (DiskFor): its
	// conversion unpacked and its tables, not the checkpoint (models.yaml's disk).
	DiskBy PerVendor
	About  string
	// NoThink is the model's own default of answering without thinking first, which -think
	// given on the command line, reasoning_effort or enable_thinking override (Qwen3.5's 0.8B
	// and 2B: their chat templates think only when asked, and the 0.8B thinks to the budget on
	// simple questions, docs/design/llm-qwen.md).
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
	// export is not here: the exporter's model and options (PrepareAMD), from its recipe
	// (model-recipes.yaml amd: the args but {in} and {out}); none for a model AMD's NPU has no
	// form of from that script.
	AMDExport []string
	// Recipe is the model whose recipe (model-recipes.yaml) this one is made by: its own, else
	// that of the model it is like (models.yaml like); "" for none.
	Recipe string
	// ExpertsCore are the experts of each layer that every kind of text runs (models.yaml's
	// experts_core), wired from the first run when the experts do not all fit (Config.Experts).
	ExpertsCore [][]int
	// File is the model's checkpoint file in a repository that holds several models (Anima's
	// diffusion models: anima-turbo-v1.1.safetensors), "" when the repository is the one model.
	File string
	// With are the other repositories the model reads, by role (an OCR's detector): OWNER/NAME
	// or OWNER/NAME@REVISION as fetch takes them. Their files' sha256 are SHA256's ROLE/FILE.
	With map[string]string
	// NPU is the model's export for each NPU vendor (models.yaml's npu: apple, amd, intel); a
	// vendor it leaves out has none.
	NPU map[string]NPUSpec
	// NotOn are the chips (Chips) a family or a category does not pick the model on: it is too
	// slow there (PLaMo 3 Translate 8B on an M1, over the 2.8 GiB its Neural Engine runs at full
	// speed). Its name and FAMILY:TAG still run it.
	NotOn []string
	// License is the weights' license (models.yaml's license): its id, where it is, its class,
	// and whether the user accepts it before the first download or run (AcceptLicense).
	License License
	// Content is what the model may make that the policy refuses unless allowed (Contents: nsfw).
	Content []string
	// Settings are how the model's role prepares and runs it beyond the fields above (models.yaml's
	// settings: a speech model's format), and Programs files converted beforehand that the role
	// uses as they are, by the name it reads them as (models.yaml's programs).
	Settings map[string]string
	Programs map[string][]string
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
	if m.Revision != "" {
		return m.Repo + "@" + m.Revision
	}
	return m.Repo
}

// DeviceFor is where m runs on a machine whose NPU is vendor's: its definition's Device, "npu"
// where it names none.
func (m Model) DeviceFor(vendor string) string {
	if d, ok := m.Device[vendor]; ok {
		return d
	}
	return "npu"
}

// PerVendor is a value by the vendor of the machine's NPU (Vendors); "" holds the one for the
// vendors it does not name. A definition writes it as one number (every vendor) or as a table
// of vendors (memory: {amd: 3.5, apple: 4.1}): what a model takes and downloads can differ by
// NPU (Anima-Turbo: 3.5 GB on AMD's, 4.1 on Apple's). The table may also name Apple's chips
// (Chips) where one differs from the others (memory: {amd: 2.2, apple: 3.1, M1: 1.7}: an M1
// does not load LFM3-1.7B's whole program of short conversations).
type PerVendor map[string]float64

// For is the value for vendor: on Apple's, this machine's chip's when the table names it, else
// the vendor's own, else the one for every vendor, else the largest of the table (a vendor it
// leaves out is not assumed to need less).
func (p PerVendor) For(vendor string) float64 {
	if vendor == "apple" {
		if c := ChipName(); c != "" {
			if v, ok := p[c]; ok {
				return v
			}
		}
	}
	if v, ok := p[vendor]; ok {
		return v
	}
	if v, ok := p[""]; ok {
		return v
	}
	m := 0.0
	for _, v := range p {
		m = max(m, v)
	}
	return m
}

// UnmarshalYAML reads a number or a table of vendors and chips.
func (p *PerVendor) UnmarshalYAML(unmarshal func(any) error) error {
	var one float64
	if err := unmarshal(&one); err == nil {
		*p = PerVendor{"": one}
		return nil
	}
	var by map[string]float64
	if err := unmarshal(&by); err != nil {
		return fmt.Errorf("a number or a table of vendors (%s) and chips (%s)", strings.Join(Vendors, ", "), strings.Join(Chips, ", "))
	}
	for v := range by {
		if !slices.Contains(Vendors, v) && !slices.Contains(Chips, v) {
			return fmt.Errorf("%q is not a vendor (%s) or a chip (%s)", v, strings.Join(Vendors, ", "), strings.Join(Chips, ", "))
		}
	}
	*p = by
	return nil
}

// over is p with q written over it: q's number replaces p, q's table changes the vendors it
// names.
func (p PerVendor) Over(q PerVendor) PerVendor {
	if _, one := q[""]; one || p == nil {
		return maps.Clone(q)
	}
	r := maps.Clone(p)
	maps.Copy(r, q)
	return r
}

// MemoryFor is what m takes running on a machine whose NPU is vendor's, in GB (0: unknown).
func (m Model) MemoryFor(vendor string) float64 { return m.MemoryBy.For(vendor) }

// BytesFor is the size of m's download on a machine whose NPU is vendor's (0: unknown).
func (m Model) BytesFor(vendor string) float64 { return m.BytesBy.For(vendor) }

// DiskFor is what m keeps in the cache to run on a machine whose NPU is vendor's (0: unknown):
// its conversion unpacked and its tables. Preparing it takes the download and this at once for
// a while (a packed export is unpacked beside its pack). Unlike memory and bytes, a vendor the
// table leaves out is not known (0), not the largest: the sizes come from what the sources hold,
// and a vendor's may not be there yet.
func (m Model) DiskFor(vendor string) float64 {
	if v, ok := m.DiskBy[""]; ok {
		return v
	}
	if vendor == "apple" {
		if v, ok := m.DiskBy[ChipName()]; ok {
			return v
		}
	}
	return m.DiskBy[vendor]
}

// HubRepo is a repository of a model's conversions on Hugging Face (models.yaml's hub).
type HubRepo struct {
	Repo string // OWNER/NAME
	// Pins are the commit each target reads (amd-xdna2, apple, coreai-ARCH): 40 hex digits, or
	// main (to try what is there now).
	Pins map[string]string
	// File is the definition file that wrote the repository last (Builtin for the built-in one).
	File string
}

// Over is r with the fields o gives.

// License is a model's license in models.yaml:
//
//	license: {id: qwen-research, url: https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/LICENSE,
//	          class: non-commercial, terms: "research and evaluation only, not commercial use", accept: true}
//
// With Accept, the license restricts what the weights may be used for (non-commercial ones): the
// user accepts it once before the model is first downloaded or run (AcceptLicense), and the
// acceptance is kept in LicensesFile.
type License struct {
	// ID names the license: an SPDX identifier (apache-2.0) or the license's own name
	// (qwen-research). Acceptances are kept by it, so models of one license ask once.
	ID string `yaml:"id"`
	// URL is where the license's text is.
	URL string `yaml:"url"`
	// Terms is what it allows, in one line, shown when asking.
	Terms string `yaml:"terms"`
	// Class is what it allows of commercial use (LicenseClasses: open, conditional,
	// non-commercial), which the model policy reads (license:non-commercial).
	Class string `yaml:"class"`
	// Accept makes AcceptLicense ask.
	Accept bool `yaml:"accept"`
}

// Pin is a conversion models.yaml's exports pin (hwnpu-dev models pin writes them): the folder
// of its entry in a source, converted/MODEL/TARGET/EXPORT/At (vN, vN-HASH or v0-HASH), and the
// digest of what it holds (store.Entry.Digest). It is taken from that folder in any source, in
// the model's repository on Hugging Face at the commit its hub pins, or here, when its entry has
// that digest, whatever its recipe (the tools that made it): what changes the pinned exports'
// bytes changes the pin.
type Pin struct {
	At     string
	Digest string
}

func (p Pin) String() string { return p.At + " " + p.Digest }

// Version is the version of the pin's folder (N of vN or vN-HASH; 0 of v0-HASH).
func (p Pin) Version() int {
	n := 0
	for _, c := range strings.TrimPrefix(p.At, "v") {
		if c < '0' || c > '9' {
			break
		}
		n = n*10 + int(c-'0')
	}
	return n
}

// OfVersion reports whether the pin is of version v of its export (models.yaml targets, or
// model-recipes.yaml amd.version): its folder's version is v; v0-HASH stands for version 1 too,
// the version of the exports pinned before the recipes gave versions (2026-11-23). A pin of
// another version is not taken (placeOf), so that raising a recipe's version does not take the
// bytes of before.
func (p Pin) OfVersion(v int) bool {
	return p.Version() == v || p.Version() == 0 && v <= 1
}

// Run is how hwnpu-chat runs a chat model when its flags do not say: models.yaml's prefill,
// format, context, paged and tier, the defaults of --prefill, --format, --context, --paged and
// --tier ("" and 0: hwnpu-chat's own).
type Run struct {
	Prefill string // PrefillAuto, PrefillOn or PrefillOff
	Format  string // the weight format of its programs (lut8, int8, ...)
	Context int    // tokens of a prompt and its answer together
	Paged   int    // the chunk length to split it at its attention with, -1 never
	// Tier is the caches shorter than the context to run on while a conversation is short, as
	// --tier takes them (Tiers): auto, none or a list. LFM3-1.7B has none on a machine of 8 GB,
	// where the whole program of short conversations would hold its weights a second time
	// (3.1 GB against 1.7), and auto from 16 GB.
	Tier string
	// Codes is the hash of the lookup tables chosen against the activations (GPTQ) that its
	// programs are converted with (Setup.CodesHash): Qwen2.5-Coder's 4 bit tables, which
	// rounding alone leaves far from the model (docs/design/llm-qwen.md).
	Codes string
}

// over is r with the fields o gives.
func (r Run) Over(o Run) Run {
	return Run{Prefill: cmp.Or(o.Prefill, r.Prefill), Format: cmp.Or(o.Format, r.Format),
		Context: cmp.Or(o.Context, r.Context), Paged: cmp.Or(o.Paged, r.Paged), Tier: cmp.Or(o.Tier, r.Tier),
		Codes: cmp.Or(o.Codes, r.Codes)}
}

// zero reports whether r gives nothing.
func (r Run) Zero() bool { return r == Run{} }

// Variant is a model's settings on a machine of more memory than its Machine (models.yaml's
// variants): what it takes running and how it runs, the fields it gives replacing the model's.
// Qwen3-1.7B reads prompts with a program of its own from 16 GB and with the one that generates
// on 8 GB; LFM3-1.7B has its whole program of short conversations from 16 GB.
type Variant struct {
	Memory PerVendor // nil: the model's
	Run    Run
	// AMDExport replaces the model's (nil: the model's), from its recipe's variant of the same
	// GB: gpt-oss-20b keeps 16 of its 32 experts a layer on AMD's NPU on 16 GB, all of them from 24 GB
	AMDExport []string
}

// On is the model as a machine of machine GB runs it: with the variant of the largest memory the
// machine has (Variants), or as it is when none is that small.
func (m Model) On(machine int) Model {
	best := 0
	for gb := range m.Variants {
		if gb <= machine && gb > best {
			best = gb
		}
	}
	if best == 0 {
		return m
	}
	v := m.Variants[best]
	if v.Memory != nil {
		m.MemoryBy = m.MemoryBy.Over(v.Memory)
		m.Memory = m.MemoryBy.For("")
	}
	m.Run = m.Run.Over(v.Run)
	if v.AMDExport != nil {
		m.AMDExport = v.AMDExport
	}
	return m
}

// Vendors are the NPU vendors a model's Device names (types.Vendor's names).
var Vendors = []string{"amd", "apple", "intel"}

// Devices are where a model runs: the NPU, or the CPU beside it (a small model can cost less
// there: YuNet on AMD's NPU took 16 times the energy of the CPU, docs/amd-npu-models.md 17).
var Devices = []string{"npu", "cpu"}
