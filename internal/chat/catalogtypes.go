package chat

import "github.com/shibukawa/hwnpu/internal/catalog"

// The data of the definitions are internal/catalog's, which a client of hwnpud reads without the
// engine; these are its names here.
type (
	Kind      = catalog.Kind
	Model     = catalog.Model
	NPUSpec   = catalog.NPUSpec
	Pin       = catalog.Pin
	HubRepo   = catalog.HubRepo
	PerVendor = catalog.PerVendor
	Run       = catalog.Run
	Variant   = catalog.Variant
	License   = catalog.License
)

const (
	KindChat       = catalog.KindChat
	KindEmbed      = catalog.KindEmbed
	KindTranscribe = catalog.KindTranscribe
	KindSpeech     = catalog.KindSpeech
	KindDecision   = catalog.KindDecision
	KindImage      = catalog.KindImage
	KindDepth      = catalog.KindDepth
	KindFace       = catalog.KindFace
	KindPitch      = catalog.KindPitch
	KindPose       = catalog.KindPose
	KindDetect     = catalog.KindDetect
	KindOCR        = catalog.KindOCR
	KindLabel      = catalog.KindLabel
	KindSegment    = catalog.KindSegment
	KindNotes      = catalog.KindNotes
	KindMusic      = catalog.KindMusic
	KindSFX        = catalog.KindSFX
)

var (
	Vendors     = catalog.Vendors
	Devices     = catalog.Devices
	Chips       = catalog.Chips
	ExportChips = catalog.ExportChips
)

// Chip is the machine's chip (catalog.Chip).
func Chip() string { return catalog.Chip() }

// ActAsChip makes the process convert as a Mac of chip would (catalog.ActAsChip).
func ActAsChip(chip string) (undo func()) { return catalog.ActAsChip(chip) }

// TakesExportsOf is catalog.TakesExportsOf.
func TakesExportsOf(arch, chip string) bool { return catalog.TakesExportsOf(arch, chip) }

// chipName is the machine's chip (catalog.ChipName), as model.go names it: its code is the hash
// of the chat models' exports (model-recipes.yaml's code), which a rename would change.
func chipName() string { return catalog.ChipName() }
