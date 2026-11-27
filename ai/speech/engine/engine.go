// Package engine runs ai/speech's models in this process: the transcription models of
// internal/transcribe and the speech models of internal/speech and internal/supertonic.
// ai/engine imports it; ai/speech alone is their client through hwnpud (docs/design/hwnpud.md,
// "Clients without the engine").
package engine

import (
	"fmt"
	"io"

	"github.com/shibukawa/hwnpu/ai/speech/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/llm/whisper"
	"github.com/shibukawa/hwnpu/internal/speechwire"
)

// speechwire.TranscribeRate (speech.TranscribeRate) is whisper's: this does not compile when they
// differ.
var _ = [1]struct{}{}[speechwire.TranscribeRate-whisper.SampleRate]

func init() {
	t := bridge.Roles["transcribe"]
	t.Prepare, t.Load = prepareTranscribe, loadTranscribe
	s := bridge.Roles["speak"]
	s.Prepare, s.Load = prepareSpeak, loadSpeak
}

// first is the first of ps, "" for none.
func first(ps []string) string {
	if len(ps) == 0 {
		return ""
	}
	return ps[0]
}

// logf writes lines to w (nil for nowhere).
func logf(w io.Writer) func(string, ...any) {
	return func(format string, args ...any) {
		if w != nil {
			fmt.Fprintf(w, format+"\n", args...)
		}
	}
}
