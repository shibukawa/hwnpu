package transcribe

import "testing"

// TestConvRows checks the encoder's input of tools/amd/iron_whisper.py: conv1's im2col (bin-major,
// then the tap, padding 1), the even frames from row 0 and the odd from row rows.
func TestConvRows(t *testing.T) {
	const mels, frames, rows, convIn = 2, 6, 4, 8
	mel := make([]float32, frames*mels)
	for f := 0; f < frames; f++ {
		for c := 0; c < mels; c++ {
			mel[f*mels+c] = float32(10*f + c + 1)
		}
	}
	at := func(f, c int) float32 {
		if f < 0 || f >= frames {
			return 0
		}
		return mel[f*mels+c]
	}
	got := ConvRows(mel, mels, frames, rows, convIn)
	if len(got) != 2*rows*convIn {
		t.Fatalf("%d values, want %d", len(got), 2*rows*convIn)
	}
	for f := 0; f < frames; f++ {
		r := f/2 + f%2*rows
		for c := 0; c < mels; c++ {
			for tap := 0; tap < 3; tap++ {
				if v, want := got[r*convIn+c*3+tap], at(f+tap-1, c); v != want {
					t.Errorf("frame %d bin %d tap %d: %v, want %v", f, c, tap, v, want)
				}
			}
		}
		for i := 3 * mels; i < convIn; i++ {
			if got[r*convIn+i] != 0 {
				t.Errorf("frame %d: column %d of the padding is %v", f, i, got[r*convIn+i])
			}
		}
	}
	// the rows past the frames are zeros
	for _, r := range []int{3, rows + 3} {
		for i := 0; i < convIn; i++ {
			if got[r*convIn+i] != 0 {
				t.Errorf("row %d is past the frames, column %d is %v", r, i, got[r*convIn+i])
			}
		}
	}
}
