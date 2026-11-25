package main

import (
	"fmt"
	"image"
	"sync"
	"time"

	"github.com/shibukawa/bonsaiai/ai"
	"github.com/shibukawa/bonsaiai/ai/flow"
	"github.com/shibukawa/bonsaiai/ai/vision"
)

// tracker is the part of the app that follows the viewer's head: a network that keeps running
// on the camera's frames.
//
//	camera → faces (the newest frame only) → head → steady → view
//
// faces is a model (YuNet). head (vision.HeadNode) works out where the nearest head is from its
// face. steady is a node of the program: it smooths what the detector finds and brings the view
// back to the middle when no face was seen for a while. view keeps the newest for the window,
// which draws on its own beat (60 times a second) with whatever was last found.
type tracker struct {
	camera *flow.Src[image.Image]

	// Gain is how far the view goes for a head at the edge of the camera's view: a head moves
	// over a part of the camera's view only, so 2.5 has the picture turn fully for a head 40 %
	// of the way to the edge. It is set before the network runs; AddGain changes it after.
	Gain float64

	mu     sync.Mutex
	target View
	at     time.Time // when a face was last found
	found  int       // the frames a face was found in
	head   vision.Head
	// the camera's newest frame, and what the detector found in the newest frame it read, for
	// the window to show how the head is found
	frame image.Image
	faces vision.Found
}

func newTracker(app *ai.App, faces ai.Use[*vision.FaceDetector], o vision.HeadOptions) *tracker {
	t := &tracker{camera: flow.Source[image.Image]("camera"), Gain: 2.5}
	var smooth View
	steady := flow.Each("steady", func(c flow.Chunk[vision.Head], emit func(flow.Chunk[View])) error {
		t.mu.Lock()
		gain := t.Gain
		t.mu.Unlock()
		to := View{X: clamp(c.Data.X * gain), Y: clamp(c.Data.Y * gain)}
		smooth = smooth.Follow(to, 0.5) // half of a detector's jitter
		t.mu.Lock()
		t.head, t.at, t.found = c.Data, time.Now(), t.found+1
		t.mu.Unlock()
		emit(flow.Chunk[View]{Data: smooth, Start: c.Start, Final: true})
		return nil
	}, flow.WhenIdle(1500*time.Millisecond, func(emit func(flow.Chunk[View])) error {
		// no face for a while (the viewer left, or looked away): back to the middle
		smooth = View{}
		emit(flow.Chunk[View]{Data: smooth, Final: true})
		return nil
	}))
	view := flow.Sink[View]("view").Do(func(c flow.Chunk[View]) error {
		t.mu.Lock()
		t.target = c.Data
		t.mu.Unlock()
		return nil
	})
	// the frames and the faces, as they pass, for the window
	frame := flow.Tap("frame", func(c flow.Chunk[image.Image]) {
		t.mu.Lock()
		t.frame = c.Data
		t.mu.Unlock()
	})
	found := flow.Tap("found", func(c flow.Chunk[vision.Found]) {
		t.mu.Lock()
		t.faces = c.Data
		t.mu.Unlock()
	})
	// the detector reads the newest frame only: a frame that waited is of no use
	app.From(t.camera).To(frame).To(vision.FacesNode(faces), flow.Newest()).To(found).To(vision.HeadNode(o)).To(steady).To(view)
	return t
}

// Seen is the camera's newest frame, the faces found in the newest frame the detector read
// (in that frame's pixels), and the head worked out from the nearest; nil before the first frame.
func (t *tracker) Seen() (image.Image, vision.Found, vision.Head, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.frame, t.faces, t.head, t.found > 0 && time.Since(t.at) < time.Second
}

func clamp(x float64) float64 { return min(max(x, -1), 1) }

// AddGain changes the gain by step, within 0.5 to 10, and returns it: the window's keys and
// wheel, for a viewer whose head moves little.
func (t *tracker) AddGain(step float64) float64 {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.Gain = min(max(t.Gain+step, 0.5), 10)
	return t.Gain
}

// Target is where the view is to go, and a line that says what the tracker sees.
func (t *tracker) Target() (View, string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.found == 0 {
		return t.target, "camera: no face yet"
	}
	if since := time.Since(t.at); since > time.Second {
		return t.target, fmt.Sprintf("camera: no face for %.0f s", since.Seconds())
	}
	return t.target, fmt.Sprintf("camera: head %+.2f %+.2f, %.2f m away", t.head.X, t.head.Y, t.head.Z)
}
