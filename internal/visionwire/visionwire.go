// Package visionwire is what the models of pictures (internal/vision/..., internal/pose) and
// their clients (ai/vision) share without the engine: the names of the models' classes and
// keypoints, the default labels and the small rules over the models' results
// (docs/design/hwnpud.md, "Clients without the engine"). The engine's packages keep their own
// copies (their code is hashed for the prebuilt programs: model-recipes.yaml's code); the tests of
// ai/vision/engine check that they agree.
package visionwire

import (
	"bufio"
	_ "embed"
	"math"
	"strings"
)

// COCO are the names of COCO's 80 classes, in the detection models' order (detect.COCO).
var COCO = []string{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
	"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
	"cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
	"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
	"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
	"orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
	"dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
	"toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"}

// Keypoints are the names of the COCO keypoints, in the pose models' order (pose.Names).
var Keypoints = []string{"nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
	"left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
	"left_ankle", "right_ankle"}

// SkyThreshold is Depth Anything 3's: a pixel whose sky output is at least it is sky
// (depth.SkyThreshold).
const SkyThreshold = 0.3

//go:embed labels.txt
var labelsTxt string

// DefaultLabels are the labels a picture is labeled with when none are given: common things,
// places and kinds of picture (labels.txt, siglip.DefaultLabels).
func DefaultLabels() []string { return ParseLabels(labelsTxt) }

// ParseLabels is the labels of a list: one a line, blank lines and lines starting with # left
// out.
func ParseLabels(s string) []string {
	var out []string
	sc := bufio.NewScanner(strings.NewReader(s))
	for sc.Scan() {
		l := strings.TrimSpace(sc.Text())
		if l == "" || strings.HasPrefix(l, "#") {
			continue
		}
		out = append(out, l)
	}
	return out
}

// Dot is a · b in float64 (siglip.Dot).
func Dot(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

// Bilinear resizes the map xs (w x h) to ow x oh as torch's interpolate does with
// align_corners=False and no antialias (moge.Bilinear).
func Bilinear(xs []float32, w, h, ow, oh int) []float32 {
	if w == ow && h == oh {
		return append([]float32(nil), xs...)
	}
	src := func(o, n, no int) (int, int, float32) {
		s := math.Max(0, (float64(o)+0.5)*float64(n)/float64(no)-0.5)
		i0 := min(int(s), n-1)
		return i0, min(i0+1, n-1), float32(s - float64(i0))
	}
	out := make([]float32, ow*oh)
	for y := 0; y < oh; y++ {
		y0, y1, fy := src(y, h, oh)
		for x := 0; x < ow; x++ {
			x0, x1, fx := src(x, w, ow)
			a := xs[y0*w+x0] + fx*(xs[y0*w+x1]-xs[y0*w+x0])
			b := xs[y1*w+x0] + fx*(xs[y1*w+x1]-xs[y1*w+x0])
			out[y*ow+x] = a + fy*(b-a)
		}
	}
	return out
}
