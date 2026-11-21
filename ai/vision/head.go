package vision

import (
	"cmp"
	"image"
	"math"
)

// Head is where a head is before a camera, worked out from its face in a frame.
type Head struct {
	// X and Y are the head's place across the camera's view, -1 to 1: X from the left edge of
	// the view to the right as the person before the camera has them (a frame is the camera's
	// view: the person's right is on its left, so X is mirrored from the frame), Y from the
	// bottom to the top. 0, 0 is the centre.
	X, Y float64
	// Z is how far the head is from the camera, in meters, from how far apart its eyes are in
	// the frame: an estimate that holds for a face looking at the camera.
	Z float64
	// Right, Up and Away are the same place in meters from the camera's axis: to the person's
	// right, up, and away from the camera (Away is Z).
	Right, Up, Away float64
}

// HeadOptions are what a head's place is worked out with. The zero value of a field takes its
// default.
type HeadOptions struct {
	// FOV is the camera's horizontal field of view in degrees (default 60, a laptop's camera).
	FOV float64
	// Eyes is how far apart a person's eyes are, in meters (default 0.063).
	Eyes float64
}

// HeadOf is where the head of a face is before the camera whose frame (frame: its bounds) the
// face was found in, by a pinhole camera of the options' field of view: its eyes' middle for
// the direction, their distance in the frame for how far it is.
func HeadOf(f Face, frame image.Rectangle, o HeadOptions) Head {
	fov := cmp.Or(o.FOV, 60) * math.Pi / 180
	eyes := cmp.Or(o.Eyes, 0.063)
	w, h := float64(frame.Dx()), float64(frame.Dy())
	focal := w / 2 / math.Tan(fov/2) // in pixels
	// the middle of the eyes, from the frame's centre
	ex := (f.RightEye.X+f.LeftEye.X)/2 - float64(frame.Min.X) - w/2
	ey := (f.RightEye.Y+f.LeftEye.Y)/2 - float64(frame.Min.Y) - h/2
	apart := math.Hypot(f.LeftEye.X-f.RightEye.X, f.LeftEye.Y-f.RightEye.Y)
	if apart < 1 { // no landmarks: the box, whose width is about 2.2 times the eyes' distance
		apart = float64(f.Box.Dx()) / 2.2
		ex = float64(f.Box.Min.X+f.Box.Max.X)/2 - float64(frame.Min.X) - w/2
		ey = float64(f.Box.Min.Y+f.Box.Max.Y)/2 - float64(frame.Min.Y) - h/2
	}
	hd := Head{X: -ex / (w / 2), Y: -ey / (h / 2)}
	if apart >= 1 {
		hd.Z = focal * eyes / apart
		hd.Away = hd.Z
		hd.Right = -ex * hd.Z / focal
		hd.Up = -ey * hd.Z / focal
	}
	return hd
}

// NearestHead is the head of the nearest face among those found in a frame (the largest: the
// one before the screen), false when there is none.
func NearestHead(faces []Face, frame image.Rectangle, o HeadOptions) (Head, bool) {
	best, ok := Face{}, false
	for _, f := range faces {
		if !ok || f.Box.Dx()*f.Box.Dy() > best.Box.Dx()*best.Box.Dy() {
			best, ok = f, true
		}
	}
	if !ok {
		return Head{}, false
	}
	return HeadOf(best, frame, o), true
}
