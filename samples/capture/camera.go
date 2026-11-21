package capture

import (
	"cmp"
	"context"
	"fmt"
	"image"

	"github.com/shibukawa/hwcamera"

	"github.com/shibukawa/hwnpu/ai/flow"
)

// CameraOptions are how a camera is read. The zero value is the system's default camera in a
// small mode, as a model that looks at its frames wants them.
type CameraOptions struct {
	// Device is a camera of Cameras; "" is the system's default camera.
	Device string
	// Width and Height ask for a mode of that size (0: a small one, about 640 x 480); FPS for
	// that many frames a second (0: the camera's own).
	Width, Height int
	FPS           float64
}

// Camera is the stream of a camera's frames, each a picture in Go's own memory (hwcamera's
// frame is copied out of the driver's and given back at once), timed by the camera's clock. The
// camera is opened at the first read and closed when the stream ends, which is, without an
// error, when ctx is cancelled.
//
// A camera does not wait for its reader, and a frame that was not read in time is of no use:
// what reads this stream slower than the frames come takes the newest only (flow.Latest, or an
// edge of flow.Newest).
func Camera(ctx context.Context, o CameraOptions) flow.Stream[image.Image] {
	return func(yield func(flow.Chunk[image.Image], error) bool) {
		// the camera's nearest mode is used; 640 x 480 is plenty to find a face in
		opts := []hwcamera.Option{hwcamera.WithFormat(hwcamera.RGBA), hwcamera.WithSize(cmp.Or(o.Width, 640), cmp.Or(o.Height, 480))}
		if o.FPS > 0 {
			opts = append(opts, hwcamera.WithFrameRate(o.FPS))
		}
		cam, err := hwcamera.Open(ctx, o.Device, opts...)
		if err != nil {
			if ctx.Err() == nil {
				yield(flow.Chunk[image.Image]{}, fmt.Errorf("the camera: %w", err))
			}
			return
		}
		defer cam.Close()
		for {
			f, err := cam.Read(ctx)
			if err != nil {
				if ctx.Err() == nil {
					yield(flow.Chunk[image.Image]{}, fmt.Errorf("the camera: %w", err))
				}
				return
			}
			img := image.NewRGBA(image.Rect(0, 0, f.Width, f.Height))
			stride := f.Strides[0]
			for y := 0; y < f.Height; y++ {
				copy(img.Pix[y*img.Stride:y*img.Stride+f.Width*4], f.Planes[0][y*stride:])
			}
			at := f.Timestamp
			f.Release()
			if !yield(flow.Chunk[image.Image]{Data: img, Start: at, Final: true}, nil) {
				return
			}
		}
	}
}

// Cameras lists the cameras, for a sample's --device: their IDs and names.
func Cameras(ctx context.Context) ([][2]string, error) {
	ds, err := hwcamera.Devices(ctx)
	if err != nil {
		return nil, err
	}
	out := make([][2]string, len(ds))
	for i, d := range ds {
		out[i] = [2]string{d.ID, d.Name}
	}
	return out, nil
}
