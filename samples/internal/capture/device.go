package capture

import (
	"context"
	"fmt"
	"strconv"
)

// pick is the ID of the device a sample's --device names: its number in the list (1 the first,
// as the samples' mics and cameras print them), or the ID itself. "" is the system's default.
func pick(device string, list func() ([]string, error)) (string, error) {
	n, err := strconv.Atoi(device)
	if err != nil || device == "" {
		return device, nil
	}
	ids, err := list()
	if err != nil {
		return "", err
	}
	if n < 1 || n > len(ids) {
		return "", fmt.Errorf("device %d: there are %d (1 to %d)", n, len(ids), len(ids))
	}
	return ids[n-1], nil
}

// MicID is the ID of the audio input device names: its number in Mics (1 the first), or an ID.
func MicID(ctx context.Context, device string) (string, error) {
	return pick(device, func() ([]string, error) {
		ins, err := Mics(ctx)
		ids := make([]string, len(ins))
		for i, in := range ins {
			ids[i] = in.ID
		}
		return ids, err
	})
}

// CameraID is the ID of the camera device names: its number in Cameras (1 the first), or an ID.
func CameraID(ctx context.Context, device string) (string, error) {
	return pick(device, func() ([]string, error) {
		cams, err := Cameras(ctx)
		ids := make([]string, len(cams))
		for i, c := range cams {
			ids[i] = c[0]
		}
		return ids, err
	})
}
