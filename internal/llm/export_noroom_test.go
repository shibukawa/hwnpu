package llm

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/shibukawa/hwnpu/internal/cache"
)

// An export the cache had no room for is not converted here instead: that would need the
// checkpoint, more room still.
func TestMayConvertAfterNoRoom(t *testing.T) {
	ctx := context.Background()
	export := t.TempDir() + "/m.coreai"
	NoteFetch(export, fmt.Errorf("webdav: %w", cache.ErrNoRoom))
	if err := MayConvert(ctx, export); !errors.Is(err, cache.ErrNoRoom) {
		t.Fatalf("MayConvert after no room = %v, want ErrNoRoom", err)
	}
	if err := MayConvert(ctx, export+"/"); !errors.Is(err, cache.ErrNoRoom) {
		t.Fatalf("MayConvert of the same path = %v, want ErrNoRoom", err)
	}
	// of several layers' errors, the one without room
	noRoomErr := fmt.Errorf("webdav: %w", cache.ErrNoRoom)
	NoteFetch(export, errors.Join(errors.New("s3: denied"), noRoomErr, errors.New("not there")))
	if err := FetchFailed(export); err != noRoomErr {
		t.Fatalf("FetchFailed of joined errors = %v, want %v", err, noRoomErr)
	}
	NoteFetch(export, errors.New("not found"))
	if err := MayConvert(ctx, export); err != nil {
		t.Fatalf("MayConvert after another error = %v, want nil", err)
	}
	NoteFetch(export, fmt.Errorf("x: %w", cache.ErrNoRoom))
	NoteFetch(export, nil)
	if err := FetchFailed(export); err != nil {
		t.Fatalf("FetchFailed after a fetch = %v, want nil", err)
	}
}
