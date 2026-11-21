// The samples are a module of their own, so that what they take beside hwnpu (hwcamera for
// the microphone and the camera) is not a dependency of the library. They are built against
// the hwnpu of this checkout.
module github.com/shibukawa/hwnpu/samples

go 1.27.0

require (
	github.com/alecthomas/kong v1.17.0
	github.com/ebitengine/oto/v3 v3.5.1
	github.com/hajimehoshi/ebiten/v2 v2.10.6
	github.com/shibukawa/hwcamera v0.1.0
	github.com/shibukawa/hwnpu v0.0.0
	golang.org/x/image v0.45.0
)

require (
	github.com/ebitengine/gomobile v0.0.0-20260820040257-d11f821a26a6 // indirect
	github.com/ebitengine/hideconsole v1.0.0 // indirect
	github.com/ebitengine/purego v0.11.3 // indirect
	github.com/goccy/go-yaml v1.19.2 // indirect
	github.com/jfreymuth/pulse v0.1.3 // indirect
	github.com/klauspost/compress v1.20.3 // indirect
	github.com/shibukawa/configdir v1.1.0 // indirect
	github.com/shibukawa/tinygodriver v1.3.10 // indirect
	golang.org/x/sync v0.24.0 // indirect
	golang.org/x/sys v0.50.0 // indirect
	golang.org/x/text v0.43.0 // indirect
	google.golang.org/protobuf v1.36.12 // indirect
)

replace github.com/shibukawa/hwnpu => ../
