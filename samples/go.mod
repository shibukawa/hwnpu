// The samples are a module of their own, so that what they take beside hwnpu (hwcamera for
// the microphone and the camera) is not a dependency of the library. They are built against
// the hwnpu of this checkout.
module github.com/shibukawa/hwnpu/samples

go 1.27.0

require (
	github.com/alecthomas/kong v1.17.0
	github.com/shibukawa/hwcamera v0.1.0
	github.com/shibukawa/hwnpu v0.0.0
)

require (
	github.com/ebitengine/purego v0.11.2 // indirect
	github.com/goccy/go-yaml v1.19.2 // indirect
	github.com/klauspost/compress v1.20.3 // indirect
	github.com/shibukawa/configdir v1.1.0 // indirect
	github.com/shibukawa/tinygodriver v1.3.10 // indirect
	golang.org/x/sys v0.50.0 // indirect
	golang.org/x/text v0.43.0 // indirect
	google.golang.org/protobuf v1.36.12 // indirect
)

replace github.com/shibukawa/hwnpu => ../
