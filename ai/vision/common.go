package vision

// The settings and programs the definition of a model of pictures may give (ai.Definition), as
// each role reads them:
//
//	settings: threads   the CPU's threads where it runs on the CPU (1, the default: the least
//	                    CPU time and energy)
//	settings: cpu       interpreter: the converted programs on hwnpu's CPU interpreter instead
//	                    of the NPU (slow; for checking a conversion), of a model of device cpu
//	programs: export    an export's directory (tools/amd/iron_*.py, or one Prepare made): depth,
//	                    faces and labels
//	programs: onnx      faces: an ONNX file of YuNet
//
// A model of device cpu runs on the CPU: its ONNX files with ONNX Runtime where its library is
// here, else on hwnpu's own executor. A device the definition gives for every NPU (an app's
// Definition.Device) is chosen as a command's --device is: over HWNPU_DEVICE, and an NPU without
// a form of the model is then an error rather than the CPU. The engine (ai/vision/engine) reads
// them.

// Point is a point of a picture, in its pixels.
type Point struct{ X, Y float64 }

// Box is a rectangle of a picture, in its pixels: Min its top left corner, Max its bottom right.
type Box struct{ Min, Max Point }

// Dx and Dy are the box's width and height.
func (b Box) Dx() float64 { return b.Max.X - b.Min.X }
func (b Box) Dy() float64 { return b.Max.Y - b.Min.Y }
