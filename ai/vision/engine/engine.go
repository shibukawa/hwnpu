// Package engine runs ai/vision's models in this process: the depth, face, detection, pose,
// segmentation, OCR and label models of internal/vision/..., internal/pose and their runtimes.
// ai/engine imports it; ai/vision alone is their client through hwnpud (docs/design/hwnpud.md,
// "Clients without the engine"). Each role's Prepare and Load (and Usable) are set here, through
// ai/vision/internal/bridge, and Load makes its model the client's type (bridge.Wrap).
package engine
