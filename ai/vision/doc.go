// Package vision has the models of pictures for an app (ai.App): the depth of a picture (Depth),
// the faces in one (Faces), with the position of a head worked out from its face (Head), the
// objects and their labels (Detect), people and their keypoints (Pose), the thing a prompt picks
// cut out (Segment), the text (OCR), and labels of any candidates and embeddings of pictures and
// texts in one space (Label).
//
//	var (
//		depth = vision.Depth("depth")
//		faces = vision.Faces("face-detect")
//	)
//	app := ai.NewApp("window", depth, faces)
//	...
//	est, err := depth.Get(ctx, rt)
//	d, err := est.Estimate(ctx, picture)     // the whole picture's depth, once
//	near := d.Near(w, h)                     // how near each pixel is, 0 to 1
//
//	det, err := faces.Get(ctx, rt)
//	found, err := det.Detect(ctx, frame)     // each frame of a camera
//	head, ok := vision.NearestHead(found, frame.Bounds(), vision.HeadOptions{})
//
// Their nodes (DepthNode, FacesNode) put them in a network or a call of the flow package.
// Pictures are image.Image; a camera's frames are a flow.Stream of them. Each result says what
// it took, and each model where it runs (Device, Where). A model of a definition of the app's
// (ai.Definition) may give the device, the CPU's threads, the CPU interpreter and an export's
// directory (common.go); what changes from call to call (scores, thresholds, prompts) are the
// calls' options.
package vision
