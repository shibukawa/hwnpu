// Package vision has the models of pictures for an app (ai.App): the depth of a picture (Depth),
// the faces in one (Faces), with the position of a head worked out from its face (Head), and the
// people of one apart from what is behind them (Cutout).
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
//	cutter, err := vision.Cutout("cutout").Get(ctx, rt)
//	matte, err := cutter.Matte(ctx, frame)   // how much of each pixel is a person, 0 to 1
//	alone := matte.Cut(frame)                // the frame with what is behind made clear
//
// Their nodes (DepthNode, FacesNode, CutoutNode) put them in a network or a call of the flow package.
// Pictures are image.Image; a camera's frames are a flow.Stream of them.
package vision
