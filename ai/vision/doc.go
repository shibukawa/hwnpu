// Package vision has the models of pictures for an app (ai.App): the depth of a picture (Depth)
// and the faces in one (Faces), with the position of a head worked out from its face (Head).
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
// Pictures are image.Image; a camera's frames are a flow.Stream of them.
package vision
