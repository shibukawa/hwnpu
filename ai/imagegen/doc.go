// Package imagegen draws pictures: from a prompt, or from a prompt and reference pictures that
// it edits or composes from (FLUX.2 klein).
//
//	var draw = imagegen.Generate("image")
//
//	g, err := draw.Get(ctx, rt) // *imagegen.Generator
//	img, err := g.Generate(ctx, "a red fox in the snow", imagegen.Options{})
//	out, err := g.Generate(ctx, "Remove the fence of image 1.", imagegen.Options{References: []image.Image{img}})
//
// In a network or a call the generator is a node (GenerateNode): requests in, pictures out.
//
// A picture takes tens of seconds, and the model takes several gigabytes while it draws; an app
// that draws now and then gives the memory back between (the use's Release).
package imagegen
