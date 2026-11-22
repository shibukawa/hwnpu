// Package imagegen draws pictures: from a prompt (FLUX.2 klein, FLUX.3 klein, Anima, Chroma1),
// or from a prompt and reference pictures that it edits or composes from (FLUX.2 klein).
//
//	var draw = imagegen.Edit("image") // Generate draws from a prompt alone on Apple's Neural Engine
//
//	g, err := draw.Get(ctx, rt) // *imagegen.Generator
//	img, err := g.Generate(ctx, "a red fox in the snow", imagegen.Options{})
//	out, err := g.Generate(ctx, "Remove the fence of image 1.", imagegen.Options{References: []image.Image{img}})
//
// GenerateAll draws several prompts with the programs loaded once, a batch of them together
// where the model does, and returns what it took (Pictures); Options' callbacks tell each batch,
// step and stage of the VAE as they are done. A model made from another with settings of its own
// (the weight formats, Anima's conversion, the CPU, an export's directory: Export) is an
// ai.Definition (the settings are SettingTextFormat's and the others').
//
// In a network or a call the generator is a node (GenerateNode): requests in, pictures out.
//
// A picture takes tens of seconds, and the model takes several gigabytes while it draws; an app
// that draws now and then gives the memory back between (the use's Release).
package imagegen
