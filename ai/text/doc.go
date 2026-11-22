// Package text has the models of text for an app (ai.App): translation (Translate), chat (Chat:
// tools, images and audio, answers as JSON), embeddings for search (Embed) and typed decisions
// about a state (Decide, Liquid AI's d1).
//
//	var tr = text.Translate("translate")
//	app := ai.NewApp("subtitles", tr)
//	...
//	t, err := tr.Get(ctx, rt)
//	en, err := t.Translate(ctx, "こんにちは", text.TranslateOptions{To: "en"})
//
// A chat answers with its measurements, and tells what each token adds as it goes:
//
//	c, err := chat.Get(ctx, rt) // chat = text.Chat("chat")
//	a, err := c.Reply(ctx, []text.Message{{Role: "user", Text: "Hi"}}, text.ChatOptions{}, func(d text.Delta) bool {
//		fmt.Print(d.Text) // d.Reasoning is its thinking
//		return true
//	})
//	fmt.Printf("%d tokens at %.1f tokens/s\n", a.Stats.Generated, a.Stats.TokensPerSecond())
//
// Their stream forms read and give flow.Streams of flow.Text: a translator after a transcriber,
// a speaker after a chat. The package ai/server/openai serves Chat and Embed with the OpenAI API.
package text
