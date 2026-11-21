// Package text has the models of text for an app (ai.App): translation (Translate) and chat
// (Chat).
//
//	var tr = text.Translate("translate")
//	app := ai.NewApp("subtitles", tr)
//	...
//	t, err := tr.Get(ctx, rt)
//	en, err := t.Translate(ctx, "こんにちは", text.TranslateOptions{To: "en"})
//
// Their stream forms read and give flow.Streams of flow.Text: a translator after a transcriber,
// a speaker after a chat.
package text
