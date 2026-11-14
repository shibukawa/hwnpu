// hwnpu-chat runs language models on an NPU: chat models (LFM3-1.7B) to talk with or to work as a
// coding agent, and embedding models (PLaMo-Embedding-1B) to turn texts into vectors for search.
//
//	hwnpu-chat                                        # talk in the terminal
//	hwnpu-chat "Say hello in one word"                # one answer on standard output, then exit
//	hwnpu-chat -m translate "こんにちは"                 # PLaMo 3 Translate, size picked by memory
//	hwnpu-chat -m LFM3-VL-2B --image photo.jpg "What is in it?"
//	hwnpu-chat agent "fix the bug in the add function of main.go"
//	hwnpu-chat agent -C ~/src/demo --yes "add a mul function"
//	hwnpu-chat search -m plamo-embedding-1b           # search the texts typed by their embeddings
//	hwnpu-chat serve -m LFM3-1.7B,plamo-embedding-1b  # the OpenAI compatible API
//	hwnpu-chat opencode [-- opencode arguments]       # OpenCode on the server
//	hwnpu-chat systemone request.json                 # d1-2B's decisions on a state (SystemOneCmd)
//	hwnpu-chat list
//
// chat (the command when none is named) and agent take a prompt or a task on the command line,
// answer it and exit; without one they take them in the terminal. What the user asked for (the
// answer, the server's address) is printed as it is; what goes on (downloads, conversions,
// loading) is logged in gray, with "all on the NPU" in green and "NOT all on the NPU" in red for
// each program loaded.
//
// --model (-m) takes a name, a family or a use of models.yaml (LFM3-1.7B, chat, translate,
// plamo3translate:2b; a family alone picks for the machine's memory), a Hugging Face repository
// OWNER/NAME, or an export's directory (a value with a / or a leading . or ~: on AMD's NPU, one of
// tools/amd/iron_export_lm.py). list shows the models.
//
// A model lives in --dir/NAME (hwnpu/models/NAME in the user cache directory). On Apple's Neural
// Engine and Intel's NPU the first run downloads it from Hugging Face and converts it; later runs
// load what is there. A chat model is converted twice with a KV cache of --context tokens,
// --draft (8) tokens per call to generate (a token and the guesses after it, checked at once) and
// 64 to read prompts about 20 times faster; the two share one cache, and a conversation reads
// only what is new since the last answer. LFM3 is split at its full attention layers (--paged):
// hwnpu-chat attends over the cache in chunks of 8192 itself, so that a step costs in proportion
// to the conversation and the cache's memory grows with it, and the context is the model's
// longest, 262144 tokens. A model that is not split, or --paged -1, costs in proportion to
// --context (16320) whatever the conversation holds; with --tier (4096) its pair is converted for
// the shorter cache too, and a conversation runs on the shortest cache that holds it, moving up
// as it grows. An embedding model is converted once per length in --embed-seq, and a text runs on
// the shortest that holds it. On AMD's NPU a name runs ~/models/NAME.amd-npu2-q4 (else
// NAME.amd-npu2), exports of tools/amd/iron_export_lm.py (docs/amd-npu-models.md).
//
// --package (or --embed-package) takes programs converted by hwnpu-dev convert instead, separated by
// commas: for a chat model with the same -cache and different -seq, for an embedding model with
// different -seq.
//
// LFM3-VL-2B, LFM3-Audio-2B and LFM3-Omni-2B read images and audio: their first run also converts
// the encoders of the checkpoint, the vision encoder for images of 512x512 and the audio encoder
// for clips of up to 30 seconds. --image and --audio send them with a prompt; in the terminal
// /image FILE and /audio FILE.wav do; image_url and input_audio parts of the API's user messages
// are read. With --package, --vision takes vision encoders converted by hwnpu-dev convert -arch
// lfm3-vision (a package per image size) and --audio-encoder audio encoders (-arch
// lfm3-audio-encoder, a package per clip length). Speech output is not supported (see hwnpu-audio tts).
//
// agent is a coding agent: a chat model with four tools, read, edit, write and bash, in the loop
// of Pi (pi.dev) written in Go on the same engine (see AgentCmd). opencode serves on a free port
// from 30000 (or --listen), writes an OpenCode config for the server's chat models, and runs
// OpenCode on this terminal with the arguments after --; when it exits, so does the server.
package main

import (
	"context"
	"os"

	"github.com/alecthomas/kong"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
)

type cli struct {
	Chat      ChatCmd      `cmd:"" default:"withargs" help:"Talk with a chat model, or answer the prompt given and exit (the command when none is named)."`
	Agent     AgentCmd     `cmd:"" help:"A coding agent with read, edit, write and bash: run the task given, or take tasks in the terminal."`
	Search    SearchCmd    `cmd:"" help:"Search the texts typed by their embeddings (an embedding model)."`
	Serve     ServeCmd     `cmd:"" help:"Serve the OpenAI compatible API: chat completions and embeddings."`
	OpenCode  OpenCodeCmd  `cmd:"" name:"opencode" help:"Serve the chat models and run OpenCode on them in this terminal."`
	SystemOne SystemOneCmd `cmd:"" name:"systemone" help:"Answer typed questions about a state with a d1 decision model: the System One API of llama.cpp's server (Apple's Neural Engine)."`
	List      ListCmd      `cmd:"" help:"List the chat and embedding models, what each is for and whether it is downloaded."`
}

func main() {
	ctx, stop := console.Interrupt() // Ctrl-C ends the command, and the process when the command does not return
	defer stop()
	var c cli
	k := kong.Parse(&c,
		kong.Name("hwnpu-chat"),
		kong.Description("Language models on an NPU: chat, a coding agent, search by embeddings and the OpenAI compatible API."),
		kong.UsageOnError(),
		kong.Vars{
			"models_usage": chat.ModelsUsage,
			"tiers":        chat.DefaultTiers,
			"embed_seqs":   ints(chat.DefaultEmbedSeqs),
			"mtp":          chat.MTPAuto,
		},
		kong.BindTo(ctx, (*context.Context)(nil)),
		kong.Bind(stop))
	console.MeasureEnergy(console.Log(os.Stderr)) // HWNPU_ENERGY: the idle power, then each phase's energy in the last lines
	k.FatalIfErrorf(console.EndInterrupted(k.Run()))
	console.Done(console.Log(os.Stderr)) // the times, when the command said console.Running; HWNPU_PROFILE's profile
}
