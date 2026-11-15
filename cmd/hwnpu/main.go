// hwnpu looks after what hwnpu keeps on this machine: the models downloaded and the programs
// converted from them (the cache, docs/cache-sharing.md 7). The models are run by hwnpu-chat,
// hwnpu-image and hwnpu-audio.
//
//	hwnpu cache                              # where the cache is, what it holds, the disk's free space
//	hwnpu cache ls                           # each unit: its size and when it was last used
//	hwnpu cache prune --max-size 200GB       # remove the least recently used until it fits
//	hwnpu cache prune --older-than 30d       # remove what was not used for 30 days
//	hwnpu cache clear Qwen3-1.7B             # remove a model's files and programs
//	hwnpu cache clear --all                  # remove everything the cache holds of the models
//
// The cache is hwnpu in the user cache directory (~/.cache/hwnpu, ~/Library/Caches/hwnpu,
// %LOCALAPPDATA%\hwnpu), or HWNPU_CACHE. It is held in units, each removed whole: a repository's
// files (original/OWNER/NAME), a model's programs (models/NAME, converted/...), an export for
// AMD's NPU (entries/...), a kernel IRON compiled (iron/KEY) and a program compiled for this
// machine (KEY.mlmodelc...). What is removed is downloaded or made again when it is needed.
//
// Downloads keep HWNPU_CACHE_MIN_FREE (10GB unless set, 0 for none, or a percentage of the disk)
// free on the cache's disk: one that would leave less is not started. HWNPU_CACHE_MAX_SIZE
// (200GB) is the most the cache holds: a download that would go over, or leave the disk short,
// removes the units least recently used first. Nothing used in the last hour is removed by itself
// or by prune (another process may be reading it).
package main

import (
	"context"

	"github.com/alecthomas/kong"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/console"
)

type cli struct {
	Cache CacheCmd `cmd:"" help:"What the cache of models holds, the disk's free space; remove what is not used."`
}

func main() {
	ctx, stop := console.Interrupt()
	defer stop()
	var c cli
	k := kong.Parse(&c,
		kong.Name("hwnpu"),
		kong.Description("Look after what hwnpu keeps on this machine: the models downloaded and the programs converted from them."),
		kong.UsageOnError(),
		kong.Vars{"kinds": kindHelp, "max_size_env": cache.MaxSizeEnv},
		kong.BindTo(ctx, (*context.Context)(nil)))
	k.FatalIfErrorf(console.EndInterrupted(k.Run()))
}
