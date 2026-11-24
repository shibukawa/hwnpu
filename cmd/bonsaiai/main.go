// bonsaiai looks after bonsaiai on this machine: the NPU and what uses it, the models downloaded and
// the programs converted from them (the cache, docs/cache-sharing-design.md 7), and reports of problems.
// The models are run by bonsaiai-chat, bonsaiai-image and bonsaiai-audio.
//
//	bonsaiai status                             # the NPU (driver, firmware, how busy, power), what uses it, the cache
//	bonsaiai status -w 2s                       # again every 2 seconds
//	bonsaiai issue                              # a draft issue about the last command that failed, opened in the browser
//	bonsaiai cache                              # where the cache is, what it holds, the disk's free space
//	bonsaiai cache ls                           # by model and kind: the units, size, when last used (-u: each unit)
//	bonsaiai cache prune --max-size 200GB       # remove the least recently used until it fits
//	bonsaiai cache prune --older-than 30d       # remove what was not used for 30 days
//	bonsaiai cache clear Qwen3-1.7B             # remove a model's files and programs (and Core AI's compiled ones)
//	bonsaiai cache clear --all                  # remove everything the cache holds of the models
//	bonsaiai memory                             # this machine's memory, the models with a memory chosen
//	bonsaiai memory LFM3-1.7B                   # the memory it runs as, and how
//	bonsaiai memory LFM3-1.7B 8                 # run it as on a machine of 8 GB, in every command
//	bonsaiai memory LFM3-1.7B --forget          # as on this machine again
//	bonsaiai service install                    # bonsaiaid as a service (LaunchAgent, systemd user units): it starts at the first connection
//	bonsaiai service status                     # whether bonsaiaid runs, its settings, what it holds
//	bonsaiai service start|stop|restart         # not installed: start runs bonsaiaid here, in the foreground
//	bonsaiai config                             # the settings in effect and where each comes from (config keys: all there are)
//	bonsaiai config set model-source webdav://nas/llm  # kept in ~/.config/bonsaiai/.env; bonsaiaid applies it at once
//	bonsaiai config get model-source
//	bonsaiai config remove model-source
//	bonsaiai config add model-allowlist +nsfw   # a list: an item at a time
//	bonsaiai config get model-allowlist         # its items, numbered: 1: +nsfw
//	bonsaiai config remove model-allowlist 1    # by its number (or the item itself)
//
// The cache is bonsaiai in the user cache directory (~/.cache/bonsaiai, ~/Library/Caches/bonsaiai,
// %LOCALAPPDATA%\bonsaiai), or BONSAIAI_CACHE. It is held in units, each removed whole: a repository's
// files (original/OWNER/NAME), a model's programs (models/NAME, converted/...), an export for
// AMD's NPU (entries/...), a kernel IRON compiled (iron/KEY) and a program compiled for this
// machine (KEY.mlmodelc...). What is removed is downloaded or made again when it is needed.
//
// Downloads keep BONSAIAI_CACHE_MIN_FREE (10GB unless set, 0 for none, or a percentage of the disk)
// free on the cache's disk: one that would leave less is not started. BONSAIAI_CACHE_MAX_SIZE
// (200GB) is the most the cache holds: a download that would go over, or leave the disk short,
// removes the units least recently used first. Nothing used in the last hour is removed by itself
// or by prune (another process may be reading it).
//
// status reads what the OS tells of the NPU (npuinfo) as it is, and does not wake it: the NPU and
// its vendor are known from the driver (Linux), the I/O Registry (macOS) or the compute-only
// adapter (Windows), and the devices are not opened. --wake opens them and asks the NPU too (on
// Linux amdxdna wakes for a few seconds). The processes that use it are listed with the NPU's
// memory each holds.
//
// cache and memory run in bonsaiaid (internal/cachecmd) unless BONSAIAI_DAEMON is off: bonsaiaid reads its
// own cache, removes what it does not hold loaded, and is the only process that removes; status
// shows bonsaiaid's cache too.
//
// memory chooses the memory a model runs as on this machine: how a model runs on a machine of
// some memory (its programs: models.yaml's variants) is the definition's, and the user chooses
// only the machine, to run a model the smaller way beside other programs on a machine of more.
// Every command that resolves the model (bonsaiai-chat, bonsaiai-image, bonsaiai-audio, an application
// through modelstore) runs it so, and converts or fetches the programs of that memory alone. It
// is kept in run.json in bonsaiai's directory of settings (beside the user's models.yaml), not in
// the cache. bonsaiai-chat --memory GB (or a tag of memory: lfm3:8g) chooses it too.
//
// bonsaiai-chat, bonsaiai-image and bonsaiai-audio keep their last 10 runs in logs/ under the cache (the
// arguments, the log, how it ended). issue writes a draft of an issue of shibukawa/bonsaiai about the
// last that failed in the past week (or --run N of --list), with the last lines of its log and
// status, the home folder, the user's and the machine's names, the sources' addresses, a URL's
// password and tokens hidden, a long text argument (a prompt) replaced by its length, and the
// processes' command lines left out. It shows the draft, keeps it in logs/issue-TIME.md, and opens
// GitHub's page of a new issue with it filled in (open, xdg-open, or Windows' URL handler):
// nothing is sent until the button there is pressed.
package main

import (
	"context"
	"os"

	"github.com/alecthomas/kong"

	"github.com/shibukawa/bonsaiai/internal/cache"
	"github.com/shibukawa/bonsaiai/internal/cachecmd"
	"github.com/shibukawa/bonsaiai/internal/console"
	"github.com/shibukawa/bonsaiai/internal/npustatus"
	"github.com/shibukawa/bonsaiai/internal/servicecmd"
	shared "github.com/shibukawa/bonsaiai/internal/store"
)

type cli struct {
	Status  npustatus.Cmd      `cmd:"" help:"This machine's NPU: the device, its driver, how busy it is and what uses it; what bonsaiai runs on; the cache."`
	Cache   cachecmd.CacheCmd  `cmd:"" help:"What the cache of models holds, the disk's free space; remove what is not used."`
	Memory  cachecmd.MemoryCmd `cmd:"" help:"The memory each model runs as on this machine, in every bonsaiai command: choose a smaller one (bonsaiai memory LFM3-1.7B 8) to run a model beside other programs, the way models.yaml defines for that memory."`
	Service servicecmd.Cmd     `cmd:"" help:"bonsaiaid, which owns the NPU and the models for the commands and the apps: install (as a service that starts at the first connection), status, start (not installed: in the foreground), stop, restart, uninstall."`
	Config  ConfigCmd          `cmd:"" help:"bonsaiai's settings by key, as git config (model-source, model-allowlist: config keys): set, get and remove; add, get and remove N of a list. Kept in ~/.config/bonsaiai/.env, which bonsaiaid reads and applies at once."`
	Issue   IssueCmd           `cmd:"" help:"Draft an issue about the last command that failed, with this machine's status (names and secrets hidden), and open it in the browser to check and send."`
}

func main() {
	ctx, stop := console.Interrupt()
	defer stop()
	var c cli
	k := kong.Parse(&c,
		kong.Name("bonsaiai"),
		kong.Description("Look after what bonsaiai keeps on this machine: the models downloaded and the programs converted from them."),
		kong.UsageOnError(),
		kong.Vars{"kinds": cachecmd.KindHelp, "max_size_env": cache.MaxSizeEnv},
		kong.BindTo(ctx, (*context.Context)(nil)))
	err := console.EndInterrupted(k.Run())
	shared.Wait(console.Log(os.Stderr)) // what was made here goes in the sources written back to
	k.FatalIfErrorf(err)
}
