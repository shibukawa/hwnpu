// Command kleinonly writes FLUX.2 klein's Core AI programs without drawing, for checking a change
// to how parts are written.
//
//	go run ./tools/coreai/kleinonly parts ROOT OUT singles[,doubles,...]   flux2klein.Convert's Only, 512² and 1024²
//	go run ./tools/coreai/kleinonly prepare [WxH]                         hwnpu-image's preparation of the export for the size
//	                                                                       (fetched, or converted and written back, with entry.json); prints its folder
//	go run ./tools/coreai/kleinonly put EXPORT                            put an export made here in the sources written back to
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/imagegen/flux2klein"
	"github.com/shibukawa/hwnpu/internal/llm"
	shared "github.com/shibukawa/hwnpu/internal/store"
)

func main() {
	if len(os.Args) < 2 {
		log.Fatal("usage: kleinonly parts ROOT OUT KINDS | prepare [WxH]")
	}
	// the definitions and the sources of the environment (HWNPU_MODEL_SOURCE), as hwnpu-image reads them
	if err := chat.LoadUserModels("", os.Stderr); err != nil {
		log.Fatal(err)
	}
	switch os.Args[1] {
	case "parts":
		only := map[string]bool{}
		for _, k := range strings.Split(os.Args[4], ",") {
			only[k] = true
		}
		o := flux2klein.ConvertOptions{Width: 1024, Height: 1024, Only: only}
		if err := o.FromRecipe("FLUX.2-klein-4B"); err != nil {
			log.Fatal(err)
		}
		if err := flux2klein.Convert(os.Args[2], os.Args[3], o, os.Stderr); err != nil {
			log.Fatal(err)
		}
	case "prepare":
		w, h := 1024, 1024
		if len(os.Args) > 2 {
			if _, err := fmt.Sscanf(os.Args[2], "%dx%d", &w, &h); err != nil {
				log.Fatal(err)
			}
		}
		m, ok := chat.Lookup("FLUX.2-klein-4B")
		if !ok {
			log.Fatal("FLUX.2-klein-4B is not in models.yaml")
		}
		spec := m.NPU["apple"]
		o := flux2klein.ConvertOptions{Width: w, Height: h, TextFormat: spec.Text, DitFormat: spec.Dit}
		ctx := chat.WithModel(context.Background(), m)
		dir, err := flux2klein.Prepare(ctx, flux2klein.Setup{Dir: filepath.Join(chat.ModelsDir(), "FLUX.2-klein"), Repo: m.Repo,
			Name: m.Name, ConvertOptions: o, Log: console.Log(os.Stderr)})
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(dir)
	case "put":
		m, _ := chat.Lookup("FLUX.2-klein-4B")
		fmt.Fprintf(os.Stderr, "%d sources written back to\n", len(shared.Writebacks()))
		llm.ShareExport(chat.WithModel(context.Background(), m), os.Args[2])
	default:
		log.Fatalf("unknown command %q", os.Args[1])
	}
	shared.Wait(console.Log(os.Stderr)) // the puts in the background
}
