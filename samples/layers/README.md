# layers

[日本語](README.ja.md) · [All samples](../README.md)

Parts a picture into layers by depth and writes them as a Photoshop document (PSD), with what each layer hid drawn in. In the document a layer can be moved, scaled or hidden, and there is a picture behind it.

```
the picture → depth → cut the nearest layer out → paint it over → draw what is behind
    ↑                                                                     |
    └───────────────────── the picture without that layer ←───────────────┘
```

The nearest layer is cut out, where it was is painted in one plain colour, and the picture model is asked to draw the background there (FLUX.2 klein, an edit). The depth of that picture gives the next layer, and so on to the farthest, which is whole. The two models run in turns, each unloaded before the other is asked, so they need not fit in memory together.

## Run

```bash
cd samples
go run ./layers models prepare                   # download, convert and compile the models
go run ./layers plan picture.png                 # where it would cut, as a picture coloured by layer (no drawing)
go run ./layers picture.png -o picture.psd       # part it into 3 layers
go run ./layers picture.png --layers 4 --png parts/
```

| Option | What it does |
|---|---|
| `--layers N` | How many layers, 2 to 8 (default 3); the cuts are found from the depth |
| `--cuts 0.3,0.7` | Cut at these nearnesses instead (0 the farthest, 1 the nearest) |
| `--png DIR` | Also writes each layer (`layer-N.png`), its nearness (`near-N.png`, 16-bit), and what the model was asked and drew. `parallax DIR` shows these |
| `--hint "no people"` | Words added to the request to draw what a layer hid |
| `--tries 3` | When the model leaves the layer's shape behind (a flat colour or a shadow), it draws again from another seed |
| `--seed`, `--edge`, `--clean`, `--far` | The seed; how far a layer reaches past its cut; how far past it the drawn picture is used; how the far end of the depth is spread |

## Download

| Model | Apple | AMD |
|---|---|---|
| Depth Anything 3 Base (depth) | 0.5 GB | 2.4 GB |
| FLUX.2 klein 4B (what a layer hid) | 16 GB | 12.5 GB |
| **Total** | **16.5 GB** | **14.9 GB** |

For a machine of 16 GB; `layers models` prints this machine's. On a Mac FLUX.2 klein is its checkpoint, converted here into two exports (7.5 GB, and 4.1 GB of the edit's programs) that stay on the disk beside it.

## What it takes

- Models: Depth Anything 3 Base (0.5 GB) and FLUX.2 klein 4B.
- A layer's drawing: about a minute at 1024×1024 on AMD's NPU (three layers, two drawings: 2 to 3.5 minutes); on an M3, 39 s at 512×512 and 89 s at 1024×1024.
- On Apple's Neural Engine drawing from a reference needs FLUX.2 klein's edit programs, converted from the checkpoint (a 16 GB download) and compiled at the first edit (up to an hour; prebuilt programs are to take this off).
- The picture is drawn at the model's size nearest its shape and stretched back. The PSD is 8-bit RGB with layers of the document's size.
