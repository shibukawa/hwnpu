# parallax

[日本語](README.ja.md) · [All samples](../README.md)

Shows a picture as if it were deep. The picture's depth is estimated once, your head is followed with the camera, and the picture is drawn from where your head is: far points move a little, near ones more, like a view through a window (Ebitengine, a Kage shader).

The window shows the picture as soon as its depth is there. Beside it the picture is parted into layers by depth and the picture model draws what each layer hid, so that where a near thing moves aside there is a picture behind it instead of its edge stretched. The window shows each step and takes the layers as they come.

```
calls, in turns:          the picture → depth;  a painted picture → what is behind   (Depth Anything 3, FLUX.2 klein)
a network, kept running:  camera → faces → head → steady → view                      (YuNet, and steps of the program)
the window:               60 times a second, with the view the network last gave
```

## Run

```bash
cd samples
go run ./parallax models prepare                   # download, convert and compile the models
go run ./parallax picture.png                      # the window, following your head with the camera
go run ./parallax picture.png --head pointer       # the mouse stands in for your head (no camera)
go run ./parallax render picture.png -o deep.gif   # no window: the picture from a head going around
```

| Command | What it does |
|---|---|
| `parallax PICTURE` | The window. `--layers 1` uses the depth alone (no picture model); `--head sweep` turns by itself; `--for 5m` closes after that long |
| `parallax DIR` | A picture already in layers (`layers PICTURE --png DIR`): no models needed |
| `parallax render PICTURE -o out.gif` | Writes the picture as seen from a head going around (`--frames`, `--width`, `--png DIR`) |
| `parallax track` | The camera part alone: prints where your head is. `--frames a.png,b.png` uses pictures instead of a camera |
| `parallax cameras` | The cameras, numbered: `--device 2` takes the second |
| `parallax models`, `parallax status` | The models and what they take; this machine's NPU |

The look: `--strength` (how far the nearest point moves), `--focus` (which depth stays put), `--far` (how far the far end of the depth is pulled apart), `--edge`, `--gain` (how far the view turns for a head's move), `--fov` (the camera's field of view).

Pictures with clear layers work best: a figure before a landscape, things at several depths.

## Download

| Model | Apple | AMD |
|---|---|---|
| YuNet (faces) | 0.2 MB | 0.2 MB |
| Depth Anything 3 Base (depth) | 0.5 GB | 2.4 GB |
| FLUX.2 klein 4B (what a layer hid) | 16 GB | 12.5 GB |
| **Total** | **16.5 GB** | **14.9 GB** |

For a machine of 16 GB; `parallax models` prints this machine's. On a Mac FLUX.2 klein is its checkpoint, converted here into two exports (7.5 GB, and 4.1 GB of the edit's programs) that stay on the disk beside it.

## What it takes

- Models: YuNet (faces, 0.2 MB), Depth Anything 3 Base (0.5 GB), FLUX.2 klein 4B (the layers).
- The depth takes a fraction of a second, a face a few milliseconds a frame. Parting into layers takes about a minute a layer on AMD's NPU; on an M3, 39 s at 512×512 and 89 s at 1024×1024.
- On Apple's Neural Engine the layers need FLUX.2 klein's edit programs, converted from the checkpoint (a 16 GB download) and compiled at the first edit (up to an hour; prebuilt programs are to take this off). `--layers 1` needs none of this.
- On a Mac the terminal app needs camera permission; macOS asks at the first run. A machine without a display or Ebitengine's libraries builds with `-tags headless` (`render` and `track` work).
