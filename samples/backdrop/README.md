# backdrop

[日本語](README.ja.md) · [All samples](../README.md)

Blurs what is behind you in the camera's picture, or puts a picture or a colour there, as a video call does. In every frame the person is told from what is behind them (MediaPipe Selfie Segmentation), and the window draws the frame with its matte (Ebitengine, Kage shaders).

```
a network, kept running:  camera → cutout → steady → view   (MediaPipe Selfie Segmentation, and steps of the program)
the window:               60 times a second, with the frame and the matte the network last gave
```

The model runs on the CPU, on one thread: 5.9 ms for a 1280×720 frame on a Ryzen AI 9 365, so every frame of the camera gets its matte. It is 0.46 MB of weights, too small to pay on an NPU.

## Run

```bash
cd samples
go run ./backdrop models prepare               # download the model (0.5 MB)
go run ./backdrop                              # the window: the camera, what is behind you blurred
go run ./backdrop --behind beach.jpg           # a picture behind you
go run ./backdrop --behind "#00aa44"           # a colour: a green screen
go run ./backdrop --frames a.png,b.png         # pictures in place of the camera (a machine without one)
```

| Command | What it does |
|---|---|
| `backdrop` | The window. `--behind blur\|FILE\|#RRGGBB`, `--blur 0.02` (how far the blur reaches, as a share of the frame's width), `--steady 0.5` (how much of the matte before stays in each frame's), `--no-mirror`, `--size 1280x720`, `--for 30s` |
| `backdrop cameras` | The cameras, numbered: `--device 2` takes the second |
| `backdrop models`, `backdrop status` | The model and what it takes; this machine's NPU |

In the window: B goes through what is behind (blurred, the picture or the colour, the frame as it is), M shows the matte, up and down change the blur, Q quits.

## How what is behind is blurred

A blur of the frame itself would spread your colours around you. So the frame is first weighed by how much of each pixel is not a person (one less the matte), that picture is made a quarter of the size and blurred there, and the result is divided by the blurred weight: only what is behind you is mixed. The matte is the model's size (256×256) and is stretched on the GPU.

## What it takes

- Model: MediaPipe Selfie Segmentation (Google, Apache-2.0; 0.5 MB). `-m` is not a flag: the app uses the definitions' `cutout`.
- ONNX Runtime's shared library, for the CPU: the system's (Ubuntu's `libonnxruntime1.23` is found), or set `HWNPU_ONNXRUNTIME` to `libonnxruntime.so` of a [release](https://github.com/microsoft/onnxruntime/releases).
- The model is made for a person before a camera: a frame without one gives a matte of whatever stands out, and thin things (spread hair, fingers) are not followed.
- It was checked on an AMD machine with pictures given as the frames, in a display drawn by software. A real camera, a real display and a Mac were not tried.
- A machine without a display or Ebitengine's libraries builds with `-tags headless` (`cameras`, `models` and `status` work).
