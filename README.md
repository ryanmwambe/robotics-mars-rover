# Mars Rover — Raspberry Pi Vision Detection

Live object-detection streams for a Mars rover competition, running on a **Raspberry Pi** with an **IMX500 camera**. Each task exposes an MJPEG web stream you can open in a browser on the same network.

## Features

| Script | Target | Port | Method |
|--------|--------|------|--------|
| `traffic_cone.py` | Traffic cone | **5000** | YOLO (`models/traffic_cone.pt`) |
| `hammer.py` | Hammer | **5001** | YOLO + ROI filters (`models/hammer.pt`) |
| `tennis_ball.py` | Tennis ball | **5002** | YOLO (`models/tennis_ball.pt`) |
| `balloon.py` | 5 coloured balloons | **5003** | **Hybrid YOLO + HSV** |

Open streams at `http://<pi-ip>:<port>` (e.g. `http://192.168.137.195:5003`).

---

## Hardware

- Raspberry Pi (tested on Pi 5 / BCM2712)
- Sony IMX500 camera module (`picamera2` / libcamera)
- Optional: place the Pi on the same network as your laptop/phone for browser viewing

---

## Setup

### 1. System packages

```bash
sudo apt update
sudo apt install -y python3-venv python3-picamera2
```

> **Note:** Do not install Python packages with system `pip` on Raspberry Pi OS (PEP 668). Always use the project virtual environment below.

### 2. Virtual environment

```bash
cd ~/robotics-mars-rover
python3 -m venv venv
source venv/bin/activate
```

### 3. Python dependencies

Install core packages (CPU-only PyTorch for Pi — no CUDA):

```bash
pip install -r requirements.txt
pip install ultralytics --no-deps
```

`requirements.txt` pins `torch==2.7.1` from piwheels to avoid pulling large NVIDIA/CUDA libraries that are useless on a Pi.

`picamera2` comes from the system package (`python3-picamera2`), not pip. The scripts automatically add `/usr/lib/python3/dist-packages` to `sys.path` when running inside the venv.

### 4. Model files

Place YOLO weights under `models/`:

```
models/
├── hammer.pt
├── tennis_ball.pt
├── traffic_cone.pt
└── ballons/models/          # note: folder name is "ballons"
    ├── yellow_balloon.pt
    ├── pink_balloon.pt
    ├── light_blue_balloon.pt
    ├── white_balloon.pt
    ├── navy_blue_balloon.pt   # used as shape finder for the "black" slot
    └── black_balloon.pt       # spare / alternate
```

---

## Running detection

Activate the venv, then run one script at a time (each uses the camera):

```bash
source venv/bin/activate
python traffic_cone.py   # port 5000
python hammer.py         # port 5001
python tennis_ball.py    # port 5002
python balloon.py        # port 5003
```

Or use the helper for traffic cones:

```bash
./run.sh
```

### Stopping

- **Ctrl+C** once in the terminal (clean shutdown), or
- Click **Stop detection** on the balloon web page (`/stop` endpoint).

---

## Balloon detection (hybrid YOLO + HSV)

The competition requires five balloons in a fixed row:

| Label on screen | Colour |
|-----------------|--------|
| yellow balloon | Yellow |
| pink balloon | Pink |
| light blue balloon | Light blue / cyan |
| white balloon | White |
| black balloon | Navy blue stand-in (when no true black balloon is available) |

### Why hybrid?

| Approach | Strength | Weakness |
|----------|----------|----------|
| **HSV only** | Accurate colours | Picks up non-balloon coloured objects |
| **YOLO only** | Finds balloon shapes | Poor / inconsistent colour labels |
| **Hybrid** | YOLO finds balloons; HSV names the colour | Best of both |

### Pipeline

1. **YOLO (shape)** — Each balloon position has a dedicated model. A cropped region around the expected location is scanned for a round balloon shape.
2. **HSV (colour)** — Pixels inside each YOLO box are sampled; the dominant colour vote becomes the label (`yellow balloon`, etc.).
3. **Combined score** — `55%` shape confidence + `45%` colour vote ratio.

Search windows assume the five balloons stay in a left-to-right row on the table. If you move them significantly, adjust `SEARCH_WINDOWS` in `balloon.py`.

### Calibration / test

With all five balloons in front of the camera:

```bash
source venv/bin/activate
python calibrate_balloons.py
```

This captures one frame, runs detection, saves debug images to `debug_balloons/`, and reports `Detected N/5`. Exit code `1` if any balloon is missing.

---

## Project layout

```
robotics-mars-rover/
├── README.md
├── requirements.txt
├── run.sh                 # launches traffic_cone.py via venv
├── traffic_cone.py
├── hammer.py
├── tennis_ball.py
├── balloon.py             # hybrid balloon detection
├── calibrate_balloons.py  # one-shot balloon test
└── models/                # YOLO .pt weights (not in git if LFS/ignored)
```

Generated / local only (gitignored): `venv/`, `debug_balloons/`, `__pycache__/`.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `externally-managed-environment` | Use `venv/bin/pip`, not system `pip` |
| `ModuleNotFoundError: cv2` | Activate venv and install `requirements.txt` |
| `No module named picamera2` | `sudo apt install python3-picamera2` |
| Camera busy | Stop other scripts using the camera |
| Balloon model misses | Lower `SHAPE_CONFIDENCE` in `balloon.py`; ensure balloons are in the expected row |
| Slow balloon stream | Hybrid runs 5 cropped YOLO inferences per frame (~1–2 FPS on Pi) |

---

## License

Competition / educational use. Model weights and code as provided in this repository.
