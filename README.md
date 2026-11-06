# VisionCount

Real-time human detection, multi-object tracking, directional people counting, and occupancy analytics.

## Current scope

VisionCount is being developed as a modular Python application with:

- YOLO person detection (COCO class 0)
- Ultralytics tracking with ByteTrack or BoT-SORT
- OpenCV video input and annotation
- Configurable virtual counting line
- Entry, exit, and occupancy statistics
- Streamlit dashboard for video analysis
- CSV event export

> Tracking IDs are short-term tracker identities, not guaranteed biometric identities. Occlusion, camera motion, and crowded scenes can cause ID switches or missed counts. Validate performance on representative footage before deployment.

## Architecture

```text
Video source
    |
OpenCV frame reader
    |
YOLO person detector
    |
ByteTrack / BoT-SORT
    |
Track IDs + person centroids
    |
Virtual-line crossing analytics
    |
Annotated video + metrics + CSV events
    |
Streamlit dashboard
```

## Requirements

- Python 3.10 or 3.11 recommended
- A webcam or video file
- Optional NVIDIA GPU with a compatible PyTorch CUDA installation

## Quick start (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

For CPU-only execution, leave the device setting as `cpu`. For an NVIDIA GPU, install the PyTorch build compatible with your CUDA driver, then set the device to `0` in the dashboard.

The first run downloads the configured YOLO weights (default: `yolo11n.pt`) from Ultralytics.

## Usage

1. Launch the Streamlit app.
2. Upload an MP4/AVI/MOV video.
3. Choose ByteTrack or BoT-SORT.
4. Configure the counting line and confidence threshold.
5. Run analysis and inspect the annotated output and metrics.
6. Download the crossing-event CSV.

## Project layout

```text
VisionCount/
├── app.py
├── config.yaml
├── requirements.txt
├── src/
│   └── visioncount/
│       ├── __init__.py
│       ├── analytics.py
│       ├── counter.py
│       └── detector.py
├── tests/
│   └── test_analytics.py
└── outputs/
```

## Important measurement notes

- Entry/exit counts are based on tracked centroid crossings of the configured line.
- Occupancy is calculated as entries minus exits and is clamped at zero. Incorrect or missed crossings can therefore cause occupancy drift.
- FPS is measured during processing and depends on model size, input resolution, hardware, and tracking configuration.
- Compare ByteTrack and BoT-SORT on the same videos; do not assume one is always more accurate.

## Roadmap

- [x] Initialize repository and modular project scaffold
- [ ] Validate installation and baseline inference on target hardware
- [ ] Test crossing logic on controlled entry/exit clips
- [ ] Compare ByteTrack and BoT-SORT
- [ ] Add trajectory and movement heatmap visualization
- [ ] Add webcam/RTSP reliability and reconnect handling
- [ ] Add benchmark report and broader automated tests

## License

Project code is released under the MIT License. See [LICENSE](LICENSE).
