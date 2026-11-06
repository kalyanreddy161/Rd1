# VisionCount

Human detection, multi-object tracking, and directional people counting from uploaded videos or a live browser webcam.

## Features

- YOLO person detection (COCO class 0)
- ByteTrack or BoT-SORT tracking
- Bounding boxes, confidence scores, and tracking IDs
- Recent movement trajectories
- Configurable virtual counting line with IN/OUT counts
- Occupancy estimate and processing FPS
- H.264 annotated video playback directly inside the Streamlit app
- IN/OUT cumulative count graph against the video timeline
- Crossing-event CSV export
- Two input modes: uploaded video and live webcam

> Tracking IDs are short-term tracker identities, not guaranteed biometric identities. Occlusion, camera motion, and crowded scenes can cause ID switches or missed counts.

## Requirements

- Python 3.10 or 3.11 recommended
- Webcam for live mode
- Optional NVIDIA GPU with a compatible CUDA-enabled PyTorch installation

## Windows setup

```powershell
git clone https://github.com/kalyanreddy161/Rd1.git
cd Rd1
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

For CPU-only execution, select `cpu` in the sidebar. Select `0` only after confirming CUDA-enabled PyTorch is installed and working.

The first run downloads the configured YOLO weights (default: `yolo11n.pt`).

## Input modes

### Upload video

1. Select **Upload video**.
2. Upload an MP4, AVI, MOV, or MKV file.
3. Choose ByteTrack or BoT-SORT and configure the confidence threshold and counting line.
4. Click **Run detection and tracking**.
5. Play the processed H.264 video directly in the app.
6. Inspect the IN/OUT timeline graph and crossing-event table.

The processed video is not exposed as a download button. H.264 encoding is used for browser compatibility.

### Live webcam

1. Select **Webcam (live)**.
2. Press **START** in the webcam panel.
3. Allow browser camera permission.
4. Inspect the live annotated stream.
5. Press **STOP** to end the stream.

Webcam access is provided by WebRTC. It works on localhost and HTTPS deployments. Remote deployments over plain HTTP generally cannot request browser camera access. Network/firewall restrictions may also require TURN server configuration.

## Counting interpretation

- The implementation counts tracked centroid transitions across a horizontal line.
- Above-to-below is treated as IN; below-to-above is OUT. Adjust camera placement or code convention as needed.
- The graph plots cumulative IN and OUT crossing counts against video time in seconds.
- The graph samples once per second and adds a point at each crossing.
- Occupancy is calculated as entries minus exits and clamped at zero. It is an estimate; a video may start mid-session or tracking may miss crossings.

## Project layout

```text
Rd1/
├── app.py
├── config.yaml
├── requirements.txt
├── src/
│   └── visioncount/
│       ├── analytics.py
│       ├── counter.py
│       ├── detector.py
│       └── live.py
├── tests/
└── outputs/
```

## Validation status

The repository contains unit tests for line-crossing analytics. Run the application locally and test on representative footage before relying on the counts. FPS and counting accuracy depend on hardware, video resolution, camera angle, and scene density.
