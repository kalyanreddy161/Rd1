# VisionCount

Human detection, multi-object tracking, and directional people counting from uploaded videos or a live browser webcam.

## Features

- YOLO person detection (COCO class 0)
- ByteTrack or BoT-SORT tracking
- Bounding boxes, confidence scores, and tracking IDs
- Recent movement trajectories
- Configurable **horizontal or vertical** counting line
- Configurable IN direction: top-to-bottom / bottom-to-top or left-to-right / right-to-left
- Crossing debounce to reduce duplicate counts
- Occupancy estimate and processing FPS
- H.264 annotated video playback directly inside Streamlit
- IN/OUT cumulative count graph against video time
- Crossing-event CSV export
- Uploaded-video and live browser-webcam input modes

> Tracking IDs are short-term tracker identities, not guaranteed biometric identities. Occlusion, camera motion, and crowded scenes can cause ID switches or missed counts.

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

Select `cpu` for CPU inference. Select GPU `0` only after installing and verifying CUDA-enabled PyTorch. The first run downloads the configured YOLO weights (default: `yolo11n.pt`).

## Counting line configuration

The sidebar lets the operator choose:

- **Horizontal:** spans the image width; position is measured from the top.
- **Vertical:** spans the image height; position is measured from the left.
- **IN direction:** choose which side-to-side movement represents entry. The reverse crossing is OUT.
- **Crossing debounce:** minimum time between counted crossings for the same track.

Default IN direction is top-to-bottom for a horizontal line and left-to-right for a vertical line. Change this to match the camera's actual traffic flow.

## Input modes

### Upload video

Upload MP4, AVI, MOV, or MKV, set the line/tracker options, and run analysis. The annotated H.264 video plays in the app. The timeline graph shows cumulative IN and OUT events against video time. Crossing events can be exported as CSV.

### Live webcam

Select Live webcam, press START, and allow browser camera access. The annotated stream is processed live. Browser camera access requires localhost or HTTPS. Network/firewall restrictions may require TURN configuration.

## Counting interpretation and limitations

- The system counts tracked centroid transitions across the configured line.
- IN/OUT are cumulative crossing counts. Occupancy is estimated as IN minus OUT.
- Occupancy may be wrong if a video starts mid-session or tracking misses a crossing.
- A full-width/full-height infinite line is a baseline. For angled paths, narrow doorways, or multiple traffic directions, a finite segment with a defined counting region and direction vector is a stronger next step.
- Tracking IDs can switch under occlusion or crowded conditions; they are not persistent biometric identities.

## Tests

```powershell
python -m unittest discover -s tests -v
```

Test both line orientations with footage where the traffic direction is known before relying on counts.
