# Computer Vision Football (Work in Progress)

## Current Functionality
This is a computer vision project that takes in a football video, and uses the YOLOv8 model to perform object detection of ball, players and referee. Object tracking is performed for players using **ByteTrack** via `supervision` for ID tracking.


---

## Current Features 
- YOLO-based object detection (players, referees, ball)
- Player/Referee ID association using ByteTrack (`supervision.ByteTrack`)
- Linear interpolation used for ball tracking 
- A halo-like ring is used to annotate players and referee, while a triangle marker follows the top of the ball.



---

## Project Structure

```
Computer_Vision_Football/
├── main.py                       
├── object_detection/
│   ├── detection.py               
│   ├── yolo_inference.py          
│   └── keypoint_detector.py       
├── object_tracking/
│   ├── tracking.py                
│   └── annotations.py            
├── input_videos/                 # Put your input videos in this folder
├── output_videos/                # output videos appear here 
└── models/                       
    ├── best.pt                   # Models folder (you must create your own model)
```

---

## Getting Started

### 1) Create a Python environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
 
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Prepare folders & weights
Create a **models/** folder at the repo root and add your YOLO model:
- `models/best.pt` – detection model (players/referee/ball)


Add your test videos in **`input_videos/`**.  
Use the sample videos from this source to test your model: [Bundesliga 30-second clips on Kaggle](https://www.kaggle.com/datasets/saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv).


### 4) Run
```bash
python main.py
 
```

The output video will be found in **`output_videos/`**.

---

 

