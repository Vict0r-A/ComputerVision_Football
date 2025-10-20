# Computer Vision Football (Work in Progress)

## Current Functionality
This is a computer vision project that takes in a football video, and uses the YOLOv8 model to perform object detection of ball, players and referee. Object tracking is performed for players using **ByteTrack** via `supervision` for ID tracking.



It uses **Ultralytics YOLO** for detection (and optional keypoints/pose), **ByteTrack** via `supervision` for ID tracking, and custom drawing utilities for clean on-frame annotations.
The link for the dataset is - 
 [Click here](https://www.kaggle.com/datasets/saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv).

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

## How to Run it

### 1) Create a Python environment
```bash
python -m venv .venv
source .venv/bin/activate   # For Windows: .venv\Scripts\activate
 
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Set up the model and input videos
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

## Skills Demonstrated
This project demonstrates several skills in Computer Vision and Machine Learning.


- Use of YOLO to perform **object detection** and **multi-object tracking** via ByteTrack.  
- Building of a YOLO model from a custom-dataset.  
- Proficiency in Python, competent in numerous libraries, such as: ultralytics, supervision, deep_sort_realtime, opencv, numpy, pandas.
- Application of **linear interpolation** to enable accurate ball tracking.  
- Structured and modular project design, with separate components for detection, tracking, and annotation.  
- Competency in version control.  
- Writing of clean maintainable code, via modularisation of the project and comments throughout the code to explain the functionality .  
- Integration of real-world datasets from external sources (Kaggle).  
- Pre-processing and organisation of video data for model inference.  

 




