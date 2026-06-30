# AirWrite: Gesture-Based Air Handwriting Recognition

Write text in the air using hand gestures and have it recognized in real-time using computer vision and handwriting recognition.

## Demo

Draw letters in the air with your index finger, trigger recognition using a hand gesture, and the recognized text is automatically typed into the active application.

---

## Features

- ✍️ Air writing using fingertip tracking
- 🖐️ Real-time hand gesture recognition with MediaPipe
- 🧠 Handwriting recognition (HTR)
- 📄 Automatic preprocessing of drawn strokes
- ⌨️ Automatic typing of recognized text
- 🧹 Gesture-based canvas clearing
- 📐 Writing guidelines for improved accuracy
- 🎥 Live webcam visualization

---

## Tech Stack

- Python
- OpenCV
- MediaPipe Tasks
- TensorFlow
- SimpleHTR
- NumPy
- SciPy
- PyAutoGUI

---

## Controls

### Right Hand

| Gesture | Action |
|---------|--------|
| Pointer | Draw |
| Open Palm | Lift pen / End current stroke |

### Left Hand

| Gesture | Action |
|---------|--------|
| Close Fist | Recognize handwriting and type text |
| OK | Clear canvas |

---

## Pipeline

```
Webcam
   │
   ▼
MediaPipe Hand Tracking
   │
   ▼
Gesture Classification
   │
   ▼
Collect Fingertip Coordinates
   │
   ▼
Render Drawing
   │
   ▼
Image Preprocessing
   │
   ▼
Handwriting Recognition
   │
   ▼
Automatic Typing
```

---

## Project Structure

```
.
├── app.py
├── hand_landmarker.task
├── SimpleHTR/
├── model/
│   ├── keypoint_classifier/
│   └── point_history_classifier/
├── utils/
├── drawing.jpg
└── data/
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/<username>/AirWrite.git
cd AirWrite
```

Install dependencies

```bash
pip install -r requirements.txt
```

Download the MediaPipe hand landmark model (`hand_landmarker.task`) and place it in the project root.

If using SimpleHTR, download the pretrained weights and place them in the expected directory.

---

## Usage

```bash
python app.py
```

Write in the green guide box using the **Pointer** gesture.

Use the **Close** gesture on your left hand to recognize and type the written text.

Use the **OK** gesture to clear the canvas.

---

## How It Works

1. Detects both hands using MediaPipe.
2. Classifies predefined hand gestures.
3. Tracks the index fingertip while the draw gesture is active.
4. Stores strokes until writing is complete.
5. Renders all strokes onto a clean canvas.
6. Crops and preprocesses the image.
7. Runs handwritten text recognition.
8. Types the recognized text into the active application.

---

## Future Improvements

- Multi-word recognition
- Sentence-level handwriting recognition
- Language model assisted decoding
- Undo gesture
- Variable brush thickness
- Custom gesture mapping
- Better stroke smoothing
- Continuous writing without explicit segmentation

---

## Acknowledgements

- MediaPipe
- OpenCV
- TensorFlow
- SimpleHTR