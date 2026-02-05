AI Virtual Painter
Draw, Write, or Present on a Digital Canvas Using Hand Gestures and Your Webcam!

Overview
AI Virtual Painter is a real-time, Python-based drawing and presentation tool that lets you write or draw on the screen using only your hand movements — no stylus or touchscreen required! Powered by OpenCV and MediaPipe, it turns your webcam into a virtual whiteboard or blackboard for learning, teaching, or creative art, using advanced gesture detection.

Demo

Features
Draw/Write with Your Finger: Point your index finger to draw, just like a real marker.

Erase with Hand Gestures: Hold up index+middle fingers to activate eraser mode.

Pinch to Clear Canvas: Bring thumb and index finger together (pinch gesture) to instantly clear all drawings.

Presentation Mode: Open palm toggles a black screen — perfect for teaching or sharing concepts.

Undo Stroke: Make an “OK” sign to undo your last line.

8 Colors, 5 Brush Sizes: Switch instantly using your keyboard (r, b, g, y, c, m, k, w; 1-5).

Ultra-Smooth Lines: Jitter-reduced, responsive drawing at up to 60 FPS.

Keyboard Shortcuts: Quick access to color, size, save, undo, canvas clear, and mode switch.

Fully Offline: Once dependencies/models are downloaded, works with no internet needed.

Cross-Platform: Works on Windows, Linux, or Mac with any webcam.

Installation 	 	
1. Clone this repository
bash
git clone https://github.com/yourusername/ai-virtual-painter.git
cd ai-virtual-painter

2. Create and Activate Virtual Environment
bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate

3. Install Dependencies
bash
pip install --upgrade pip
pip install opencv-python mediapipe numpy

4. Run the Application
bash
python RemDAPP_pro.py

Controls & Gestures

Keyboard		Effect		Gesture			Effect
r/b/g/c/m/y/w/k		Colors		Index finger only			Draw
1-5			Brush size	Index+middle (hold 0.75s)		Toggle eraser
s			Save drawing	Pinch (thumb+index)		Clear all
p			Presentation	Open palm (hold 1s)		Black screen mode
u			Undo stroke	“OK” sign (hold 1s)		Undo last stroke
x			Clear canvas		
q			Quit		
t			Toggle view

		
Presentation Mode
Open palm gesture or p key instantly toggles between normal canvas + webcam view and a fullscreen blackboard.

Great for online classes, YouTube recordings, or meetings!


How It Works
Captures webcam video with OpenCV

Detects your hand and fingers with MediaPipe Hands

Recognizes gestures (finger count, pinch, palm, OK sign)

Draws lines using smoothed, tracked finger position

Handles gestures for eraser/undo/presentation mode

