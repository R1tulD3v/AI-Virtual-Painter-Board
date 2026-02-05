import cv2
import mediapipe as mp
import numpy as np
import math
import time
from collections import deque

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Get actual resolution
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Canvas
imgCanvas = np.zeros((height, width, 3), np.uint8)

# Drawing parameters
drawColor = (0, 0, 255)  # Red
previousColor = drawColor
thickness = 10
tipIds = [4, 8, 12, 16, 20]

# Smoothing buffer
smooth_buffer = deque(maxlen=3)

# Stroke history for undo (store last 10 strokes)
stroke_history = []
current_stroke = []

# Previous positions
xp, yp = 0, 0

# Colors
key_color_mapping = {
    'r': (0, 0, 255), 'b': (255, 0, 0), 'g': (0, 255, 0),
    'c': (255, 255, 0), 'm': (255, 0, 255), 'y': (0, 255, 255),
    'k': (0, 0, 0), 'w': (255, 255, 255)  # Added white
}

# Thickness mapping
thickness_mapping = {'1': 5, '2': 10, '3': 15, '4': 20, '5': 25}

# Gesture state tracking
eraser_timer_start = None
last_pinch_time = 0
pinch_cooldown = 2.0
last_gesture_state = None
open_palm_start_time = None

# View modes
presentation_mode = False  # Black screen with drawing only
show_webcam = True  # Toggle webcam visibility

# Performance tracking
fps_list = deque(maxlen=30)
last_time = time.time()

print("=" * 75)
print("AI VIRTUAL PAINTER - PROFESSIONAL EDITION".center(75))
print("=" * 75)
print("\n🎨 DRAWING GESTURES:")
print("  ✏️  INDEX FINGER ONLY = Draw")
print("  🧹 INDEX + MIDDLE (hold 0.75s) = Toggle Eraser")
print("  🗑️  PINCH (thumb + index close) = Clear Canvas")
print("  ✋ OPEN PALM (all 5 fingers, hold 1s) = Toggle Presentation Mode")
print("  👌 OK SIGN (thumb + index circle, hold 1s) = Undo Last Stroke")
print("\n⌨️  KEYBOARD SHORTCUTS:")
print("  Colors: r/b/g/c/m/y/k/w | Size: 1-5 | View: t | Presentation: p")
print("  Save: s | Undo: u | Clear: x | Quit: q")
print("\n📺 VIEW MODES:")
print("  • Normal: Webcam + Drawing")
print("  • Canvas Only: Press 't' (no webcam)")
print("  • Presentation: Press 'p' (fullscreen black + drawing)")
print("\n💡 PRO TIPS:")
print("  • Presentation mode = Perfect for teaching/presenting")
print("  • Undo = Remove mistakes without clearing everything")
print("  • Open palm gesture = Quick blackout for focus")
print("=" * 75 + "\n")

def smooth_position(x, y):
    """Apply smoothing to reduce jitter"""
    smooth_buffer.append((x, y))
    if len(smooth_buffer) >= 2:
        avg_x = sum(p[0] for p in smooth_buffer) // len(smooth_buffer)
        avg_y = sum(p[1] for p in smooth_buffer) // len(smooth_buffer)
        return avg_x, avg_y
    return x, y

def count_fingers(points):
    """Count raised fingers"""
    fingers = []
    
    # Thumb
    if points[4][0] < points[3][0]:
        fingers.append(1)
    else:
        fingers.append(0)
    
    # Other fingers
    for i in range(1, 5):
        if points[tipIds[i]][1] < points[tipIds[i] - 2][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    
    return fingers

def save_stroke():
    """Save current stroke to history"""
    global current_stroke, stroke_history
    if len(current_stroke) > 0:
        stroke_history.append(current_stroke.copy())
        if len(stroke_history) > 10:  # Keep only last 10 strokes
            stroke_history.pop(0)
        current_stroke = []

def undo_stroke():
    """Undo last stroke"""
    global imgCanvas, stroke_history
    if len(stroke_history) > 0:
        stroke_history.pop()
        # Redraw canvas from history
        imgCanvas = np.zeros((height, width, 3), np.uint8)
        for stroke in stroke_history:
            for i in range(1, len(stroke)):
                cv2.line(imgCanvas, stroke[i-1][0], stroke[i][0], 
                        stroke[i-1][1], stroke[i-1][2])
        print("↩️  Undo: Removed last stroke")
    else:
        print("⚠️  Nothing to undo")

# Main loop
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7,
    model_complexity=1
) as hands:
    
    while cap.isOpened():
        current_time = time.time()
        fps = 1 / (current_time - last_time) if (current_time - last_time) > 0 else 0
        fps_list.append(fps)
        last_time = current_time
        
        success, image = cap.read()
        if not success:
            print("⚠️  Camera disconnected")
            break
        
        # Process image
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        current_gesture = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract points
                points = [(int(lm.x * width), int(lm.y * height)) 
                         for lm in hand_landmarks.landmark]
                
                # Count fingers
                fingers = count_fingers(points)
                fingers_up = sum(fingers)
                
                # Get key points
                thumb_tip = points[4]
                index_tip = points[8]
                middle_tip = points[12]
                
                # Calculate distances
                pinch_distance = math.sqrt(
                    (thumb_tip[0] - index_tip[0])**2 + 
                    (thumb_tip[1] - index_tip[1])**2
                )
                
                # OK sign distance (thumb tip to index middle joint)
                ok_distance = math.sqrt(
                    (thumb_tip[0] - points[6][0])**2 + 
                    (thumb_tip[1] - points[6][1])**2
                )
                
                # GESTURE PRIORITY:
                
                # 1. OPEN PALM (all 5 fingers up) - Toggle Presentation Mode
                if fingers_up == 5:
                    if open_palm_start_time is None:
                        open_palm_start_time = current_time
                    
                    elapsed = current_time - open_palm_start_time
                    
                    if elapsed >= 1.0:
                        if current_gesture != 'palm_toggle':
                            presentation_mode = not presentation_mode
                            mode = "ON" if presentation_mode else "OFF"
                            print(f"📺 Presentation Mode: {mode}")
                            open_palm_start_time = None
                            current_gesture = 'palm_toggle'
                    else:
                        remaining = 1.0 - elapsed
                        cv2.putText(image, f"Presentation in {remaining:.1f}s", 
                                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                   1, (255, 0, 255), 2)
                        current_gesture = 'palm_wait'
                    
                    save_stroke()
                    xp, yp = 0, 0
                    smooth_buffer.clear()
                
                # 2. OK SIGN (thumb + index circle) - Undo
                elif ok_distance < 40 and fingers[1] and not fingers[2] and not fingers[3]:
                    if open_palm_start_time is None:
                        open_palm_start_time = current_time
                    
                    elapsed = current_time - open_palm_start_time
                    
                    if elapsed >= 1.0:
                        if current_gesture != 'ok_undo':
                            undo_stroke()
                            open_palm_start_time = None
                            current_gesture = 'ok_undo'
                    else:
                        remaining = 1.0 - elapsed
                        cv2.putText(image, f"Undo in {remaining:.1f}s", 
                                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                   1, (0, 165, 255), 2)
                        current_gesture = 'ok_wait'
                    
                    save_stroke()
                    xp, yp = 0, 0
                    smooth_buffer.clear()
                
                # 3. PINCH (clear canvas)
                elif pinch_distance < 30 and (current_time - last_pinch_time) > pinch_cooldown:
                    if current_gesture != 'pinch':
                        print("🗑️  Canvas cleared!")
                        imgCanvas = np.zeros((height, width, 3), np.uint8)
                        stroke_history = []
                        current_stroke = []
                        last_pinch_time = current_time
                        current_gesture = 'pinch'
                        xp, yp = 0, 0
                        smooth_buffer.clear()
                
                # 4. TWO FINGERS (eraser toggle)
                elif fingers[1] and fingers[2] and not fingers[0] and not fingers[3] and not fingers[4]:
                    if eraser_timer_start is None:
                        eraser_timer_start = current_time
                    
                    elapsed = current_time - eraser_timer_start
                    
                    if elapsed >= 0.75:
                        if current_gesture != 'eraser_toggle':
                            if drawColor == key_color_mapping['k']:
                                drawColor = previousColor
                                print("✏️  Drawing mode")
                            else:
                                previousColor = drawColor
                                drawColor = key_color_mapping['k']
                                print("🧹 Eraser mode")
                            
                            eraser_timer_start = None
                            current_gesture = 'eraser_toggle'
                    else:
                        remaining = 0.75 - elapsed
                        cv2.putText(image, f"Eraser in {remaining:.1f}s", 
                                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                   1, (0, 255, 255), 2)
                        current_gesture = 'eraser_wait'
                    
                    save_stroke()
                    xp, yp = 0, 0
                    smooth_buffer.clear()
                
                # 5. INDEX FINGER ONLY (draw)
                elif fingers[1] and not any([fingers[0], fingers[2], fingers[3], fingers[4]]):
                    x1, y1 = smooth_position(index_tip[0], index_tip[1])
                    
                    # Visual indicator
                    cv2.circle(image, (x1, y1), thickness + 8, drawColor, 2)
                    cv2.circle(image, (x1, y1), thickness // 2, drawColor, cv2.FILLED)
                    
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1
                    
                    # Draw and save stroke
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)
                    current_stroke.append(((xp, yp), (x1, y1), drawColor, thickness))
                    
                    xp, yp = x1, y1
                    current_gesture = 'drawing'
                    eraser_timer_start = None
                    open_palm_start_time = None
                
                # 6. OTHER (reset)
                else:
                    if current_gesture == 'drawing':
                        save_stroke()
                    xp, yp = 0, 0
                    eraser_timer_start = None
                    open_palm_start_time = None
                    smooth_buffer.clear()
                
                # Draw hand landmarks (only in non-presentation mode)
                
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(80, 80, 80), thickness=1, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(150, 150, 150), thickness=1)
                    )
                
                last_gesture_state = current_gesture
        
        else:
            if current_gesture == 'drawing':
                save_stroke()
            xp, yp = 0, 0
            eraser_timer_start = None
            open_palm_start_time = None
            smooth_buffer.clear()
        
        # Create display based on mode
        if presentation_mode:
            # Presentation mode: Black background with drawing only
            display_img = imgCanvas.copy()
        else:
            # Normal mode: Combine canvas and webcam
            imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
            _, imgInv = cv2.threshold(imgGray, 5, 255, cv2.THRESH_BINARY_INV)
            imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
            img = cv2.bitwise_and(image, imgInv)
            display_img = cv2.bitwise_or(img, imgCanvas)
            
            # UI Overlay (only in normal mode)
            # Color indicator
            cv2.rectangle(display_img, (10, height - 70), (70, height - 10), drawColor, -1)
            cv2.rectangle(display_img, (10, height - 70), (70, height - 10), (255, 255, 255), 2)
            
            # Info panel
            info_bg = np.zeros((100, 280, 3), dtype=np.uint8)
            info_bg[:] = (30, 30, 30)
            cv2.addWeighted(display_img[10:110, 10:290], 0.3, info_bg, 0.7, 0, display_img[10:110, 10:290])
            
            cv2.putText(display_img, f"Size: {thickness}px", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display_img, f"FPS: {int(sum(fps_list)/len(fps_list))}", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(display_img, f"Strokes: {len(stroke_history)}", (20, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)
            
            mode = "ERASER" if drawColor == key_color_mapping['k'] else "DRAW"
            color = (255, 100, 0) if mode == "ERASER" else (100, 255, 100)
            cv2.putText(display_img, mode, (180, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        window_name = 'PRESENTATION MODE' if presentation_mode else 'AI Virtual Painter - Pro Edition'
        cv2.imshow(window_name, display_img)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n👋 Goodbye!")
            break
        
        if key == ord('p'):
            presentation_mode = not presentation_mode
            mode = "ON" if presentation_mode else "OFF"
            print(f"📺 Presentation Mode: {mode}")
        
        if key == ord('u'):
            undo_stroke()
        
        if key == ord('x'):
            print("🗑️  Canvas cleared!")
            imgCanvas = np.zeros((height, width, 3), np.uint8)
            stroke_history = []
            current_stroke = []
        
        if chr(key) in key_color_mapping:
            drawColor = key_color_mapping[chr(key)]
            print(f"🎨 Color: {chr(key).upper()}")
        
        if chr(key) in thickness_mapping:
            thickness = thickness_mapping[chr(key)]
            print(f"📏 Size: {thickness}px")
        
        if key == ord('s'):
            filename = f"drawing_{int(time.time())}.png"
            cv2.imwrite(filename, imgCanvas)
            print(f"💾 Saved: {filename}")

cap.release()
cv2.destroyAllWindows()
print("\n✅ Application closed successfully")