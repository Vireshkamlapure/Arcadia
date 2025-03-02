from ultralytics import YOLO
import cv2
import requests
import os
import subprocess

# Load the YOLO model
model = YOLO("yolov11m.pt")  # Replace with your trained YOLO model

# Open webcam (0 for default camera, change if using external camera)
capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

object_detected = False

# Function to get electronic component information
def get_component_info(component):
    try:
        api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{component.replace(' ', '_')}"
        response = requests.get(api_url)
        
        if response.status_code == 200:
            data = response.json()
            summary = data.get("extract", "No description available.")
            summary_text = summary[:300] + "..."  # Limiting to 50 words approx.
            print(f"{component}: {summary_text}")
            
            # Save information to a text file
            with open("detected_component_info.txt", "w") as file:
                file.write(f"{component}: {summary_text}\n")
                
            # Speak the information using macOS built-in TTS
            #os.system(f'say "{summary_text}"')
            subprocess.run(["say", summary_text])
        else:
            print(f"No information found for {component}.")
    except Exception as e:
        print(f"Error fetching data: {e}")

while True:
    ret, frame = capture.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Perform object detection
    results = model.predict(source=frame, conf=0.5, verbose=False)
    
    # Loop through detected objects and draw bounding boxes with labels
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)  # Convert bounding box to integers
            label = model.names[int(cls)]  # Get class label

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  

            # Put label text
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Get component info
            get_component_info(label)
            
            object_detected = True
            break  # Stop once an object is detected
    
    # Display the live feed with detections
    cv2.imshow("Live Object Detection", frame)
    
    # Wait for a short period to allow display update
    cv2.waitKey(1)
    
    if object_detected:
        cv2.waitKey(3000)  # Keep window open for 3 seconds after detection
        break  # Close the window once an object is detected

# Release resources
capture.release()
cv2.destroyAllWindows()
