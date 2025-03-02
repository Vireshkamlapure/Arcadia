from ultralytics import YOLO
import cv2
import requests
import subprocess


model = YOLO("yolov11m.pt")  

capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

def get_component_info(component):
    try:
        api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{component.replace(' ', '_')}"
        response = requests.get(api_url)
        
        if response.status_code == 200:
            data = response.json()
            summary = data.get("extract", "No description available.")
            summary_text = summary[:300] + "..."  # Limiting to 50 words approx.
            print(f"{component}: {summary_text}")
            
            # Speak the information 
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
    
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls)]
            
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow("Live Object Detection", frame)
            cv2.waitKey(3000)  
            get_component_info(label) 
    
    cv2.imshow("Live Object Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  

capture.release()
cv2.destroyAllWindows()