from ultralytics import YOLO

model = YOLO("EFFGRP/yolov11s-warehouse-pallets-640")
results = model.predict("Image1.jpg", conf=0.25, save=True)

for result in results:
    for box in result.boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"Pallet detected: conf={confidence:.2f}, bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")