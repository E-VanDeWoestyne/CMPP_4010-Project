from ultralytics import YOLOvv11

model = YOLOvv11.from_pretrained("EFFGRP/yolov11n-warehouse-pallets-960")
source = 'http://images.cocodataset.org/val2017/000000039769.jpg'
model.predict(source=source, save=True)