from ultralytics import YOLO

if __name__ == '__main__':


    # Load a pretrained YOLOv8n model
    model = YOLO('runs/detect/mb_v8/weights/best.pt')
    # Run inference on 'bus.jpg' with arguments
    model.predict(r'D:\MBdataset\images\val', save=True, imgsz=640, conf=0.25)