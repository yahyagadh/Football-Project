from ultralytics import YOLO
import torch

def main():
    # ----------------------------
    # 1. Check GPU
    # ----------------------------
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA not available. Check your NVIDIA setup.")
    
    print("✅ GPU detected:", torch.cuda.get_device_name(0))

    # ----------------------------
    # 2. Load YOLOv8 model
    # ----------------------------
    model = YOLO("yolov8m.pt")   # you can later switch to yolov8x.pt

    # ----------------------------
    # 3. Train
    # ----------------------------
    model.train(
        data=r"C:/Users/user/Desktop/Football Project/football_players_detection/data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,            # safe for RTX 3050 (6GB)
        device=0,            # GPU
        workers=8,           # multiprocessing SAFE now
        name="football_yolo",
        project="runs/detect",
        pretrained=True
    )

if __name__ == "__main__":
    main()
