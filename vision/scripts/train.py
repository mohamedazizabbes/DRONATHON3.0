import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=float, default=-1)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--freeze", type=int, default=0)
    ap.add_argument("--cache", default="disk")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    model = YOLO(args.model)
    batch = int(args.batch) if args.batch >= 1 or args.batch == -1 else args.batch
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        patience=args.patience,
        workers=args.workers,
        freeze=args.freeze or None,
        cache=args.cache if args.cache != "none" else False,
        project=str(Path("runs").resolve()),
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        cos_lr=True,
        close_mosaic=10,
        degrees=10.0,
        flipud=0.5,
        fliplr=0.5,
        scale=0.6,
        mixup=0.1,
        copy_paste=0.1,
        hsv_v=0.5,
        amp=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
