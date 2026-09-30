"""YOLOv3 or YOLOv5s VOC2007 inference; reports VOC2007 mAP@0.5."""
from pathlib import Path

import numpy as np
import torch

from _common import engine, parser, report
from _yolo import annotation, load_model, voc_ap50, voc_root
from memintelli import convert_model
from memintelli.checkpoints import checkpoint_path


def main():
    p = parser(__doc__)
    p.add_argument("--model", choices=("yolov3", "yolov5s"), default="yolov3")
    p.add_argument("--yolo-source", type=Path, help="Optional local checkout of the pinned upstream implementation")
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--confidence", type=float, default=.001)
    p.add_argument("--nms-iou", type=float, default=.6)
    args = p.parse_args()
    if args.image_size < 32 or args.image_size % 32:
        raise ValueError("Image size must be a positive multiple of 32")
    asset = torch.load(args.checkpoint or checkpoint_path(f"{args.model}_voc.pt"),
                       map_location="cpu", weights_only=True)
    if asset["architecture"] != args.model:
        raise ValueError("Checkpoint architecture differs from --model")
    model = load_model(asset, args.yolo_source).to(args.device).eval()
    import cv2
    from yolo.data.augmentations import letterbox
    from yolo.utils.general import non_max_suppression
    cv2.setNumThreads(1)
    simulator = engine(args) if not args.digital else None
    if simulator is not None:
        model = convert_model(model, simulator)
    root = voc_root(args.data_root)
    ids = (root / "ImageSets/Main/test.txt").read_text().split()
    if args.limit:
        ids = ids[:args.limit]
    predictions, annotations = {}, {}
    with torch.no_grad():
        for sample in ids:
            original = cv2.imread(str(root / f"JPEGImages/{sample}.jpg"))
            if original is None:
                raise ValueError(f"Cannot read VOC image {sample}")
            resized, ratio, padding = letterbox(original, (args.image_size, args.image_size), auto=False, stride=32)
            x = torch.from_numpy(np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1)))[None]
            output = model(x.to(args.device).float()/255)
            output = output[0] if isinstance(output, (list, tuple)) else output
            if not torch.isfinite(output).all():
                raise RuntimeError("Nonfinite detector output")
            detections = non_max_suppression(output.float(), args.confidence, args.nms_iou,
                                             multi_label=True, max_det=300)[0].cpu().numpy()
            if len(detections):
                detections[:, [0, 2]] = (detections[:, [0, 2]]-padding[0])/ratio[0]
                detections[:, [1, 3]] = (detections[:, [1, 3]]-padding[1])/ratio[1]
                detections[:, [0, 2]] = detections[:, [0, 2]].clip(0, original.shape[1])
                detections[:, [1, 3]] = detections[:, [1, 3]].clip(0, original.shape[0])
                # VOC annotation coordinates are one-based.
                detections[:, :4] += 1
            predictions[sample] = detections
            annotations[sample] = annotation(root / f"Annotations/{sample}.xml")
            print(f"Evaluated {len(predictions)}/{len(ids)} images", flush=True)
    mean, per_class = voc_ap50(predictions, annotations)
    report(args, {"model": args.model, "images": len(ids), "mAP50": mean,
                  "AP50_by_class": per_class, "metric": "VOC2007 11-point AP, IoU=0.5",
                  "limited_evaluation": bool(args.limit)}, simulator)


if __name__ == "__main__":
    main()
