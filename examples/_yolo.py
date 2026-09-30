"""Pinned optional YOLO implementation and VOC2007 AP50 evaluation."""
import os
from pathlib import Path
import sys
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import torch

COMMIT = "09e4ad6ca9ccc0b3cd25853057904b2ea71a8673"
NAMES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
         "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
         "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def source_path(local=None):
    if local is not None:
        root = Path(local).resolve()
        if not (root / "yolo/model/yolov5.py").is_file():
            raise ValueError("--yolo-source must point to the zjykzj/YOLOv5 checkout")
    else:
        cache = Path(torch.hub.get_dir()) / "memintelli-yolo"
        root = cache / f"YOLOv5-{COMMIT}"
        if not (root / "yolo/model/yolov5.py").exists():
            cache.mkdir(parents=True, exist_ok=True)
            archive = cache / f"{COMMIT}.zip"
            torch.hub.download_url_to_file(
                f"https://github.com/zjykzj/YOLOv5/archive/{COMMIT}.zip", str(archive))
            with zipfile.ZipFile(archive) as src:
                for info in src.infolist():
                    target = (cache / info.filename).resolve()
                    if not target.is_relative_to(cache.resolve()) or (info.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError("Unsafe source archive member")
                src.extractall(cache)
    os.environ["YOLOv5_AUTOINSTALL"] = "false"
    sys.path.insert(0, str(root))
    return root


def load_model(asset, source=None):
    source_path(source)
    import setuptools  # Provides distutils for upstream thop on Python >= 3.12.
    from yolo.model.yolov5 import DetectionModel
    model = DetectionModel(asset["config"], ch=3, nc=20)
    model.load_state_dict(asset["state_dict"], strict=True)
    model.names = asset["names"]
    model.requires_grad_(False)
    with torch.no_grad():
        return model.float().fuse().eval()


def voc_root(root):
    for candidate in (root, root / "VOC2007", root / "VOCdevkit/VOC2007"):
        if (candidate / "ImageSets/Main/test.txt").is_file():
            return candidate
    raise FileNotFoundError("Expected extracted VOC2007 test data with ImageSets/Main/test.txt")


def annotation(path):
    tree = ET.parse(path).getroot()
    result = []
    for obj in tree.findall("object"):
        box = [float(obj.findtext("bndbox/"+k)) for k in ("xmin", "ymin", "xmax", "ymax")]
        result.append({"class": NAMES.index(obj.findtext("name")), "box": box,
                       "difficult": obj.findtext("difficult", "0") == "1"})
    return result


def voc_ap50(predictions, annotations):
    """VOC2007 11-point AP at IoU .5; difficult objects do not count as negatives."""
    per_class = {}
    for cls, name in enumerate(NAMES):
        gt = {key: [obj for obj in values if obj["class"] == cls] for key, values in annotations.items()}
        positives = sum(not obj["difficult"] for values in gt.values() for obj in values)
        if not positives:
            continue
        used = {key: set() for key in gt}
        detections = sorted(((float(row[4]), key, row[:4]) for key, rows in predictions.items()
                             for row in rows if int(row[5]) == cls), reverse=True, key=lambda x: x[0])
        tp, fp = [], []
        for score, key, box in detections:
            objects = gt[key]
            if not objects:
                tp.append(0); fp.append(1)
                continue
            boxes = np.asarray([obj["box"] for obj in objects])
            low, high = np.maximum(box[:2], boxes[:, :2]), np.minimum(box[2:], boxes[:, 2:])
            intersection = np.maximum(high-low+1, 0).prod(1)
            union = np.maximum(box[2:]-box[:2]+1, 0).prod() + np.maximum(boxes[:, 2:]-boxes[:, :2]+1, 0).prod(1) - intersection
            iou = intersection / np.maximum(union, 1e-12)
            index = int(iou.argmax())
            if iou[index] >= .5:
                if objects[index]["difficult"]:
                    continue
                match = index not in used[key]
                used[key].add(index)
                tp.append(int(match)); fp.append(int(not match))
            else:
                tp.append(0); fp.append(1)
        tp, fp = np.cumsum(tp), np.cumsum(fp)
        recall, precision = tp/positives, tp/np.maximum(tp+fp, 1)
        per_class[name] = sum(float(precision[recall >= t].max()) if (recall >= t).any() else 0
                              for t in np.linspace(0, 1, 11))/11
    return float(np.mean(list(per_class.values()))) if per_class else 0.0, per_class
