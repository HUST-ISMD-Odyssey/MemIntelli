"""Speech Commands v0.02 inference using the released two-layer GRU."""
from pathlib import Path

import torch
from torch.utils.data import Dataset

from _common import engine, loader, parser, report
from memintelli import convert_model
from memintelli.checkpoints import checkpoint_path
from memintelli.speech import AudioGRU, SpeechFeatures


class SpeechTest(Dataset):
    def __init__(self, root, checkpoint, download=False):
        root = Path(root)
        if not (root / "testing_list.txt").is_file():
            candidate = root / "SpeechCommands/speech_commands_v0.02"
            if not (candidate / "testing_list.txt").is_file() and download:
                import torchaudio
                root.mkdir(parents=True, exist_ok=True)
                torchaudio.datasets.SPEECHCOMMANDS(str(root), download=True, subset="testing")
            root = candidate
        self.root = root
        self.ids = (root / "testing_list.txt").read_text().splitlines()
        self.classes = checkpoint["classes"]
        self.features = SpeechFeatures(checkpoint)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        name = self.ids[index]
        return self.features(self.root / name), self.classes.index(name.split("/")[0])


def main():
    p = parser(__doc__)
    p.add_argument("--wav", type=Path, help="Classify a single mono 16 kHz WAV file")
    args = p.parse_args()
    asset = torch.load(args.checkpoint or checkpoint_path("gru_speech_commands.pt"),
                       map_location="cpu", weights_only=True)
    model = AudioGRU()
    model.load_state_dict(asset["state_dict"])
    model.to(args.device).eval()
    simulator = engine(args) if not args.digital else None
    if simulator is not None:
        model = convert_model(model, simulator)
    with torch.no_grad():
        if args.wav:
            x = SpeechFeatures(asset)(args.wav)[None].to(args.device)
            logits = model(x).float().cpu()[0]
            report(args, {"file": args.wav.name, "prediction": asset["classes"][int(logits.argmax())],
                          "logits": logits.tolist()}, simulator)
            return
        correct, count = 0, 0
        for x, y in loader(SpeechTest(args.data_root, asset, args.download), args):
            logits = model(x.to(args.device))
            if not torch.isfinite(logits).all():
                raise RuntimeError("Nonfinite GRU output")
            correct += int((logits.argmax(1).cpu() == y).sum())
            count += len(y)
    report(args, {"model": "2-layer GRU", "samples": count, "accuracy": correct/count,
                  "limited_evaluation": bool(args.limit)}, simulator)


if __name__ == "__main__":
    main()
