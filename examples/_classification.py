"""Shared image-classification evaluation and hardware-aware training."""
import os

import torch
from torch import nn
from torchvision import datasets, models, transforms

from _common import engine, loader, parser, report
from memintelli import convert_model


def build_model(kind, pretrained=True):
    if kind == "mlp":
        return nn.Sequential(nn.Flatten(), nn.Linear(784, 512), nn.ReLU(),
                             nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 10))
    if kind == "vgg_cifar" or kind == "vgg_cifar100":
        from memintelli.NN_models.vgg_cifar import vgg_cifar_zoo
        return vgg_cifar_zoo("vgg16_bn", num_classes=100 if kind.endswith("100") else 10,
                             pretrained=pretrained, mem_enabled=False)
    if kind == "resnet_cifar":
        from memintelli.NN_models.resnet_cifar import ResNet_CIFAR_zoo
        return ResNet_CIFAR_zoo("resnet18", num_classes=10, pretrained=pretrained, mem_enabled=False)
    if kind == "deit":
        from memintelli.NN_models.DeiT import deit_zoo
        return deit_zoo("deit_tiny_patch16_224", pretrained=pretrained, mem_enabled=False)
    factories = {"vgg": (models.vgg16_bn, models.VGG16_BN_Weights.IMAGENET1K_V1),
                 "resnet": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
                 "mobilenet": (models.mobilenet_v2, models.MobileNet_V2_Weights.IMAGENET1K_V1)}
    factory, weights = factories[kind]
    return factory(weights=weights if pretrained else None)


def dataset(kind, args, training=False):
    if kind == "mlp":
        return datasets.MNIST(args.data_root, train=training, download=args.download,
                               transform=transforms.Compose([transforms.ToTensor(),
                                                             transforms.Normalize((.1307,), (.3081,))]))
    if "cifar" in kind:
        hundred = kind.endswith("100")
        # These are the normalizations paired with the published checkpoints.
        mean = [.507, .4865, .4409] if hundred else [.485, .456, .406]
        std = [.2673, .2564, .2761] if hundred else [.229, .224, .225]
        cls = datasets.CIFAR100 if hundred else datasets.CIFAR10
        return cls(args.data_root, train=training, download=args.download,
                   transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]))
    root = args.data_root / ("train" if training else "val")
    if not root.is_dir():
        root = args.data_root
    return datasets.ImageFolder(root, allow_empty=True, transform=transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ]))


def main(kind, training=False, distributed=False):
    p = parser(f"{kind} classification" + (" hardware-aware training" if training else " inference"))
    p.add_argument("--epochs", type=int, default=1 if training else 3)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--save-checkpoint", type=str)
    args = p.parse_args()
    rank, world = 0, 1
    if distributed:
        import torch.distributed as dist
        world = int(os.environ.get("WORLD_SIZE", "1"))
        if world > 1:
            rank = int(os.environ["RANK"])
            local = int(os.environ["LOCAL_RANK"])
            if args.device.startswith("cuda"):
                torch.cuda.set_device(local)
                args.device = f"cuda:{local}"
            dist.init_process_group("nccl" if args.device.startswith("cuda") and os.name != "nt" else "gloo")
    torch.manual_seed(args.seed)
    simulator = engine(args) if not args.digital else None
    model = build_model(kind, pretrained=not args.checkpoint and kind != "mlp")
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("state_dict", state.get("model", state)))
    model.to(args.device)
    needs_training = training or (kind == "mlp" and not args.checkpoint)
    if training and simulator is not None:
        model = convert_model(model, simulator)
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel
        model = DistributedDataParallel(model, device_ids=[int(os.environ["LOCAL_RANK"])]
                                         if args.device.startswith("cuda") else None)
    if needs_training:
        if args.epochs < 1:
            raise ValueError("Training requires at least one epoch")
        train_data = dataset(kind, args, True)
        if args.limit:
            train_data = torch.utils.data.Subset(train_data, range(min(args.limit, len(train_data))))
        sampler = torch.utils.data.distributed.DistributedSampler(train_data) if world > 1 else None
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                   sampler=sampler, shuffle=sampler is None,
                                                   num_workers=args.workers)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        for epoch in range(args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            for x, y in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(x.to(args.device)), y.to(args.device))
                loss.backward()
                optimizer.step()
            scheduler.step()
            if rank == 0:
                print(f"Epoch {epoch+1}/{args.epochs} complete", flush=True)
    if world > 1:
        model = model.module
    if args.save_checkpoint and rank == 0:
        from pathlib import Path
        target = Path(args.save_checkpoint)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict()}, target)
    if not training and simulator is not None:
        model = convert_model(model, simulator)
    model.eval()
    correct, count = 0, 0
    with torch.no_grad():
        for x, y in loader(dataset(kind, args), args):
            output = model(x.to(args.device))
            if not torch.isfinite(output).all():
                raise RuntimeError("Nonfinite network output")
            correct += int((output.argmax(1).cpu() == y).sum())
            count += len(y)
    if count == 0:
        raise ValueError("Empty evaluation dataset")
    if rank == 0:
        report(args, {"model": kind, "samples": count, "accuracy": correct/count,
                      "limited_evaluation": bool(args.limit)}, simulator)
    if world > 1:
        dist.destroy_process_group()
