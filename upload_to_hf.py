"""Download model from W&B and upload to HuggingFace Hub.

Usage examples:
    # From W&B artifact (downloads checkpoint automatically):
    python upload_to_hf.py --yolo-weights /path/to/yolo.pt

    # From a local Lightning checkpoint:
    python upload_to_hf.py --ckpt artifacts/model-t905a6v6:v0/model.ckpt --yolo-weights /path/to/yolo.pt
"""

import argparse
from pathlib import Path

import torch
from huggingface_hub import HfApi

from jazzmus.model.smt.configuration_smt import SMTConfig
from jazzmus.model.smt.modeling_smt import SMTModelForCausalLM

HF_REPO = "JuanCarlosMartinezSevilla/jazzmus-model"
WANDB_ARTIFACT = "university-alicante/jazzmus/model-t905a6v6:v0"


def main():
    parser = argparse.ArgumentParser(
        description="Upload trained model to HuggingFace Hub"
    )
    parser.add_argument(
        "--wandb-artifact",
        type=str,
        default=WANDB_ARTIFACT,
        help="W&B artifact full name (entity/project/artifact:version)",
    )
    parser.add_argument("--hf-repo", type=str, default=HF_REPO)
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default=None,
        help="Path to YOLO .pt staff-detector weights to include in the repo",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Use a local .ckpt file instead of downloading from W&B",
    )
    args = parser.parse_args()

    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        import wandb

        run = wandb.init()
        artifact = run.use_artifact(args.wandb_artifact, type="model")
        artifact_dir = artifact.download()
        ckpt_path = str(Path(artifact_dir) / "model.ckpt")
        wandb.finish()

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]

    sd = {
        k.replace("model.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }

    embed_categories = sd["decoder.embedding.weight"].shape[0]
    out_categories = sd["decoder.out_layer.weight"].shape[0]

    config = SMTConfig(
        maxh=hp["maxh"],
        maxw=hp["maxw"],
        maxlen=hp["maxlen"],
        out_categories=embed_categories,
        padding_token=hp["padding_token"],
        in_channels=hp["in_channels"],
        w2i=hp["w2i"],
        i2w=hp["i2w"],
        d_model=hp["d_model"],
        dim_ff=hp["dim_ff"],
        num_dec_layers=hp["num_dec_layers"],
    )

    model = SMTModelForCausalLM(config)

    if embed_categories != out_categories:
        from torch.nn import Conv1d
        model.decoder.out_layer = Conv1d(hp["d_model"], out_categories, kernel_size=1)

    model.load_state_dict(sd, strict=True)
    config.out_categories = out_categories

    print(f"Vocabulary size : {len(config.w2i)}")
    print(f"Max sequence len: {config.maxlen}")
    print(f"Config maxh     : {config.maxh}")
    print(f"Config maxw     : {config.maxw}")

    model.push_to_hub(args.hf_repo, commit_message="Upload SMT jazzmus model")
    print(f"SMT model pushed → https://huggingface.co/{args.hf_repo}")

    if args.yolo_weights:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=args.yolo_weights,
            path_in_repo="yolo_staff_detector.pt",
            repo_id=args.hf_repo,
            commit_message="Upload YOLO staff detector weights",
        )
        print(f"YOLO weights uploaded → https://huggingface.co/{args.hf_repo}")


if __name__ == "__main__":
    main()
