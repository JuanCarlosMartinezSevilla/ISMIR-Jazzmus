"""Full-page music score transcription: YOLO staff detection + SMT model inference.

Usage examples:
    # Transcribe a page and print to stdout:
    python predict.py page.png

    # Save to a .krn file:
    python predict.py page.png -o output.krn

    # Both print and save:
    python predict.py page.png -o output.krn -p

    # Process multiple pages into an output directory:
    python predict.py page1.png page2.png -o results/

    # Use local YOLO weights instead of downloading from HF:
    python predict.py page.png --yolo-weights path/to/yolo.pt
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image

from jazzmus.dataset.data_preprocessing import convert_img_to_tensor
from jazzmus.dataset.tokenizer import untokenize
from jazzmus.model.smt.configuration_smt import SMTConfig
from jazzmus.model.smt.modeling_smt import SMTModelForCausalLM

HF_REPO = "JuanCarlosMartinezSevilla/jazzmus-model"
FIXED_IMG_HEIGHT = 128
MAX_IMG_WIDTH = 1000


def detect_staves(image_path: str | Path, yolo_model) -> list[np.ndarray]:
    results = yolo_model(str(image_path))
    image = Image.open(image_path)

    staff_boxes = []
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            if result.names[int(cls)].lower() == "staff":
                x1, y1, x2, y2 = map(int, box.tolist())
                staff_boxes.append((y1, x1, y2, x2))

    staff_boxes.sort(key=lambda b: b[0])

    crops = []
    for y1, x1, y2, x2 in staff_boxes:
        crop = image.crop((x1, y1, x2, y2))
        crops.append(np.array(crop.convert("L")))

    logger.info(f"Detected {len(crops)} staff regions in {Path(image_path).name}")
    return crops


def preprocess_staff(img: np.ndarray) -> torch.Tensor:
    width = int(np.ceil(img.shape[1] * FIXED_IMG_HEIGHT / img.shape[0]))
    width = min(width, MAX_IMG_WIDTH)
    img = cv2.resize(img, (width, FIXED_IMG_HEIGHT))
    return convert_img_to_tensor(img)


@torch.no_grad()
def transcribe_staff(model: SMTModelForCausalLM, image_tensor: torch.Tensor) -> str:
    predicted_sequence, _ = model.predict(
        input=image_tensor, convert_to_str=True
    )
    return untokenize(predicted_sequence)


def _sync_pe_device(model: SMTModelForCausalLM, device: torch.device):
    model.positional_2D.pe = model.positional_2D.pe.to(device)
    model.decoder.positional_1D.pe = model.decoder.positional_1D.pe.to(device)


def transcribe_page(
    image_path: str | Path,
    smt_model: SMTModelForCausalLM,
    yolo_model,
    device: torch.device,
) -> str:
    staff_images = detect_staves(image_path, yolo_model)
    if not staff_images:
        logger.warning("No staff regions detected")
        return ""

    transcriptions = []
    for i, staff_img in enumerate(staff_images):
        tensor = preprocess_staff(staff_img).to(device)
        kern = transcribe_staff(smt_model, tensor)
        transcriptions.append(kern)
        logger.info(f"Staff {i + 1}/{len(staff_images)} transcribed")

    return "!!linebreak\n".join(transcriptions)


def _resolve_device(device_str: str) -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe a full-page music score to **kern notation."
    )
    parser.add_argument("input", type=str, nargs="+", help="Path(s) to input image(s)")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output path: file (.krn) for single input, directory for multiple",
    )
    parser.add_argument(
        "-p",
        "--print-output",
        action="store_true",
        default=False,
        help="Print transcription to stdout",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto | cpu | cuda | mps)",
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default=None,
        help="Local path to YOLO .pt weights (auto-downloaded from HF if omitted)",
    )
    parser.add_argument(
        "--model-repo",
        type=str,
        default=HF_REPO,
        help="HuggingFace repo ID for the SMT model",
    )
    args = parser.parse_args()

    if not args.output and not args.print_output:
        args.print_output = True

    device = _resolve_device(args.device)
    logger.info(f"Using device: {device}")

    # ---- YOLO staff detector ------------------------------------------------
    from ultralytics import YOLO

    if args.yolo_weights:
        yolo_path = args.yolo_weights
    else:
        from huggingface_hub import hf_hub_download

        yolo_path = hf_hub_download(
            repo_id=args.model_repo, filename="yolo_staff_detector.pt"
        )
    yolo = YOLO(yolo_path)
    logger.info("YOLO staff detector loaded")

    # ---- SMT transcription model --------------------------------------------
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from torch.nn import Conv1d

    config = SMTConfig.from_pretrained(args.model_repo)
    weights_file = hf_hub_download(
        repo_id=args.model_repo, filename="model.safetensors"
    )
    sd = load_file(weights_file)
    embed_size = sd["decoder.embedding.weight"].shape[0]
    out_size = sd["decoder.out_layer.weight"].shape[0]
    orig_out = config.out_categories
    config.out_categories = embed_size
    model = SMTModelForCausalLM(config)
    if embed_size != out_size:
        model.decoder.out_layer = Conv1d(config.d_model, out_size, kernel_size=1)
    model.load_state_dict(sd, strict=True)
    config.out_categories = orig_out
    model.to(device)
    _sync_pe_device(model, device)
    model.eval()
    logger.info("SMT model loaded")

    # ---- Process inputs -----------------------------------------------------
    input_paths = [Path(p) for p in args.input]
    for ip in input_paths:
        if not ip.exists():
            raise FileNotFoundError(f"Input image not found: {ip}")

    results: dict[Path, str] = {}
    for ip in input_paths:
        logger.info(f"Processing {ip.name}")
        results[ip] = transcribe_page(ip, model, yolo, device)

    # ---- Output -------------------------------------------------------------
    for ip, kern in results.items():
        if args.print_output:
            if len(results) > 1:
                print(f"=== {ip.name} ===")
            print(kern)
            if len(results) > 1:
                print()

        if args.output:
            out = Path(args.output)
            if len(input_paths) == 1 and out.suffix:
                out_file = out
            else:
                out.mkdir(parents=True, exist_ok=True)
                out_file = out / ip.with_suffix(".krn").name
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(kern)
            logger.info(f"Saved → {out_file}")


if __name__ == "__main__":
    main()
