import argparse
import json
import random

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def extract_region_image(image, bounding_box):
    fromX, toX, fromY, toY = (
        bounding_box["fromX"],
        bounding_box["toX"],
        bounding_box["fromY"],
        bounding_box["toY"],
    )
    return image.crop((fromX, fromY, toX, toY))


def process_region(score, region, image_path, partition, new_dataset_path):
    bounding_box = region["bounding_box"]
    region_id = region["id"]
    kern_content = region["**kern"]
    region_name = f"{score.stem}_{region_id}.jpg"

    try:
        with Image.open(image_path) as image:
            cropped = extract_region_image(image, bounding_box)
            cropped.save(new_dataset_path / region_name)

        kern_path = new_dataset_path / f"{score.stem}_{region_id}.kern"
        with open(kern_path, "w") as f:
            f.write(kern_content)

        return kern_path
    except Exception as e:
        print(f"Failed to process region {region_id} in {score.name}: {e}")
        return None


def get_regions(scores, partition, new_dataset_path, max_workers=8):
    partition_files = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for score in scores:
            with open(score) as f:
                score_content = json.load(f)

            image_path = score.with_suffix(".jpg")
            regions = score_content["systems"]

            for region in regions:
                futures.append(
                    executor.submit(
                        process_region,
                        score,
                        region,
                        image_path,
                        partition,
                        new_dataset_path,
                    )
                )

        for future in tqdm(
            as_completed(futures), total=len(futures), desc=f"Processing {partition}"
        ):
            result = future.result()
            if result:
                partition_files.append(result)

    split_path = new_dataset_path / "splits"
    split_path.mkdir(exist_ok=True)

    with open(split_path / f"{partition}_0.txt", "w") as f:
        for file in partition_files:
            f.write(
                f"{str(file).replace('../', '')} {str(file.with_suffix('.jpg')).replace('../', '')}\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates the splits for the dataset and the regions"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the folder containing the clean dataset",
    )
    parser.add_argument(
        "--new_dataset_path",
        type=str,
        default="./data/prueba_dataset_regions/",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    total_scores = list(dataset_path.glob("*.json"))
    print("Number of scores in the dataset: ", len(total_scores))

    filtered_scores = [
        score
        for score in total_scores
        if "version_1." in score.name or "version_" not in score.name
    ]
    test_n_samples = int(len(filtered_scores) * 0.2)
    val_n_samples = int(len(filtered_scores) * 0.1)

    print("Number of unique scores: ", len(filtered_scores))

    random.seed(42)
    filtered_scores = random.sample(filtered_scores, len(filtered_scores))

    test_samples = []
    val_samples = []

    copy_filtered_scores = filtered_scores.copy()

    for score in copy_filtered_scores:  # make a copy to iterate over
        if len(test_samples) < test_n_samples:
            test_samples.append(score)
            filtered_scores.remove(score)
            continue
        if len(val_samples) < val_n_samples:
            val_samples.append(score)
            filtered_scores.remove(score)
            continue

    train_samples = [
        score
        for score in total_scores
        if score not in test_samples and score not in val_samples
    ]
    print(
        f"Number of test, val and train samples: {len(test_samples)}, {len(val_samples)}, {len(train_samples)}"
    )

    assert len(test_samples) + len(val_samples) + len(train_samples) == len(
        total_scores
    )

    new_dataset_path = Path(args.new_dataset_path)
    new_dataset_path.mkdir(exist_ok=True)

    # Process each split using multithreading
    get_regions(test_samples, "test", new_dataset_path)
    get_regions(val_samples, "val", new_dataset_path)
    get_regions(train_samples, "train", new_dataset_path)
