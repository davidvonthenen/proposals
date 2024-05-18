# run PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py

from fastcore.all import *
from fastai.data.all import *
from fastai.vision.all import *


from PIL import Image
from duckduckgo_search import DDGS
from fastdownload import download_url

from fastai.vision.utils import download_images, resize_images, verify_images
from pathlib import Path

import os
from time import sleep


def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    with DDGS() as ddgs:
        # generator which yields dicts with:
        # {'title','image','thumbnail','url','height','width','source'}
        search_results = ddgs.images(keywords=term)
        # grap number of max_images urls
        image_urls = [result.get("image") for result in search_results[:max_images]]
        # convert to L (functionally extended list class from fastai)
        return L(image_urls)


def resize_and_pad_images(path, target_width, target_height):
    dest_path = path / "resized"
    dest_path.mkdir(exist_ok=True)

    image_files = get_image_files(path)
    for image_file in image_files:
        with Image.open(image_file) as img:
            # Calculate the new size maintaining aspect ratio
            img.thumbnail((target_width, target_height), Image.ANTIALIAS)
            new_img = Image.new("RGB", (target_width, target_height), (255, 255, 255))
            new_img.paste(
                img,
                ((target_width - img.width) // 2, (target_height - img.height) // 2),
            )

            new_img.save(dest_path / image_file.name)


dog_types = "boxer", "german shepherd", "golden retriever"
path = Path("dogs")

if not path.exists():
    path.mkdir()
    for o in dog_types:
        dest = path / o
        dest.mkdir(exist_ok=True)
        download_images(dest, urls=search_images(f"{o} dog"))
        sleep(10)  # Pause between searches to avoid over-loading server

    failed = verify_images(get_image_files(path))
    failed.map(Path.unlink)
    print(f"failed: {len(failed)}")

    # debug: list images
    # fns = get_image_files(path)
    # print(fns)

# determine smallest image
optimal_width = 384
optimal_height = 384

# generate model
dogsPath = Path("dogs_classifier_model.pkl")
if not dogsPath.exists():
    # we want to place the images in a Category (therefore using CategoryBlock)
    dogs = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(
            valid_pct=0.2, seed=42
        ),  # Set aside 20% of the data for training
        get_y=parent_label,
        item_tfms=Resize(
            (optimal_height, optimal_width)
        ),  # Resize to optimal dimensions
        batch_tfms=aug_transforms(),
    )
    dls = dogs.dataloaders(path)

    # dls (dataset) and resnet18 is your model (timm.fast.ai) from pytorch
    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(4)

    learn.save("dogs_classifier_model")
    learn.export(
        "dogs_classifier_model.pkl"
    )  # Save the entire learner object for inference

    # ClassificationInterpretation use to obtain images that are too far off, so you can remove them
    # interp = ClassificationInterpretation.from_learner(learn)
    # interp.plot_confusion_matrix()
    # print("\n\n")
    # interp.plot_top_losses(5, nrows=1)
    # print("\n\n")

# re-load the model
learn = load_learner(fname="dogs_classifier_model.pkl")

# get test image
boxerExist = os.path.exists("boxer.jpg")
if not boxerExist:
    urls = search_images("boxer dog face", max_images=1)
    print(urls[0])
    download_url(urls[0], "boxer.jpg", show_progress=False)
germanExist = os.path.exists("german-shepherd.jpg")
if not germanExist:
    urls = search_images("german shepherd face", max_images=1)
    print(urls[0])
    download_url(urls[0], "german-shepherd.jpg", show_progress=False)
goldenExist = os.path.exists("golden-retriever.jpg")
if not goldenExist:
    urls = search_images("golden retriever face", max_images=1)
    print(urls[0])
    download_url(urls[0], "golden-retriever.jpg", show_progress=False)

# get prediction
pred, pred_idx, probs = learn.predict(PILImage.create("boxer.jpg"))
print("\n")
print(f"Testing: boxer.jpg")
print(f"Prediction: {pred}")
print(f"Probability: {probs[pred_idx]:.4f}")
print("\n")

# get prediction
pred, pred_idx, probs = learn.predict(PILImage.create("german-shepherd.jpg"))
print("\n")
print(f"Testing: german-shepherd.jpg")
print(f"Prediction: {pred}")
print(f"Probability: {probs[pred_idx]:.4f}")
print("\n")

# get prediction
pred, pred_idx, probs = learn.predict(PILImage.create("golden-retriever.jpg"))
print("\n")
print(f"Testing: golden-retriever.jpg")
print(f"Prediction: {pred}")
print(f"Probability: {probs[pred_idx]:.4f}")
print("\n")
