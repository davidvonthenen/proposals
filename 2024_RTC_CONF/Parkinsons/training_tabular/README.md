# Training a Parkinsons Model (Parkinsons/training_tabular)

This generates a model which will classify a video to determine to what percentage a person's GAIT is normal vs showing signs of Parkinson's Disease.

## Prerequisites

Have only tested on MacOS, but should also work on most favors of Linux.

Using:

- Python 3.10+

I would highly recommend using something like:

- conda - <https://docs.anaconda.com/free/miniconda/>
- venv - <https://docs.python.org/3/library/venv.html>

## Installation

Install the required packages by running in your (virtual) environment:

```bash
pip install -r requirements.txt
```

## Running the Example

> If you don't have a crazy GPU, this could take some time (My Macbook m2 took about ~1 hour). If you don't want to wait, take a look at using a GPU Cloud Service.

To run the example:

```bash
python main.py
```

## To Clean Up

This will delete EVERYTHING that was downloaded or anything (like the model itself) generated by the script.

```bash
./clean.sh
```