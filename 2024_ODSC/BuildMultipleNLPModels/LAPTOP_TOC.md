# Workshop Using Your Laptop

Welcome to landing page for workshop `Building Multiple Natural Language Processing Models to Work In Concert Together` at `2024 Open Data Science Conference West`.

## Prerequisites

### Hardware/Environment Prerequisites

You are required to be on a Linux-based operating system. All materials have been tested on the following operating systems:

- MacOS on Apple Silicon (14.5)
- Ubuntu x64 (24.04)
- RHEL x64 (9.X)
- Rocky x64 (9.X)

Other Linux flavors should work without issues, but we have not explicitly verified those platform versions. You may be on your own if you encounter any problems, but you can always fall back to the [Google Colab Notebooks](https://colab.research.google.com/).

#### Windows Users

If you happen to be a Windows user, I would recommend either:

- **Easy at the Trade-off for Cost:** Use a Cloud Instance
  - AWS EC2
  - GCP
  - Azure
- **More Effort but Could be Free:** Use a Hypervisor (like [VMware Workstation](https://www.vmware.com/products/desktop-hypervisor/workstation-and-fusion)) to run a Linux Virtual Machine

We will not be providing access to either Hypervisor software OR Cloud Instances. If you choose this option, obtaining these resources and understanding the cost associated with using them (i.e., Cloud Instances) is entirely on you.

### Software Components

- Python 3.10+
- (Optional) Virtual Environment: I would highly recommend using something like:
  -	miniconda: https://docs.anaconda.com/free/miniconda/ 
  -	venv: https://docs.python.org/3/library/venv.html 
- Install Portaudio
  - MacOS: `brew install portaudio`
  - Ubuntu: `apt-get install portaudio19-dev`
  - RHEL/Rocky: `dnf install portaudio portaudio-devel`

Once you have a `conda` or `venv` environment, create a virtual workspace called `2024-odsc-west`. Then install the requirements components for Python:

```bash
pip install -r requirements.txt
```

### Free SaaS Platform Accounts Required

For the last third of the lab, we are going to make use of the following SaaS platforms:

- [Deepgram](https://deepgram.com) - This is to help facilitate building an Voice-Activated AI Assistant/Agent.
- [groq](https://console.groq.com/login) - This is to provide an LLM in the examples.

#### Deepgram Account

You need a Deepgram account in order to do the Speech-to-Text and Text-to-Speech. The account is free and you get $200 in credits! [Sign up at Deepgram.com](https://deepgram.com)

Set the API Key to the following environment variable: `DEEPGRAM_API_KEY`.

#### groq Account

You need a groq API Key to use an LLM to provide responses in the Voice Assistant. [Sign up at groq](https://console.groq.com/login).

Set the API Key to the following environment variable: `GROQ_API_KEY`.

## Workshop Materials

Here are the materials for the Google Colab version of this workshop:

- [Access Slides](https://docs.google.com/presentation/d/1KKxXRUpzlyWjjl35dY_qfuRd8jg0pryazy9ueS5xyOo/edit?usp=sharing)
- Download Workshop Materials [Download Link](https://drive.google.com/file/d/1z-TQf4vmbGcmzDaLOCdZL3Pq2CngNf8G/view?usp=drive_link)
  - [Backup Link](https://drive.google.com/file/d/1irmgS6Vve2rrt2d0fu2yR48ujabdaR8d/view?usp=drive_link)

### Setting Up Workshop Materials

#### Steps for Setup:

- Unzip the `ODSC_NLP-Standalone.zip` to some workspace on your latop. 
- (Optional) Activate your virtual environment `conda` or `venv`
- Run: `pip install -r requirements.txt`

#### Hands-On Section

There are three hands-on parts of this workshop below.

##### Part 1: Building the Question Model

The hands on components for this part is located in `question` folder.

- `main.py`: Dynamically downloads and builds the Question Classifier Model
- `service.py`: Will be used in `Part 3`. Ignore this for now.

Instructions:

- Open `main.py` and follow along on the code walkthrough. 
- When instructed to do so, run: `python main.py` to start the model training.
- Cancel the operation by hitting `Cntl + C`. **NOTE:** We will not finish this because it will take 2 hours to complete (20mins on H100).
- Rename the following files:
  - `mv train-v2.0.json.RENAME train-v2.0.json`
  - `mv questions_dataset.csv.RENAME questions_dataset.csv`
  - `mv non_questions_dataset.csv.RENAME non_questions_dataset.csv`
  - `mv question_model.RENAME question_model`
- To test the sentences, run: `python main.py` 

##### Part 2: Building a Named Entity Recognition Model

The hands on components for this part is located in `named-entity-recognition` folder.

- `main.py`: Dynamically downloads and builds the Question Classifier Model
- `service.py`: Will be used in `Part 3`. Ignore this for now.

Instructions:

- Open `main.py` and follow along on the code walkthrough. 
- When instructed to do so, run: `python update-rank.py` to understand what it is to do one form of "data grooming".
- When instructed to do so, run: `python main.py` to start the model training.
- Cancel the operation by hitting `Cntl + C`. **NOTE:** We will not finish this because it will take 12 hours to complete (2.5 hours on H100).
- Rename the following files:
  - `mv ner_model.RENAME ner_model`
- To test the sentences, run: `python main.py` 

##### Part 3: Building a Voice Activated Assistant

The hands on components for this part is located in `demo` folder.

- Open 2 more console windows. Remember those `service.py` files in the previous steps? Do this in the other two Console windows:
  - Console Windows 1: `cd question && python service.py`
  - Console Windows 2: `cd named-entity-recognition && python service.py`

If you haven't set your `DEEPGRAM_API_KEY` and `GROQ_API_KEY` environment variables for this step, do this now!

Instructions on Console Window 3:

- Open `main.py` and follow along on the code walkthrough. 
- When instructed to do so, run: `python main.py` to start the model training.
  - If don't know how to set your environment variables, run: `DEEPGRAM_API_KEY=<your key> GROQ_API_KEY=<your key> python main.py`
- Speak this sentence:
  - "I live in <name the city you live in>."
  - "Where is the headerquarters for Microsoft?"

