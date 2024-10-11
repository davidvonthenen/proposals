# 2024 ODSC West - Building Multiple Natural Language Processing Models to Work In Concert Together

Welcome to landing page for workshop `Building Multiple Natural Language Processing Models to Work In Concert Together` at `2024 Open Data Science Conference West`.

## What to Expect

The intent of this workshop is to provide an introduction into running a representative NLP application:

- building NLP models of the first time
- using multiple NLP models in a production-like environment

Without requiring GPU resources or the lengthy training times.

### Latest Resources

You will be able to find the latest version of the resources on this GitHub page:
[https://github.com/dvonthenen/proposals/tree/main/2024_ODSC/BuildMultipleNLPModels/README.md](https://github.com/dvonthenen/proposals/tree/main/2024_ODSC/BuildMultipleNLPModels/README.md)

## Prerequisites

There are 2 options available to participate in this workshop:

## Option 1: The Easy Way to Participate

If you are just looking to dip your toe into learning about Natural Language Processing (NLP) models and perhaps run some of the examples, we will offer several [Google Colab Notebooks](https://colab.research.google.com/) where you will be able to participate in the workshop without requiring any prerequisites software or configuration.

However, you will miss out on the full experience of this workshop which is designed to immerse yourself into a practical and "real-world" project using multiple NLP models.

## Option 2: A Full Production-like Experience

If you opt for the full experience, here are the prerequisites both from a hardware/operating system point of view as well as from software components required to run the examples and code in this workshop.

### Hardware/Environment Prerequisites

To run this workshop, it's required that you are on a Linux-based operating system. All materials have been tested on the following operating systems:

- MacOS on Apple Silicon (14.5)
- Ubuntu x64 (24.04)
- RHEL x64 (9.X)
- Rocky x64 (9.X)

Other flavors of Linux should also work without any issues, but we have not explicitly verified those platform versions. You may be on your own, but you can always fallback to the [Google Colab Notebooks](https://colab.research.google.com/).

#### Windows Users

If you happen to be a Windows user, I would recommend either:

- **Easy at the Trade-off for Cost:** Using a Cloud Instance
  - AWS EC2
  - GCP
  - Azure
- **More Effort but Could be Free:** Using a Hypervisor (like [VMware Workstation](https://www.vmware.com/products/desktop-hypervisor/workstation-and-fusion)) to run a Linux Virtual Machine

We will not be providing either Hypervisor software OR Cloud Instances access. Should you choose this option, obtaining these resources along with understand the cost associated with using them (ie Cloud Instances), is entirely on you.

### Software Components

- Python 3.10+

I would highly recommend using something like:

- conda - <https://docs.anaconda.com/free/miniconda/>
- venv - <https://docs.python.org/3/library/venv.html>

Once you have a `conda` or `venv` environment, create a virtual workspace called `2024-odsc-west`. Then install the requirements components for Python:

```bash
pip install -r requirements.txt
```

### Course Material

Check back here closer to the conference date for downloadable course material.

### SaaS Platforms

For the last third of the lab, we are going to make use of the following SaaS platforms:

- [Deepgram](https://deepgram.com) - This is to help facilitate building an Voice-Activated AI Assistant/Agent.
- **Tentatively** [groq](https://console.groq.com/login) - This is to provide an LLM in the examples.

#### Deepgram Account

You need a Deepgram account in order to do the Speech-to-Text and Text-to-Speech. The account is free and you get $200 in credits! [Sign up at Deepgram.com](https://deepgram.com)

Set the API Key to the following environment variable: `DEEPGRAM_API_KEY`.

#### groq Account

**Tentatively:** You need an groq API Key in order to make use of an LLM to provide responses. [Sign up at groq](https://console.groq.com/login).

Set the API Key to the following environment variable: `GROQ_API_KEY`.

**Alternatively:** We may provide a local LLM like LLaMA.
