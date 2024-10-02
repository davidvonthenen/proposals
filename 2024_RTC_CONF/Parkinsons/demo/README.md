# Running the Voice AI Parkinsons Demo (Parkinsons/demo)

This runs the final demo of the Voice AI Assistant answering questions and giving the results for the 2 fictious patients.

## Prerequisites

### Deepgram Account

You need a Deepgram account in order to do the Speech-to-Text and Text-to-Speech. The account is free! Sign up below:
https://deepgram.com/

Set the API Key to the following environment variable: `DEEPGRAM_API_KEY`.

### OpenAI Account

You need an OpenAI API account in order to the LLM processing and agent component (aka the responses). Sign up below:
https://platform.openai.com/docs/overview

Set the API Key to the following environment variable: `OPENAI_API_KEY`.

### Environment Prerequisites

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

To run the example:

```bash
python demo.py
```
