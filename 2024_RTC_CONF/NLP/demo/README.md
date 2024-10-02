# Running the Voice AI NLP Demo (NLP/demo)

This runs the final demo of the Voice AI Assistant which will answer any questions asked and also provide valuable insights for each type of Named Entity found in your speech.

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

### Running the Question and Named Entity Services

You need to run the Question and Entity Services in the other folders: `named-entity-recognition` and `question`.

Install the Python components required in each folder:

```bash
pip install -r requirements.txt
```

To start the `question` REST endpoint in the `question` folder:

```bash
python service.py
```

To start the `named-entity-recognition` REST endpoint in the `named-entity-recognition` folder:

```bash
python service.py
```

### Running the Agent

After running the `named-entity-recognition` and `question` services above, To run the Voice AI Agent demo...

Install the requirements:

```bash
pip install -r requirements.txt
```

Run the demo:

```bash
python demo.py
```
