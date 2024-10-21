# Workshop Using Google Colab

Welcome to landing page for workshop `Building Multiple Natural Language Processing Models to Work In Concert Together` at `2024 Open Data Science Conference West`.

## Prerequisites

Using the Google Colab Notebooks in this workshop is pretty straight forward, but you will need to setup the following **FREE** accounts for services we will be using in this workshop.

### Google Colab

As long as you have a Google account, you should have no problem accessing [Google Colab](https://colab.research.google.com/). If you don't have a Google account, please create one.

### Free SaaS Platforms Accounts Required

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

### Google Colab Notebooks

Here are the Google Colab Notebooks. Please do not advance until instructed to do so!

- Part 1: [Building a Question Classifier Model Notebook](https://colab.research.google.com/drive/1NAA4V1L99JY6ML9-_7R5DtzvlnypJaq_?usp=sharing)
- Part 2: [Building a Named Entity Recognition Model Notebook](https://drive.google.com/file/d/1qqbFnbp2aHH61KgRLaaC-4LfOytIbNDN/view?usp=sharing)
- Part 3: [Building a Voice Activated Assistant Notebook](https://drive.google.com/file/d/1bZVYP9zRYa8-NzrZCUMq7QXjqNovUbBa/view?usp=sharing)
  - Set the `DEEPGRAM_API_KEY` and `GROQ_API_KEY` environment variables for this step!
