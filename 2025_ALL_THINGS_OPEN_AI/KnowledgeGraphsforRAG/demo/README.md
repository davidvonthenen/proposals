# Quick Guide to Building a RAG Solution Using a Graph Database

This demo ingests a dataset based on some British Broadcasting Corporation (BBC) News data and stories.

## Prerequisites

Have only tested on MacOS, but should also work on most favors of Linux.

Using:

- Python 3.10+
- Docker Desktop

I would highly recommend using something like:

- conda - <https://docs.anaconda.com/free/miniconda/>
- venv - <https://docs.python.org/3/library/venv.html>

## Installation

### Setup Python

Install the required packages by running in your (virtual) environment:

```bash
pip install -r requirements.txt

# install the NER library
python -m spacy download en_core_web_sm
```

### Starting Neo4j on Docker

To start a Neo4j instance on Docker, run:

```bash
docker run \
    -d \
    --publish=7474:7474 --publish=7687:7687 \
    -v $HOME/neo4j/data:/data \
    -v $HOME/neo4j/logs:/logs \
    -v $HOME/neo4j/import:/var/lib/neo4j/import \
    -v $HOME/neo4j/plugins:/plugins \
    neo4j:5
```

### Run Llama 3.3

Start your LLM locally...

```bash
python server.py
```

## Running the Example

### Ingest the Data

```bash
python ingest.py
```

### Query the RAG Agent

```bash
python rag_query.py
```

### Query Neo4j Using Traditional Methods

```bash
python standalone.py
```

### Graph Query in Neo4j Console

```
MATCH (e:Entity)-[r]-(n)
WHERE toLower(e.name) = 'ernie wise'
RETURN e, r, n
```
