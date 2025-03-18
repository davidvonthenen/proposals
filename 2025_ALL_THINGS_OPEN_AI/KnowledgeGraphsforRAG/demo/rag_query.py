import requests
import spacy
from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4jneo4j"

# The local LLM server endpoint
LLM_ENDPOINT = "http://127.0.0.1:5000/generate"

def connect_neo4j():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def extract_entities_spacy(text, nlp):
    doc = nlp(text)
    # Return entity strings & labels
    return [(ent.text.strip(), ent.label_) for ent in doc.ents if len(ent.text.strip()) >= 3]

def fetch_documents_by_entities(session, entity_texts, top_k=5):
    """
    We match by e.name (in lowercase for simpler matching).
    Then return up to top_k docs, sorted by the count of matched entities.
    """
    if not entity_texts:
        return []
    
    # print entities
    print("\n")
    print(f"Fetching Documents By Entity:")
    for entity in entity_texts:
        print(f" - {entity}")

    query = """
    MATCH (d:Document)-[:MENTIONS]->(e:Entity)
    WHERE toLower(e.name) IN $entity_list
    WITH d, count(e) as matchingEntities
    ORDER BY matchingEntities DESC
    LIMIT $topK
    RETURN d.title AS title, d.content AS content, d.category AS category, matchingEntities
    """
    # Convert entity texts to lowercase
    entity_list_lower = [txt.lower() for txt in entity_texts]

    results = session.run(
        query,
        entity_list=entity_list_lower,
        topK=top_k
    )

    # retrieve the documents
    print("\n")
    print("Retrieving Documents:")
    print("Title | Category | Matched Entities")
    docs = []
    for record in results:
        print(f"{record['title']} | {record['category']} | Matched Entities: {record['matchingEntities']}")
        docs.append({
            "title": record["title"],
            "content": record["content"],
            "category": record["category"],
            "match_count": record["matchingEntities"]
        })
    print("\n")
    return docs

def generate_answer(question, context):
    """
    Send prompt to the local LLM server. The server loads the model once
    and processes each prompt. This avoids re-loading on every query.
    """
    # Build our RAG-style prompt
    prompt = f"""You are given the following context from multiple documents:
{context}

Question: {question}

Please provide a concise answer.
Answer:
"""

    payload = {
        "prompt": prompt,
        "max_new_tokens": 2048
    }

    try:
        response = requests.post(LLM_ENDPOINT, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("answer", "")
    except Exception as e:
        print("Error calling LLM server:", e)
        return "Error generating answer"

if __name__ == "__main__":
    user_query = "What do these articles say about Ernie Wise?"
    print("User Query:", user_query)

    # Load spaCy for NER
    nlp = spacy.load("en_core_web_sm")

    # Extract relevant entities from the user query
    recognized_entities = extract_entities_spacy(user_query, nlp)
    print("\n\n")
    print("Recognized entities:", recognized_entities)

    # Convert entity tuples (text, label) into just their text for matching
    entity_texts = [ent[0] for ent in recognized_entities]

    # Connect to Neo4j and fetch top documents mentioning these entities
    driver = connect_neo4j()
    with driver.session() as session:
        docs = fetch_documents_by_entities(session, entity_texts, top_k=5)

    # Build a combined context string from top docs
    combined_context = ""
    for doc in docs:
        snippet = doc["content"][:300].replace("\n", " ")
        combined_context += f"\n---\nTitle: {doc['title']} | Category: {doc['category']}\nSnippet: {snippet}...\n"

    # Send the context + question to the LLM server endpoint
    final_answer = generate_answer(user_query, combined_context)

    print("RAG-based Answer:", final_answer)
