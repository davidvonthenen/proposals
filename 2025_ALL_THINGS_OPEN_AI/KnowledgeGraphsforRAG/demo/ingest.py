import os
import uuid
import spacy
from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4jneo4j"

DATASET_PATH = "./bbc"  # Path to the unzipped BBC dataset folder

def ingest_bbc_documents_with_ner():
    # Load spaCy for NER
    nlp = spacy.load("en_core_web_sm")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        # Optional: clear old data
        session.run("MATCH (n) DETACH DELETE n")

        for category in os.listdir(DATASET_PATH):
            category_path = os.path.join(DATASET_PATH, category)
            if not os.path.isdir(category_path):
                continue  # skip non-directories

            for filename in os.listdir(category_path):
                if filename.endswith(".txt"):
                    filepath = os.path.join(category_path, filename)
                    # FIX #1: handle potential £ symbol or other characters
                    # Option 1: Use a different codec
                    # with open(filepath, "r", encoding="latin-1") as f:
                    #   text_content = f.read()
                    #
                    # Option 2: Replace invalid bytes (keep utf-8):
                    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                        text_content = f.read()

                    # Generate a UUID in Python
                    doc_uuid = str(uuid.uuid4())

                    # Create (or MERGE) the Document node
                    create_doc_query = """
                    MERGE (d:Document {doc_uuid: $doc_uuid})
                    ON CREATE SET
                        d.title = $title,
                        d.content = $content,
                        d.category = $category
                    RETURN d
                    """
                    session.run(
                        create_doc_query,
                        doc_uuid=doc_uuid,
                        title=filename,
                        content=text_content,
                        category=category
                    )

                    # Named Entity Recognition
                    doc_spacy = nlp(text_content)

                    # For each entity recognized, MERGE on name+label
                    for ent in doc_spacy.ents:
                        # Skip small or numeric or purely punctuation
                        if len(ent.text.strip()) < 3:
                            continue

                        # Generate a unique ID for new entities
                        entity_uuid = str(uuid.uuid4())

                        merge_entity_query = """
                        MERGE (e:Entity { name: $name, label: $label })
                        ON CREATE SET e.ent_uuid = $ent_uuid
                        RETURN e.ent_uuid as eUUID
                        """
                        record = session.run(
                            merge_entity_query,
                            name=ent.text.strip(),
                            label=ent.label_,
                            ent_uuid=entity_uuid
                        ).single()

                        ent_id = record["eUUID"]

                        # Now create relationship by matching on doc_uuid & ent_uuid
                        rel_query = """
                        MATCH (d:Document { doc_uuid: $docId })
                        MATCH (e:Entity { ent_uuid: $entId })
                        MERGE (d)-[:MENTIONS]->(e)
                        """
                        session.run(
                            rel_query,
                            docId=doc_uuid,
                            entId=ent_id
                        )

    print("Ingestion with NER complete!")

if __name__ == "__main__":
    ingest_bbc_documents_with_ner()
