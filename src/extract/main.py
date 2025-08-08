import os
import dspy
import duckdb
import pandas as pd
import kuzu
import shutil
from pydantic import BaseModel, Field
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv
from typing import List

# Load environment variables from .env file
load_dotenv()

# API configuration - loaded from environment variables or .env file
# API_BASE: Only needed for local models like Ollama (default: http://localhost:11434)
# For remote APIs (Gemini, OpenAI, Anthropic, etc.), leave API_BASE empty or unset
API_BASE = os.getenv("API_BASE", "")  # Default to empty for remote APIs
API_KEY = os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "ollama_chat/qwen3:30b")


def load_schema_relationships(domain):
    """Load relation labels and constraints from DuckDB schema"""
    # Create a pandas DataFrame with relation labels and constraints from DuckDB
    schema_conn = duckdb.connect("schema_relationships.duckdb", read_only=True)
    schema_result = schema_conn.execute(
        f"""
        SELECT subject as first_entity_type, predicate as relationship, object as second_entity_type
        FROM schema_relationships WHERE domain = 'base' OR domain = '{domain}'
        """
    ).df()

    schema_conn.close()
    print(f"Loaded {len(schema_result)} relations from schema_relationships.duckdb")
    return schema_result


class Entity(BaseModel):
    id: int
    name: str
    type: str


class Relations(BaseModel):
    first: int = Field(..., description="ID of the first entity")
    type: str = Field(..., description="Type of the relationship")
    second: int = Field(..., description="ID of the second entity")


class EntityRelations(BaseModel):
    entities: List[Entity]
    relations: List[Relations]


class EntityExtractionWithSchema(dspy.Signature):
    text: str = dspy.InputField(desc="The text to extract entities from")
    relationship_schema: str = dspy.InputField(
        desc="Valid relationship types from schema"
    )
    entities: EntityRelations = dspy.OutputField(
        desc="Entities and their relationships extracted from the text. The relationship must be one from the provided schema. "
        + "Entity types should match the schema. Do NOT output schema itself."
    )


class EntityExtraction(dspy.Module):
    def __init__(self, schema_df):
        self.extract = dspy.ChainOfThought(EntityExtractionWithSchema)
        self.positive_examples = [
            EntityRelations(
                entities=[Entity(id=1, name="Bill Clinton", type="Person")],
                relations=[Relations(first=1, second=2, type="memberOf")],
            ),
            EntityRelations(
                entities=[
                    Entity(id=3, name="Barack Obama", type="Person"),
                    Entity(id=4, name="Michelle Obama", type="Person"),
                ],
                relations=[Relations(first=3, second=4, type="spouse")],
            ),
        ]
        # convert schema_df to cypher like string
        schema_str = "\n".join(
            f"(({row[0]})-[:{row[1]}]->({row[2]}))"
            for row in schema_df.itertuples(index=False)
        )
        self.schema_string = "Valid relationships: " + schema_str
        print(self.schema_string)

    def forward(self, text):
        return self.extract(text=text, relationship_schema=self.schema_string)


def extract_entities_and_relations(module, sentence):
    return module(sentence).entities


def build_knowledge_graph(text):
    # Configure DSPy
    # Only set api_base for local models (like Ollama), not for remote APIs (Gemini, OpenAI, etc.)
    if API_BASE and ("ollama" in MODEL_NAME.lower() or "localhost" in API_BASE):
        # For local models like Ollama
        lm = dspy.LM(
            model=MODEL_NAME,
            api_key=API_KEY,
            api_base=API_BASE,
        )
    else:
        # For remote APIs like Gemini, OpenAI, etc. - don't pass api_base
        lm = dspy.LM(
            model=MODEL_NAME,
            api_key=API_KEY,
        )

    dspy.configure(lm=lm)

    # Load schema and create extraction module
    schema_data = load_schema_relationships("politics")
    module = EntityExtraction(schema_data)

    sentences = sent_tokenize(text)

    # Initialize KuzuDB database
    db_path = "knowledge_graph.kuzu"
    # Remove existing database if it exists
    if os.path.exists(db_path):
        if os.path.isdir(db_path):
            shutil.rmtree(db_path)
        else:
            os.remove(db_path)

    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)

    # Create node table for entities
    conn.execute(
        "CREATE NODE TABLE Entity(name STRING, type STRING, PRIMARY KEY(name))"
    )

    # Create relationship table
    conn.execute(
        "CREATE REL TABLE Relation(FROM Entity TO Entity, relation_type STRING)"
    )

    for sentence in sentences:
        try:
            ER = extract_entities_and_relations(module, sentence)
            print(f"Extracted: {ER}")

            # Add entities to the graph
            for e in ER.entities:
                try:
                    conn.execute(
                        f"CREATE (:Entity {{name: '{e.name}', type: '{e.type}'}})"
                    )
                except Exception as insert_error:
                    # Entity might already exist, that's okay
                    print(f"Entity '{e.name}' might already exist: {insert_error}")

            # Add relationships to the graph
            for r in ER.relations:
                try:
                    # Find entity names by ID
                    first_entity = next(
                        (e.name for e in ER.entities if e.id == r.first), None
                    )
                    second_entity = next(
                        (e.name for e in ER.entities if e.id == r.second), None
                    )

                    if first_entity and second_entity:
                        query = f"""
                        MATCH (a:Entity {{name: '{first_entity}'}}), (b:Entity {{name: '{second_entity}'}})
                        CREATE (a)-[:Relation {{relation_type: '{r.type}'}}]->(b)
                        """
                        conn.execute(query)
                except Exception as rel_error:
                    print(f"Failed to add relationship: {rel_error}")

        except Exception as e:
            print(
                f"Failed to extract or add entities for the sentence '{sentence}': {e}"
            )

    query_knowledge_graph(conn)


def query_knowledge_graph(conn):
    """Query and display the knowledge graph stored in KuzuDB"""
    print("\n=== Knowledge Graph Summary ===")

    # Count entities
    result = conn.execute("MATCH (e:Entity) RETURN count(*) as entity_count")
    entity_count = result.get_next()[0]
    print(f"Total entities: {entity_count}")

    # Count relationships
    result = conn.execute("MATCH ()-[r:Relation]->() RETURN count(*) as rel_count")
    rel_count = result.get_next()[0]
    print(f"Total relationships: {rel_count}")

    # Show all entities
    print("\n=== Entities ===")
    result = conn.execute("MATCH (e:Entity) RETURN e.name, e.type ORDER BY e.name")
    while result.has_next():
        row = result.get_next()
        print(f"- {row[0]} ({row[1]})")

    # Show all relationships
    print("\n=== Relationships ===")
    result = conn.execute(
        """
        MATCH (a:Entity)-[r:Relation]->(b:Entity) 
        RETURN a.name, r.relation_type, b.name 
        ORDER BY a.name
    """
    )
    while result.has_next():
        row = result.get_next()
        print(f"- {row[0]} --[{row[1]}]--> {row[2]}")

    # Show entities by type
    print("\n=== Entities by Type ===")
    # Use implicit grouping in Cypher - group by the non-aggregate expression (e.type)
    result = conn.execute(
        "MATCH (e:Entity) RETURN e.type, count(*) ORDER BY count(*) DESC"
    )
    while result.has_next():
        row = result.get_next()
        print(f"- {row[0]}: {row[1]}")

    print("\n" + "=" * 50)


text = """
Abraham Lincoln[b] (February 12, 1809 – April 15, 1865) was the 16th
president of the United States, serving from 1861 until his assassination
in 1865. He led the United States through the American Civil War,
defeating the Confederate States of America and playing a major role in
the abolition of slavery.

Lincoln was born into poverty in Kentucky and raised on the frontier. He
was self-educated and became a lawyer, Illinois state legislator,
and U.S. representative. Angered by the Kansas–Nebraska Act of 1854,
which opened the territories to slavery, he became a leader of the new
Republican Party. He reached a national audience in the 1858 Senate
campaign debates against Stephen A. Douglas. Lincoln won the 1860
presidential election, prompting the slave states to begin to secede and
form the Confederate States of America. A month after Lincoln assumed
the presidency, Confederate forces attacked Fort Sumter, starting the
Civil War.
"""

if __name__ == "__main__":
    build_knowledge_graph(text)
