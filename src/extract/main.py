import os
import dspy
import duckdb
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API configuration - loaded from environment variables or .env file
API_BASE = os.getenv("API_BASE", "http://localhost:11434")
API_KEY = os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "ollama_chat/qwen3:30b")


def load_schema_relationships(domain):
    """Load relation labels and constraints from DuckDB schema"""
    # Create a pandas DataFrame with relation labels and constraints from DuckDB
    schema_conn = duckdb.connect("schema_relationships.duckdb", read_only=True)
    schema_result = schema_conn.execute(
        f"""
        SELECT '' as first_entity, subject as first_entity_type, predicate as relationship, '' as second_entity, object as second_entity_type,
        FROM schema_relationships WHERE domain = 'base' OR domain = '{domain}'
        """
    ).df()
    examples = [EntityRelations(**row) for index, row in schema_result.iterrows()]

    schema_conn.close()
    print(f"Loaded {len(schema_result)} relations from schema_relationships.duckdb")
    return examples


class EntityRelations(BaseModel):
    first_entity: str = Field(
        ..., description="The first entity in the relationship with a type"
    )
    first_entity_type: str = Field(..., description="The type of the first entity")
    relationship: str = Field(
        ..., description="The relationship between the two entities, should be a verb"
    )
    second_entity: str = Field(
        ..., description="The second entity in the relationship with a type"
    )
    second_entity_type: str = Field(..., description="The type of the second entity")


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
                first_entity="Bill Clinton",
                first_entity_type="Person",
                relationship="memberOf",
                second_entity="President",
                second_entity_type="Person",
            ),
            EntityRelations(
                first_entity="Barack Obama",
                first_entity_type="Person",
                relationship="spouse",
                second_entity="Michelle Obama",
                second_entity_type="Person",
            ),
        ]
        # self.negative_examples = [EntityRelations(first_entity='Person', relationship='memberOf', second_entity='Organization')]
        # convert schema_df to cypher like string
        schema_str = "\n".join(
            f"(({row.first_entity_type})-[:{row.relationship}]->({row.second_entity_type}))"
            for row in schema_df
        )
        self.schema_string = "Valid relationships: " + schema_str
        print(self.schema_string)

    def forward(self, text):
        return self.extract(text=text, relationship_schema=self.schema_string)


module = EntityExtraction(load_schema_relationships("politics"))


def extract_entities_and_relations(extractor, sentence):
    return module(sentence).entities


def build_knowledge_graph(extractor, text):
    sentences = sent_tokenize(text)
    graph = nx.DiGraph()

    for sentence in sentences:
        try:
            ER = extract_entities_and_relations(extractor, sentence)
            print(f"Extracted: {ER}")
            entity1 = ER.first_entity
            entity2 = ER.second_entity
            relationship = ER.relationship
            if entity1 and entity2:
                graph.add_node(entity1)
                graph.add_node(entity2)
                graph.add_edge(entity1, entity2, relation=relationship)
        except Exception as e:
            print(
                f"Failed to extract or add entities for the sentence '{sentence}': {e}"
            )

    draw_knowledge_graph(graph)


def draw_knowledge_graph(graph):
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(12, 12))
    nx.draw(graph, pos, with_labels=True, node_size=300, node_color="skyblue")
    edge_labels = nx.get_edge_attributes(graph, "relation")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels)
    plt.title("Knowledge Graph")
    plt.show()


def configure_dspy():
    # Using Ollama with configuration from environment variables
    lm = dspy.LM(
        model=MODEL_NAME,
        api_base=API_BASE,
        api_key=API_KEY,
    )
    dspy.configure(lm=lm)
    return dspy.Predict(EntityExtraction)


entity_extractor = configure_dspy()


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

build_knowledge_graph(entity_extractor, text)
