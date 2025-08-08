import os
import dspy
import duckdb
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field
from nltk.tokenize import sent_tokenize

# API configuration - set these environment variables or modify as needed
API_BASE = os.getenv("API_BASE", "http://192.168.68.54:11434")
API_KEY = os.getenv("API_KEY", "your-api-key-here")


def load_schema_relationships(domain):
    """Load relation labels and constraints from DuckDB schema"""
    # Create a pandas DataFrame with relation labels and constraints from DuckDB
    schema_conn = duckdb.connect("schema_relationships.duckdb", read_only=True)
    schema_result = schema_conn.execute(
        f"SELECT subject as first_entity, predicate as relationship, object as second_entity FROM schema_relationships WHERE domain = 'base' OR domain = '{domain}'"
    ).df()
    examples = [EntityRelations(**row) for index, row in schema_result.iterrows()]

    schema_conn.close()
    print(f"Loaded {len(schema_result)} relations from schema_relationships.duckdb")
    print(examples[:5])  # Print first 5 examples for debugging
    return examples


class EntityRelations(BaseModel):
    first_entity: str = Field(..., description="The first entity in the relationship")
    relationship: str = Field(
        ..., description="The relationship between the two entities, should be a verb"
    )
    second_entity: str = Field(..., description="The second entity in the relationship")


class EntityExtraction(dspy.Signature):
    text: str = dspy.InputField(desc="The text to extract entities from")
    entities: EntityRelations = dspy.OutputField(
        desc="Entities and their relationships extracted from the text"
    )


class EntityExtractionModule(dspy.Module):
    def __init__(self, examples):
        self.extract = dspy.ChainOfThought(EntityExtraction)
        self.examples = examples

    def forward(self, text):
        # DSPy will automatically use examples for few-shot prompting
        with dspy.context(examples=self.examples[:5]):  # Use top 5 examples
            return self.extract(text=text)


module = EntityExtractionModule(load_schema_relationships("politics"))


def extract_entities_and_relations(extractor, sentence):
    return module(sentence).entities


def build_knowledge_graph(extractor, text):
    sentences = sent_tokenize(text)
    graph = nx.DiGraph()

    for sentence in sentences:
        try:
            entity1, relationship, entity2 = extract_entities_and_relations(
                extractor, sentence
            )
            print(f"Extracted: {entity1}, {relationship}, {entity2}")
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
    # Using Ollama with the correct host
    lm = dspy.LM(
        model="ollama_chat/qwen3:30b",
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
