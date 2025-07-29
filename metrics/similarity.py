from typing import Any, Optional
import numpy as np
from transformers import AutoModel, AutoTokenizer
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher
import ast
import networkx as nx
import tokenize
import io

model_name = "google/electra-base-discriminator"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def cos_similarity(code_snippets: list[str]) -> float:
    """
    Computes the average semantic similarity between a list of code snippets.

    Args:
    - code_snippets: List of code snippets as strings.

    Returns:
    - A float representing the average pairwise semantic similarity score.
    """
    # Load a pretrained model and tokenizer for code understanding (e.g., CodeBERT)
    # model_name = "microsoft/codebert-base"

    # model_name = "bert-base-uncased"


    # Get embeddings for each code snippet
    def get_code_embedding(code: str):
        inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use the embedding of the [CLS] token as the sentence-level embedding
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        # Normalize the embedding
        return embedding / np.linalg.norm(embedding)

    # Get embeddings for all code snippets
    embeddings = np.vstack([get_code_embedding(snippet) for snippet in code_snippets])

    # Calculate pairwise cosine similarities
    similarities = cosine_similarity(embeddings)

    # Average the pairwise similarities, excluding self-similarities (the diagonal)
    n = len(code_snippets)
    if n < 2:
        return 1.0  # If there's only one snippet, similarity is trivially 1

    total_sim = (np.sum(similarities) - np.trace(similarities)) / (n * (n - 1))

    return total_sim


def tfidf_similarity(code_snippets: list[str]) -> float:
    """
    Calculate similarity using TF-IDF vectorization.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(code_snippets)
    similarities = cosine_similarity(tfidf_matrix)
    return float(np.mean(similarities[np.triu_indices(len(code_snippets), k=1)]))

def levenshtein_similarity(code_snippets: list[str]) -> float:
    """
    Calculate similarity using Levenshtein distance.
    """
    def levenshtein(s1, s2):
        if len(s1) < len(s2):
            return levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    max_length = max(len(s) for s in code_snippets)
    similarities = []
    for i in range(len(code_snippets)):
        for j in range(i+1, len(code_snippets)):
            distance = levenshtein(code_snippets[i], code_snippets[j])
            similarity = 1 - (distance / max_length)
            similarities.append(similarity)
    return float(np.mean(similarities))


def token_sequence_similarity(code_snippets: list[str]) -> float:
    """
    Calculate similarity based on token sequence matching.
    """
    def tokenize_code(code):
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
            return [token.string for token in tokens if token.string.strip()]
        except tokenize.TokenError:
            # Handle incomplete or invalid code
            return code.split()

    similarities = []
    for i in range(len(code_snippets)):
        for j in range(i+1, len(code_snippets)):
            tokens1 = tokenize_code(code_snippets[i])
            tokens2 = tokenize_code(code_snippets[j])
            matcher = SequenceMatcher(None, tokens1, tokens2)
            similarity = matcher.ratio()
            similarities.append(similarity)
    return float(np.mean(similarities))

def ast_similarity(code_snippets: list[str]) -> float:
    """
    Calculate similarity based on Abstract Syntax Tree (AST) structure.
    """
    def ast_to_graph(code):
        tree = ast.parse(code)
        graph = nx.Graph()
        for node in ast.walk(tree):
            graph.add_node(id(node), type=type(node).__name__)
            for child in ast.iter_child_nodes(node):
                graph.add_edge(id(node), id(child))
        return graph

    def graph_edit_distance(g1, g2):
        return nx.graph_edit_distance(g1, g2, node_match=lambda n1, n2: n1['type'] == n2['type'])

    graphs = [ast_to_graph(snippet) for snippet in code_snippets]
    max_nodes = max(len(g.nodes) for g in graphs)
    similarities = []
    for i in range(len(graphs)):
        for j in range(i+1, len(graphs)):
            distance = graph_edit_distance(graphs[i], graphs[j])
            similarity = 1 - (distance / max_nodes)
            similarities.append(similarity)
    return float(np.mean(similarities))
	


# best_solution['cosine_similarity'] = float(cos_similarity(proposed_solutions))
# best_solution['tfidf_similarity'] = tfidf_similarity(proposed_solutions)
# best_solution['levenshtein_similarity'] = levenshtein_similarity(proposed_solutions)
# # best_solution['ast_similarity'] = ast_similarity(proposed_solutions)
# best_solution['token_sequence_similarity'] = token_sequence_similarity(proposed_solutions)
