import os
import re
from rake_nltk import Rake
from rapidfuzz import process
import matplotlib.pyplot as plt
import networkx as nx
import spacy

# Load spaCy model for part-of-speech tagging
nlp = spacy.load("en_core_web_sm")

# Clean and normalize text
def clean_text(text):
    text = re.sub(r'[\u00AD\u2010\u2011\u2012\u2013\u2014\u2212]', '-', text)  # Normalize dashes
    text = re.sub(r'-\s*\n\s*', '', text)  # Remove hyphenation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces and line breaks
    return text.lower()

# Normalize keywords for matching purposes (lowercase, remove special characters)
def normalize_keywords(keywords):
    return [re.sub(r'[^a-z0-9\s]', '', kw.strip().lower()) for kw in keywords]

# Extract keywords using RAKE
def extract_keywords_rake(filepath, top_n=50):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    cleaned_text = clean_text(raw_text)
    rake = Rake()
    rake.extract_keywords_from_text(cleaned_text)
    ranked_phrases = rake.get_ranked_phrases()[:top_n]
    return normalize_keywords(ranked_phrases), cleaned_text

# Load chapter-specific index keywords from each chapter's index file
def load_chapter_index_keywords(chapter_id):
    filename = f"chapter{chapter_id}_index.txt"
    chapter_keywords = set()
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                keywords = normalize_keywords(line.split(","))
                chapter_keywords.update(keywords)
    return chapter_keywords

# Exact and fuzzy matching for better comparison between extracted and chapter-specific index
def match_keywords_with_index(extracted, chapter_index_keywords, threshold=80):
    matched_keywords = []
    for extracted_kw in extracted:
        if extracted_kw in chapter_index_keywords:
            matched_keywords.append(extracted_kw)
        else:
            matches = process.extract(extracted_kw, chapter_index_keywords, limit=3)
            best_match = matches[0] if matches else None
            if best_match and best_match[1] >= threshold:
                matched_keywords.append(best_match[0])

    return matched_keywords

# Build concept map with edge limitations
def build_concept_map(keywords_list, chapter_index_keywords, max_edges=5, weight_threshold=2):
    G = nx.Graph()
    for keywords in keywords_list:
        for i in range(len(keywords)):
            for j in range(i + 1, len(keywords)):
                k1, k2 = keywords[i], keywords[j]
                weight = 3 if k1 in chapter_index_keywords and k2 in chapter_index_keywords else 2
                if weight >= weight_threshold:
                    if G.has_edge(k1, k2):
                        G[k1][k2]['weight'] += weight
                    else:
                        G.add_edge(k1, k2, weight=weight)

    for node in list(G.nodes):
        neighbors = list(G.neighbors(node))
        if len(neighbors) > max_edges:
            sorted_neighbors = sorted(neighbors, key=lambda x: G[node][x]['weight'], reverse=True)
            for neighbor in neighbors:
                if neighbor not in sorted_neighbors[:max_edges]:
                    G.remove_edge(node, neighbor)

    return G

# Plot concept map with clusters closer together
def plot_concept_map(G):
    if not G.nodes:
        return
    pos = nx.spring_layout(G, k=0.1, iterations=100)  # Adjust layout parameters
    centrality = nx.degree_centrality(G)
    sizes = [500 + centrality[n] * 2000 for n in G.nodes()]
    colors = [centrality[n] for n in G.nodes()]
    weights = [G[u][v]['weight'] for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors, cmap=plt.cm.viridis, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=[w * 0.5 for w in weights], alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

    plt.axis('off')
    plt.tight_layout()

# Main function
def main():
    chapter_dir = 'chapters'
    chapter_files = sorted([os.path.join(chapter_dir, f) for f in os.listdir(chapter_dir) if f.endswith('.txt')])

    # Define training and testing chapters
    train_chapters = [1, 2, 3, 4, 5, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19]
    test_chapters = [6, 10, 11, 12]

    # Train and test on chapters
    for files, phase in zip([train_chapters, test_chapters], ['Train', 'Test']):
        for chapter_id in files:
            file = chapter_files[chapter_id - 1]  # Get file for this chapter
            extracted, cleaned_text = extract_keywords_rake(file)
            filtered = [kw for kw in extracted if len(kw.split()) > 1]  # Only keep multi-word keywords

            # Load chapter-specific index keywords from the corresponding index file
            chapter_index = load_chapter_index_keywords(chapter_id)

            # Match extracted keywords with the chapter's index keywords
            matched = match_keywords_with_index(filtered, chapter_index, threshold=80)

            # Build and plot concept map with limited edges
            concept_map = build_concept_map([matched], chapter_index, max_edges=5, weight_threshold=2)

            # Plot the concept map
            plt.figure(figsize=(12, 8))
            plt.title(f"Concept Map – Chapter {chapter_id}")
            plot_concept_map(concept_map)
            plt.show()

if __name__ == '__main__':
    main()
