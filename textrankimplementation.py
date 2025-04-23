import os
import spacy
import pytextrank
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from statistics import mean

# Load spaCy and add PyTextRank
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank", last=True)


# Load index keywords
def load_index_keywords(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return set(line.strip().lower() for line in f)


# Extract keywords using TextRank
def extract_keywords_from_file(filepath, top_n=50):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    doc = nlp(text)
    return [phrase.text.lower() for phrase in doc._.phrases[:top_n]]


# Filter out non-concept terms based on context (e.g., "good", "bad", "more")
def filter_keywords_by_significance(keywords, min_word_length=4):
    return [kw for kw in keywords if len(kw) >= min_word_length]


# Evaluate extraction with precision, recall, f1, accuracy
def evaluate_keywords(predicted, ground_truth, index_keywords):
    all_keywords = list(index_keywords)
    y_true = [1 if kw in ground_truth else 0 for kw in all_keywords]
    y_pred = [1 if kw in predicted else 0 for kw in all_keywords]

    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return p, r, f1, acc


# Build the concept map from extracted keywords
def build_concept_map(keywords_list, index_keywords):
    G = nx.Graph()
    for keywords in keywords_list:
        for i in range(len(keywords)):
            for j in range(i + 1, len(keywords)):
                k1, k2 = keywords[i], keywords[j]
                weight = 1

                # If both keywords are in the index, give them a higher weight
                if k1 in index_keywords and k2 in index_keywords:
                    weight = 3  # Increase the edge weight for indexed terms

                # If only one keyword is in the index, give it a medium weight
                elif k1 in index_keywords or k2 in index_keywords:
                    weight = 2

                # Add the edge with weight to the graph
                if G.has_edge(k1, k2):
                    G[k1][k2]['weight'] += weight
                else:
                    G.add_edge(k1, k2, weight=weight)
    return G


# Plot the concept map
def plot_concept_map(G):
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    centrality = nx.degree_centrality(G)
    sizes = [500 + centrality[n] * 2000 for n in G.nodes()]
    colors = [centrality[n] for n in G.nodes()]
    weights = [G[u][v]['weight'] for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors, cmap=plt.cm.viridis, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=[w * 0.5 for w in weights], alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

    plt.axis('off')
    plt.tight_layout()


# Main pipeline
def main():
    chapter_dir = 'chapters'
    chapter_files = sorted([os.path.join(chapter_dir, f) for f in os.listdir(chapter_dir) if f.endswith('.txt')])

    train_chapters = [1, 2, 3, 4, 5, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19]
    test_chapters = [6, 10, 11, 12]

    def chapter_num(f): return int(''.join(filter(str.isdigit, os.path.basename(f))))

    train_files = [f for f in chapter_files if chapter_num(f) in train_chapters]
    test_files = [f for f in chapter_files if chapter_num(f) in test_chapters]

    index_keywords = load_index_keywords("index_keywords.txt")

    precisions, recalls, f1s, accuracies = [], [], [], []

    print("\nChapter-Level Evaluation with Concept Maps\n")

    for file in train_files:
        chapter_id = chapter_num(file)

        # Extract keywords from file using TextRank
        extracted = extract_keywords_from_file(file, top_n=50)  # Increased to 100

        # Filter out irrelevant concepts like "good", "bad", "know", "more"
        filtered = filter_keywords_by_significance(extracted)

        # Remove keywords not in the index (only keep relevant concepts)
        filtered_indexed = [kw for kw in filtered if kw in index_keywords]

        with open(file, 'r', encoding='utf-8') as f:
            text = f.read().lower()
        ground_truth = [kw for kw in index_keywords if kw in text]

        # Evaluate the keywords extracted vs. the ground truth
        p, r, f1, acc = evaluate_keywords(extracted, ground_truth, index_keywords)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        accuracies.append(acc)

        print(f"{os.path.basename(file):<25} | P: {p:.2f}  R: {r:.2f}  F1: {f1:.2f}  Acc: {acc:.2f}")

        # Generate the concept map for each chapter, favoring indexed keywords
        concept_map = build_concept_map([filtered_indexed], index_keywords)

        # Only create one figure per chapter
        plt.figure(figsize=(14, 10))

        # Set the title for the concept map
        plt.title(f"Concept Map – Chapter {chapter_id}", fontsize=16)

        # Plot the concept map for this chapter
        plot_concept_map(concept_map)

        # Show the plot for this chapter
        plt.show()

    print("\nAverage Evaluation Over Train Chapters")
    print(f"  Precision: {mean(precisions):.2f}")
    print(f"  Recall:    {mean(recalls):.2f}")
    print(f"  F1 Score:  {mean(f1s):.2f}")
    print(f"  Accuracy:  {mean(accuracies):.2f}")


if __name__ == '__main__':
    main()
