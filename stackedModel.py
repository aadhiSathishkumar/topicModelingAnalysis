from course_corpus import (
    courses,
    course_names,
    course_descriptions,
    preprocess_text,
    get_cleaned_descriptions,
)

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Curriculum anchors used only for post-hoc labeling after models finish.
# They do not influence fitting, just here for output clarity (no weird topic names).
CURRICULUM_SEEDS = {
    "circuits": ["kirchhoff", "mesh", "nodal", "thevenin", "norton", "phasor", "rlc", "transient"],
    "digital": ["boolean", "karnaugh", "combinational", "sequential", "vhdl", "flipflop", "gate"],
    "electronics": ["transistor", "diode", "amplifier", "bjt", "fet", "opamp", "bias", "semiconductor"],
    "signals": ["fourier", "laplace", "sampling", "baseband", "modulation", "spectrum"],
    "power": ["transformer", "synchronous", "induction", "electromechanic", "machine", "perunit"],
    "controls": ["plc", "pid", "compensator", "stability", "feedback", "automation"],
    "architecture": ["embedded", "realtime", "microcontroller", "pipeline", "cache", "scheduling"],
    "networks": ["wireless", "ethernet", "routing", "packet", "encryption", "protocol"],
    "intelligence": ["neural", "fuzzy", "evolutionary", "vision", "quantum", "game"],
}

# Terms that appear in more than one area and must not decide the anchor.
AMBIGUOUS = {"network", "logic", "power", "communication", "system", "design", "analysis", "signal", "circuit"}


def _token_hits(text, term):
    """Count whole-token hits so 'network' does not match inside unrelated compounds only by substring."""
    n = 0
    for tok in text.split():
        if tok == term or tok.startswith(term + "_") or tok.endswith("_" + term):
            n += 1
    return n


def _label_topic(top_words, cleaned_docs=None, doc_idxs=None):
    """Match a finished topic's top words to the best curriculum seed. Purely for labeling."""
    scores = {name: 0 for name in CURRICULUM_SEEDS}
    for word in top_words:
        if word in AMBIGUOUS:
            continue
        for name, terms in CURRICULUM_SEEDS.items():
            if word in terms or any(word == t or word.endswith("_" + t) or word.startswith(t + "_") for t in terms):
                scores[name] += 3
    # Optional extra signal from the documents that loaded most heavily on this topic
    if cleaned_docs is not None and doc_idxs:
        blob = " ".join(cleaned_docs[i] for i in doc_idxs)
        for name, terms in CURRICULUM_SEEDS.items():
            scores[name] += sum(_token_hits(blob, term) for term in terms)
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return None
    return best


def run(min_topic_size=2, show=True, max_topic_share=0.25):
    """BERTopic finds groups. NMF fits unsupervised. Curriculum seeds label topics afterward.

    A BERTopic topic that holds more than max_topic_share of the corpus
    is excluded when choosing K for NMF so giant dumps do not force a single column.
    """
    cleaned_descriptions = get_cleaned_descriptions()

    shared_token_pattern = r"(?u)\b\w+\b"
    vectorizer_kwargs = dict(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        token_pattern=shared_token_pattern,
    )

    topic_model = BERTopic(
        min_topic_size=min_topic_size,
        vectorizer_model=CountVectorizer(**vectorizer_kwargs),
    )
    topics, probs = topic_model.fit_transform(cleaned_descriptions)
    topics = topic_model.reduce_outliers(cleaned_descriptions, topics)
    topic_model.update_topics(cleaned_descriptions, topics)

    topic_info = topic_model.get_topic_info()
    bertopic_ids = sorted(int(t) for t in topic_info["Topic"].tolist() if t != -1)
    n_docs = len(cleaned_descriptions)
    counts = {tid: sum(t == tid for t in topics) for tid in bertopic_ids}

    # Use only non-giant BERTopic topics to set K for NMF.
    usable_ids = [tid for tid in bertopic_ids if counts[tid] / n_docs <= max_topic_share]
    if len(usable_ids) < 2:
        usable_ids = bertopic_ids
    num_topics = len(usable_ids)

    if show:
        print("BERTopic topic info:")
        print(topic_info)
        print(f"Using {num_topics} BERTopic topics as NMF columns (giant dumps excluded)")
        topic_model.visualize_topics().show()
        topic_model.visualize_barchart(top_n_topics=max(len(bertopic_ids), 1), n_words=8).show()
        for course, topic in zip(course_names, topics):
            print(f"{course}: BERTopic {topic}")

    vectorizer = TfidfVectorizer(**vectorizer_kwargs)
    X = vectorizer.fit_transform(cleaned_descriptions)
    feature_names = np.array(vectorizer.get_feature_names_out())

    # Fully unsupervised NMF
    nmf_model = NMF(
        n_components=num_topics,
        random_state=42,
        max_iter=500,
    )
    W = nmf_model.fit_transform(X)
    H = nmf_model.components_

    # Post-hoc labeling only
    col_anchor = {}
    for i in range(num_topics):
        top_idx = np.argsort(H[i])[::-1][:12]
        top_words = [feature_names[j] for j in top_idx]
        # Docs that load most on this topic (for optional extra matching signal)
        doc_idxs = np.argsort(W[:, i])[::-1][:max(5, n_docs // num_topics)]
        anchor = _label_topic(top_words, cleaned_descriptions, doc_idxs)
        col_anchor[i] = anchor or f"topic_{i+1}"

    if show:
        print("\nNMF topics (unsupervised fit, labeled afterward):")
        for i in range(num_topics):
            top_idx = np.argsort(H[i])[::-1][:8]
            top_words = ", ".join(feature_names[j] for j in top_idx)
            print(f"  NMF Topic {i + 1} [{col_anchor[i]}]: {top_words}")

    topic_names = [f"{i + 1}:{col_anchor.get(i, 'topic')}" for i in range(num_topics)]
    course_topic_df = pd.DataFrame(W, index=course_names, columns=topic_names)

    if show:
        plt.figure(figsize=(12, 10))
        sns.heatmap(course_topic_df, cmap="magma", annot=True, fmt=".2f")
        plt.title(f"Unsupervised NMF (K={num_topics}, labeled after fitting)")
        plt.xlabel("Topics")
        plt.ylabel("Courses")
        plt.tight_layout()
        plt.show()

    return {
        "topic_model": topic_model,
        "topics": topics,
        "probs": probs,
        "topic_info": topic_info,
        "num_topics": num_topics,
        "anchors": col_anchor,
        "nmf_model": nmf_model,
        "W": W,
        "H": H,
        "vectorizer": vectorizer,
        "course_topic_df": course_topic_df,
        "cleaned_descriptions": cleaned_descriptions,
    }


if __name__ == "__main__":
    run()
