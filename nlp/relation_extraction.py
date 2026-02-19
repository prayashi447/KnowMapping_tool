import pandas as pd
import itertools
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==============================
# 1. LOAD DATASET
# ==============================

print("Loading IMDB dataset...")
df = pd.read_csv("IMDB Dataset.csv")
df = df.head(2)   # small sample for testing

# ==============================
# 2. LOAD & CLEAN NER OUTPUT
# ==============================

print("Loading NER results...")
entities_df = pd.read_csv("IMDB_NER_Results_Sample.csv")

entities_df = entities_df[
    entities_df["label"].isin(["PER", "ORG", "MISC"])
]

entities_df = entities_df[
    entities_df["text"].str.len() > 3
]

entities_df = entities_df.drop_duplicates(subset=["review_id", "text"])

grouped_entities = entities_df.groupby("review_id")["text"].apply(list)

# ==============================
# 3. LOAD RELATION MODEL (BART MNLI)
# ==============================

print("Loading Relation Extraction model...")

model_name = "facebook/bart-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

model.eval()

candidate_relations = [
    "acted in",
    "directed",
    "produced",
    "written by",
    "released in",
    "based on",
    "worked with",
    "is a character in",
    "is part of"
]

CONFIDENCE_THRESHOLD = 0.5

# ==============================
# 4. RELATION EXTRACTION FUNCTION
# ==============================

def extract_relations(text, entities, review_id):

    relations = []

    entities = list(set(entities))

    if len(entities) < 2:
        return relations

    entity_pairs = list(itertools.combinations(entities, 2))

    for e1, e2 in entity_pairs:

        prompt = (
            f"In the following movie review:\n\n"
            f"{text}\n\n"
            f"What is the relationship between '{e1}' and '{e2}'?"
        )

        scores = []

        for label in candidate_relations:
            hypothesis = f"The relationship between {e1} and {e2} is {label}."

            inputs = tokenizer(
                prompt,
                hypothesis,
                return_tensors="pt",
                truncation=True
            )

            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits
            probs = F.softmax(logits, dim=1)

            entailment_score = probs[0][2].item()
            scores.append(entailment_score)

        best_idx = torch.tensor(scores).argmax().item()
        best_relation = candidate_relations[best_idx]
        confidence = scores[best_idx]

        if confidence >= CONFIDENCE_THRESHOLD:
            relations.append({
                "review_id": review_id,
                "entity_1": e1,
                "relation": best_relation,
                "entity_2": e2,
                "confidence": round(confidence, 3)
            })

    return relations


# ==============================
# 5. MAIN LOOP
# ==============================

print("Starting relation extraction...\n")

all_relations = []

for idx in tqdm(range(len(df))):

    review_text = df["review"].iloc[idx]

    if idx in grouped_entities.index:
        entities = grouped_entities[idx]
    else:
        entities = []

    print(f"\nReview {idx}")
    print("Entities:", entities)

    relations = extract_relations(review_text, entities, idx)

    print("Relations Found:")
    for r in relations:
        print(r)
        all_relations.append(r)

# ==============================
# 6. SAVE RESULTS
# ==============================

relations_df = pd.DataFrame(all_relations)

if not relations_df.empty:
    relations_df.to_csv("IMDB_Relations_Sample.csv", index=False)
    print("\n✓ Relations saved to IMDB_Relations_Sample.csv")
else:
    print("\nNo relations extracted.")
