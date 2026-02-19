import pandas as pd
from transformers import pipeline
from collections import Counter

# ==============================
# Load Dataset
# ==============================

df = pd.read_csv('IMDB Dataset.csv')
df = df.head(10)

# ==============================
# Initialize NER Pipeline
# ==============================

print("Loading BERT NER model...")

ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    grouped_entities=True,
    device=-1  # CPU
)

# ==============================
# Heentity_typeslper Functions
# ==============================

def clean_entity_text(text):
    """Remove BERT subword tokens (##)"""
    return text.replace(" ##", "").replace("##", "")

def extract_entities_transformer(text, threshold=0.85):
    """
    Extract named entities using BERT NER
    Apply confidence threshold filtering
    """
    try:
        if len(text) > 1000:
            text = text[:1000]

        entities = ner_pipeline(text)

        # Filter low-confidence entities
        entities = [e for e in entities if e['score'] >= threshold]

        return entities

    except Exception as e:
        print(f"Error processing text: {e}")
        return []

# ==============================
# Entity Type Descriptions
# ==============================

 = {
    'PER': 'PERSON (Actors, Directors, etc.)',
    'ORG': 'ORGANIZATION (Studios, Production Companies)',
    'LOC': 'LOCATION (Filming Locations)',
    'MISC': 'MISCELLANEOUS'
}

# ==============================
# Test Single Review
# ==============================

print("\n" + "="*80)
print("TESTING ON SAMPLE REVIEW")
print("="*80)

sample_review = df['review'].iloc[0]
print(f"Sample Review: {sample_review[:200]}...\n")

entities = extract_entities_transformer(sample_review)

print("Extracted Entities:")
for entity in entities:
    text_clean = clean_entity_text(entity['word'])
    label = entity['entity_group']
    description = entity_types.get(label, label)

    print(f"  Text: {text_clean:30} | Label: {label:10} | Type: {description:35} | Score: {entity['score']:.3f}")

# ==============================
# Process First 10 Reviews
# ==============================

print("\n" + "="*80)
print("PROCESSING FIRST 10 REVIEWS")
print("="*80)

all_entities = []
entity_type_counter = Counter()

for idx in range(len(df)):

    print(f"\n--- Review {idx} ---")

    review_text = df['review'].iloc[idx]
    entities = extract_entities_transformer(review_text)

    for entity in entities:

        text_clean = clean_entity_text(entity['word'])
        label = entity['entity_group']

        entity_type_counter[label] += 1

        all_entities.append({
            'review_id': idx,
            'text': text_clean,
            'label': label,
            'score': float(entity['score'])
        })

        description = entity_types.get(label, label)

        print(f"  Text: {text_clean:30} | Label: {label:10} | Type: {description:35} | Score: {entity['score']:.3f}")

# ==============================
# Entity Type Summary
# ==============================

print("\n" + "="*80)
print("ENTITY TYPES FOUND")
print("="*80)

for label, count in entity_type_counter.most_common():
    description = entity_types.get(label, label)
    print(f"{label:10} : {count:3} occurrences - {description}")

# ==============================
# Create DataFrame
# ==============================

entities_df = pd.DataFrame(all_entities)

if not entities_df.empty:

    # Remove duplicates
    entities_df = entities_df.sort_values('score', ascending=False)
    entities_df = entities_df.drop_duplicates(subset=['review_id', 'text', 'label'])

    print(f"\nTotal entities extracted: {len(entities_df)}")

    print("\nEntity type distribution:")
    print(entities_df['label'].value_counts())

    print("\nSample extracted entities:")
    print(entities_df[['text', 'label', 'score']].head(20))

    # Save CSV
    entities_df.to_csv('IMDB_NER_Results_Sample.csv', index=False)
    print("\n✓ Saved entities to: IMDB_NER_Results_Sample.csv")

else:
    print("\nNo entities found.")

print("\nDone!")
