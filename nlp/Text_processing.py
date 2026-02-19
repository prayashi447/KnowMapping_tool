import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

class IMDBPreprocessor:
    def __init__(self, remove_stopwords=True):
        self.lemmatizer = WordNetLemmatizer()
        self.remove_stopwords = remove_stopwords
        if remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
    
    def remove_html_tags(self, text):
        """Remove HTML tags like <br />"""
        clean_text = re.sub(r'<.*?>', ' ', text)
        return clean_text
    
    def remove_special_characters(self, text):
        """Remove special characters and keep only alphanumeric and spaces"""
        clean_text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        return clean_text
    
    def lowercase_text(self, text):
        """Convert text to lowercase"""
        return text.lower()
    
    def sentence_segmentation(self, text):
        """Split text into sentences"""
        sentences = sent_tokenize(text)
        return sentences
    
    def lemmatize_text(self, text):
        """Lemmatize words in the text"""
        words = text.split()
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)
    
    def remove_stopwords_from_text(self, text):
        """Remove stopwords from text"""
        if not self.remove_stopwords:
            return text
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def preprocess_text(self, text):
        """Apply all preprocessing steps to a single text"""
        # Step 1: Remove HTML tags
        text = self.remove_html_tags(text)
        
        # Step 2: Remove special characters
        text = self.remove_special_characters(text)
        
        # Step 3: Lowercase normalization
        text = self.lowercase_text(text)
        
        # Step 4: Sentence segmentation (store for reference)
        sentences = self.sentence_segmentation(text)
        
        # Step 5: Lemmatization
        text = self.lemmatize_text(text)
        
        # Step 6: Optional - Remove stopwords
        text = self.remove_stopwords_from_text(text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text, sentences
    
    def preprocess_dataframe(self, df, text_column='review'):
        """Preprocess entire dataframe"""
        processed_data = []
        all_sentences = []
        
        for idx, row in df.iterrows():
            processed_text, sentences = self.preprocess_text(row[text_column])
            processed_data.append({
                'original_review': row[text_column],
                'processed_review': processed_text,
                'sentiment': row['sentiment'],
                'num_sentences': len(sentences)
            })
            all_sentences.extend(sentences)
        
        processed_df = pd.DataFrame(processed_data)
        return processed_df, all_sentences

# Load the dataset
df = pd.read_csv('IMDB Dataset.csv')

# Display original data
print("Original Data Sample:")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
print("\n" + "="*80 + "\n")

# Create preprocessor instance
preprocessor = IMDBPreprocessor(remove_stopwords=True)

# Process a sample review first to show the steps
sample_review = df['review'].iloc[0]
print("Sample Review Processing:")
print(f"Original: {sample_review[:200]}...")
print("\nProcessing steps:")

# Step 1: Remove HTML tags
step1 = preprocessor.remove_html_tags(sample_review)
print(f"1. After HTML removal: {step1[:150]}...")

# Step 2: Remove special characters
step2 = preprocessor.remove_special_characters(step1)
print(f"2. After special char removal: {step2[:150]}...")

# Step 3: Lowercase
step3 = preprocessor.lowercase_text(step2)
print(f"3. After lowercase: {step3[:150]}...")

# Step 4: Sentence segmentation
step4_sentences = preprocessor.sentence_segmentation(step3)
print(f"4. Sentence segmentation: {len(step4_sentences)} sentences found")

# Step 5: Lemmatization
step5 = preprocessor.lemmatize_text(step3)
print(f"5. After lemmatization: {step5[:150]}...")

# Step 6: Remove stopwords
step6 = preprocessor.remove_stopwords_from_text(step5)
print(f"6. After stopword removal: {step6[:150]}...")

print("\n" + "="*80 + "\n")

# Process the entire dataset
print("Processing entire dataset...")
processed_df, all_sentences = preprocessor.preprocess_dataframe(df)

# Display results
print(f"Processed {len(processed_df)} reviews")
print(f"Total sentences extracted: {len(all_sentences)}")
print(f"Average sentences per review: {processed_df['num_sentences'].mean():.2f}")

print("\nProcessed Data Sample:")
print(processed_df[['processed_review', 'sentiment', 'num_sentences']].head())

# Save processed data
processed_df.to_csv('IMDB_Dataset_Processed.csv', index=False)

# Save all sentences to a text file
with open('all_sentences.txt', 'w', encoding='utf-8') as f:
    for sentence in all_sentences[:1000]:  # Save first 1000 sentences
        f.write(sentence + '\n')

print("\nFiles saved:")
print("- IMDB_Dataset_Processed.csv")
print("- all_sentences.txt")

# Optional: Show statistics
print("\n" + "="*80 + "\n")
print("Processing Statistics:")
print(f"Original reviews: {len(df)}")
print(f"Processed reviews: {len(processed_df)}")
print(f"Total sentences: {len(all_sentences)}")
print(f"Unique words in processed reviews: {len(set(' '.join(processed_df['processed_review']).split()))}")

# Compare original vs processed for first review
print("\nComparison Example (First Review):")
print(f"Original length: {len(df['review'].iloc[0])} characters")
print(f"Processed length: {len(processed_df['processed_review'].iloc[0])} characters")
print(f"Reduction: {(1 - len(processed_df['processed_review'].iloc[0])/len(df['review'].iloc[0]))*100:.2f}%")