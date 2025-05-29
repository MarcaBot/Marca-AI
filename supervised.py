import os
import json
import torch
import numpy as np
import glob
import pdfplumber
import Levenshtein
from typing import List, Dict, Any, Optional, Union, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    MBartForConditionalGeneration,
    MBartTokenizer,
    PreTrainedTokenizer
)
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

# Constants
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
MAX_LENGTH = 384
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_LANG = "en_XX"
NO_REPEAT_NGRAM_SIZE = 3
PDF_DATA_DIR = "/home/ubuntu/pdf_data" # Use absolute path in home dir
SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

class QADataset(Dataset):
    """Dataset for QA tasks from JSON files with support for diverse formats."""
    
    def __init__(self, data_dir: str, tokenizer: PreTrainedTokenizer, max_length: int = MAX_LENGTH):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Ensure data_dir exists before listing
        if not os.path.isdir(data_dir):
            print(f"Warning: JSON data directory {data_dir} not found.")
            return
            
        for filename in os.listdir(data_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(data_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self._process_data(data)
                except Exception as e:
                    print(f"Error loading JSON {file_path}: {e}")
    
    def _process_data(self, data: Any):
        # Processes different JSON structures (SQuAD-like, list of dicts, single dict)
        if isinstance(data, list) and data and isinstance(data[0], dict) and "paragraphs" in data[0]: # SQuAD-like
            for item in data:
                if "paragraphs" in item:
                    for paragraph in item["paragraphs"]:
                        context = paragraph.get("context", "")
                        if "qas" in paragraph:
                            for qa in paragraph["qas"]:
                                question = qa.get("question", "")
                                # Ensure answer exists and is not marked impossible
                                if not qa.get("is_impossible", False) and qa.get("answers", []):
                                    answer = qa["answers"][0].get("text", "")
                                    if question and answer: # Ensure both Q and A are non-empty
                                        self.data.append({
                                            "question": question,
                                            "answer": answer,
                                            "context": context,
                                            "evidence": [] # Placeholder for evidence if needed later
                                        })
        elif isinstance(data, list): # List of QA dicts
            for item in data:
                if isinstance(item, dict) and "question" in item:
                    question = item.get("question", "")
                    answer = item.get("answer", "")
                    evidence = item.get("evidence", [])
                    if question and answer:
                        self.data.append({
                            "question": question,
                            "answer": answer,
                            "context": "", # No context in this format
                            "evidence": evidence
                        })
        elif isinstance(data, dict) and "question" in data: # Single QA dict
            question = data.get("question", "")
            answer = data.get("answer", "")
            evidence = data.get("evidence", [])
            if question and answer:
                self.data.append({
                    "question": question,
                    "answer": answer,
                    "context": "",
                    "evidence": evidence
                })

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Prepares item for mBART training/evaluation
        item = self.data[idx]
        encoded_input = self.tokenizer(
            item["question"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        encoded_output = self.tokenizer(
            item["answer"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded_input["input_ids"].squeeze(0),
            "attention_mask": encoded_input["attention_mask"].squeeze(0),
            "labels": encoded_output["input_ids"].squeeze(0),
            "question": item["question"],
            "answer": item["answer"],
            "context": item["context"],
            "evidence": item["evidence"]
        }

def segment_text(text: str, max_len: int = 100) -> List[str]:
    """Segment long text into smaller chunks based on word count, preserving context."""
    words = text.split()
    segments = []
    current_segment = []
    current_len = 0
    
    for word in words:
        word_len = len(word)
        # Check if adding the next word exceeds max_len
        if current_len + word_len + (1 if current_segment else 0) <= max_len:
            current_segment.append(word)
            current_len += word_len + (1 if len(current_segment) > 1 else 0) # Add 1 for space after first word
        else:
            # Add the current segment if it's not empty
            if current_segment:
                segments.append(" ".join(current_segment))
            # Start a new segment with the current word
            current_segment = [word]
            current_len = word_len
            
    # Add the last segment if it exists
    if current_segment:
        segments.append(" ".join(current_segment))
    return segments

class SemanticSearcher:
    """Handles semantic search for QA pairs and Hybrid Scoring for PDF paragraphs."""
    
    def __init__(self, model_name: str = SENTENCE_TRANSFORMER_MODEL, 
                 mbart_tokenizer=None, mbart_model=None, dataset_lang: str = "en_XX"):
        print(f"Initializing Sentence Transformer: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            self.model.to(DEVICE)
        except Exception as e:
            print(f"Error loading embedding model {model_name}: {e}. Falling back to 'distiluse-base-multilingual-cased-v1'...")
            self.model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
            self.model.to(DEVICE)
        
        self.mbart_tokenizer = mbart_tokenizer
        self.mbart_model = mbart_model
        self.dataset_lang = dataset_lang # Primarily for translating QA questions if needed
        
        # Data storage
        self.qa_pairs = []
        self.pdf_paragraphs = []
        
        # Embeddings storage
        self.qa_embeddings = None
        self.pdf_embeddings = None
        
        # Utilities
        self.embedding_cache = {}  # Cache for computed embeddings to speed up repeated lookups
        self.word_dict = set()  # Dictionary for typo correction (built from QA questions)
    
    def add_qa_pairs(self, qa_pairs: List[Dict[str, str]]):
        """Add QA pairs, build word dictionary, and compute embeddings."""
        self.qa_pairs = qa_pairs
        self._build_word_dict()
        self._compute_qa_embeddings()

    def add_pdf_paragraphs(self, paragraphs: List[str]):
        """Add paragraphs extracted from PDF files and compute embeddings."""
        self.pdf_paragraphs = paragraphs
        self._compute_pdf_embeddings()
    
    def _build_word_dict(self):
        """Build a dictionary of unique words from QA questions for typo correction."""
        print("Building word dictionary from QA questions...")
        for qa in self.qa_pairs:
            # Simple whitespace split and lowercasing
            words = qa["question"].lower().split()
            self.word_dict.update(words)
        print(f"Built dictionary with {len(self.word_dict)} unique words.")
    
    def _translate_to_english(self, text: str, source_lang: str = "auto") -> str:
        """Translate text to English using mBART (if available). Used for QA search consistency."""
        # If mBART is not loaded, return original text
        if not self.mbart_model or not self.mbart_tokenizer:
            # print("mBART not available, skipping translation.")
            return text
            
        # Avoid translating if already English
        if source_lang == DEFAULT_LANG or source_lang == "en": 
             return text
             
        try:
            # Determine source language if 'auto'
            if source_lang == "auto":
                # Basic language detection could be added here if needed
                # For now, assume it's the primary dataset language if not specified
                source_lang = self.dataset_lang 
            
            # Check if the source language code is valid for the tokenizer
            if source_lang not in self.mbart_tokenizer.lang_code_to_id:
                print(f"Warning: Source language '{source_lang}' not supported by mBART tokenizer. Using default '{self.dataset_lang}'.")
                source_lang = self.dataset_lang

            # Set the source language for the tokenizer
            self.mbart_tokenizer.src_lang = source_lang
            
            # Prepare input for mBART
            inputs = self.mbart_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH
            ).to(DEVICE)
            
            # Generate translation
            self.mbart_model.eval() # Set model to evaluation mode
            with torch.no_grad(): # Disable gradient calculations for inference
                translated_ids = self.mbart_model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=MAX_LENGTH,
                    forced_bos_token_id=self.mbart_tokenizer.lang_code_to_id.get(DEFAULT_LANG), # Force English output
                    num_beams=4, # Beam search for better quality
                    length_penalty=1.0,
                    early_stopping=True
                )
            # Decode the generated IDs to text
            return self.mbart_tokenizer.decode(translated_ids[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error translating text '{text[:50]}...': {e}")
            return text # Return original text on error
    
    def _correct_typos(self, text: str) -> str:
        """Correct typos in text using Levenshtein distance against the QA word dictionary."""
        # Skip correction if no dictionary is built (e.g., only PDF data loaded)
        if not self.word_dict:
            return text
            
        words = text.lower().split()
        corrected_words = []
        for word in words:
            # Keep word if it's already in the dictionary
            if word in self.word_dict:
                corrected_words.append(word)
                continue
                
            # Find the closest match in the dictionary (within tolerance)
            min_dist = float('inf')
            best_match = word
            # Limit search space for efficiency if dict is very large (optional)
            # search_subset = random.sample(self.word_dict, min(len(self.word_dict), 5000)) if len(self.word_dict) > 5000 else self.word_dict
            search_subset = self.word_dict
            
            for dict_word in search_subset:
                dist = Levenshtein.distance(word, dict_word)
                # Update best match if distance is smaller and within tolerance (e.g., <= 2 edits)
                if dist < min_dist and dist <= 2: 
                    min_dist = dist
                    best_match = dict_word
                    
            corrected_words.append(best_match)
            
        return " ".join(corrected_words)
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using SentenceTransformer, utilizing cache."""
        # Return cached embedding if available
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        else:
            # Compute embedding, cache it, and return
            try:
                # Ensure text is not empty
                if not text.strip():
                     return np.zeros(self.model.get_sentence_embedding_dimension())
                embedding = self.model.encode(text, convert_to_numpy=True, device=DEVICE)
                self.embedding_cache[text] = embedding
                return embedding
            except Exception as e:
                print(f"Error encoding text \'{text[:50]}...\': {e}")
                # Return zero vector on error
                return np.zeros(self.model.get_sentence_embedding_dimension())

    def _compute_qa_embeddings(self):
        """Compute and store embeddings for all QA questions."""
        if not self.qa_pairs:
            print("Warning: No QA pairs loaded. Skipping QA embedding computation.")
            self.qa_embeddings = np.array([]) # Ensure it's an empty array
            return
        
        print(f"Computing embeddings for {len(self.qa_pairs)} QA questions...")
        questions = [qa_pair["question"] for qa_pair in self.qa_pairs]
        # Translate questions to English first for consistent embedding space (if mBART available)
        translated_questions = [self._translate_to_english(q, source_lang=self.dataset_lang) for q in questions]
        
        # Get embeddings for all translated questions
        self.qa_embeddings = np.array([self._get_embedding(q) for q in translated_questions])
        print(f"Finished computing QA embeddings. Shape: {self.qa_embeddings.shape}")

    def _compute_pdf_embeddings(self):
        """Compute and store embeddings for all PDF paragraphs."""
        if not self.pdf_paragraphs:
            print("Warning: No PDF paragraphs loaded. Skipping PDF embedding computation.")
            self.pdf_embeddings = np.array([]) # Ensure it's an empty array
            return
        
        print(f"Computing embeddings for {len(self.pdf_paragraphs)} PDF paragraphs...")
        # Assume PDF paragraphs are directly encodable by the SentenceTransformer model
        # No translation applied here by default
        self.pdf_embeddings = np.array([self._get_embedding(p) for p in self.pdf_paragraphs])
        print(f"Finished computing PDF embeddings. Shape: {self.pdf_embeddings.shape}")

    def _calculate_token_overlap(self, tokens1: List[str], tokens2: List[str]) -> float:
        """Calculate token overlap score using Jaccard similarity."""
        set1 = set(tokens1)
        set2 = set(tokens2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        # Avoid division by zero if both sets are empty
        return float(intersection) / union if union > 0 else 0.0

    def _calculate_levenshtein_similarity(self, s1: str, s2: str) -> float:
        """Calculate normalized Levenshtein similarity (1.0 is identical, 0.0 is max distance)."""
        distance = Levenshtein.distance(s1, s2)
        max_len = max(len(s1), len(s2))
        # Avoid division by zero if both strings are empty
        return 1.0 - (float(distance) / max_len) if max_len > 0 else 1.0

    def _compute_hybrid_score(self, query: str, paragraph: str, 
                              query_embedding: np.ndarray, paragraph_embedding: np.ndarray) -> float:
        """Compute the weighted hybrid score combining semantic, lexical, and edit distance similarity."""
        # Ensure inputs are valid
        if not query or not paragraph:
            return 0.0
            
        query_tokens = query.lower().split()
        paragraph_tokens = paragraph.lower().split()

        # --- Scoring Components --- 
        
        # 1. Semantic Similarity (Cosine): Measures contextual meaning similarity.
        #    Weight: 0.5 (Highest importance for understanding context)
        query_emb_r = query_embedding.reshape(1, -1)
        para_emb_r = paragraph_embedding.reshape(1, -1)
        # Handle potential zero vectors from embedding errors
        if np.all(query_emb_r == 0) or np.all(para_emb_r == 0):
            semantic_score = 0.0
        else:
            semantic_score = cosine_similarity(query_emb_r, para_emb_r)[0][0]
        # Clip score to be within [0, 1] as cosine can range from -1 to 1
        semantic_score = max(0.0, float(semantic_score)) # Ensure float type

        # 2. Levenshtein Similarity: Measures edit distance, good for minor variations/typos.
        #    Weight: 0.3 (Important for near-exact matches or slight differences)
        levenshtein_sim = self._calculate_levenshtein_similarity(query, paragraph)

        # 3. Token Overlap (Jaccard): Measures exact keyword overlap.
        #    Weight: 0.2 (Useful for keyword presence, less emphasis than context/structure)
        token_overlap_score = self._calculate_token_overlap(query_tokens, paragraph_tokens)

        # --- Combine Scores --- 
        hybrid_score = (0.5 * semantic_score) + (0.3 * levenshtein_sim) + (0.2 * token_overlap_score)
        
        # Ensure the final score is within [0, 1]
        return max(0.0, min(1.0, hybrid_score))

    def search_qa(self, query: str, top_k: int = 3, threshold: float = 0.6, source_lang: str = "auto") -> List[Dict[str, Any]]:
        """Search QA pairs using semantic similarity after typo correction and translation."""
        # Check if QA embeddings are available
        if self.qa_embeddings is None or len(self.qa_embeddings) == 0:
            print("No QA embeddings available for search.")
            return []
        
        try:
            # Preprocess query: Correct typos -> Translate to English
            corrected_query = self._correct_typos(query)
            # print(f"Corrected QA query: {corrected_query}") # Optional debug print
            translated_query = self._translate_to_english(corrected_query, source_lang)
            
            # Get embedding for the processed query
            query_embedding = self._get_embedding(translated_query)
            
            # Calculate cosine similarities between query embedding and all QA question embeddings
            similarities = cosine_similarity(query_embedding.reshape(1, -1), self.qa_embeddings)[0]
            
            # Get top-k results efficiently
            k = min(top_k, len(similarities))
            if k == 0: return [] # Handle case with no similarities
            # Use argpartition for efficiency: finds k largest elements without full sort
            top_indices_unsorted = np.argpartition(-similarities, k)[:k]
            # Sort only the top-k indices based on their similarities
            top_indices = top_indices_unsorted[np.argsort(-similarities[top_indices_unsorted])]

            # Filter results by threshold and format output
            results = []
            for idx in top_indices:
                similarity = float(similarities[idx]) # Ensure float
                if similarity >= threshold:
                    results.append({
                        "question": self.qa_pairs[idx]["question"],
                        "answer": self.qa_pairs[idx]["answer"],
                        "context": self.qa_pairs[idx].get("context", ""),
                        "evidence": self.qa_pairs[idx].get("evidence", []),
                        "similarity": similarity
                    })
            
            return results
        except Exception as e:
            print(f"Error during QA semantic search for query '{query[:50]}...': {e}")
            return [] # Return empty list on error

    def search_pdf(self, query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """Search PDF paragraphs using the calculated Hybrid Score."""
        # Check if PDF embeddings are available
        if self.pdf_embeddings is None or len(self.pdf_embeddings) == 0:
            print("No PDF embeddings available for search.")
            return []
        
        try:
            # Get embedding for the raw query (no typo correction/translation applied here)
            query_embedding = self._get_embedding(query)
            
            # Calculate hybrid score for the query against each PDF paragraph
            scores = []
            for i, paragraph in enumerate(self.pdf_paragraphs):
                paragraph_embedding = self.pdf_embeddings[i]
                hybrid_score = self._compute_hybrid_score(query, paragraph, query_embedding, paragraph_embedding)
                scores.append(hybrid_score)
            
            scores = np.array(scores)
            
            # Get top-k results based on hybrid score
            k = min(top_k, len(scores))
            if k == 0: return []
            top_indices_unsorted = np.argpartition(-scores, k)[:k]
            top_indices = top_indices_unsorted[np.argsort(-scores[top_indices_unsorted])]

            # Format output
            results = []
            for idx in top_indices:
                results.append({
                    "paragraph": self.pdf_paragraphs[idx],
                    "hybrid_score": float(scores[idx]) # Ensure float
                })
            
            return results
        except Exception as e:
            print(f"Error during PDF hybrid search for query '{query[:50]}...': {e}")
            return []

class SmartAssistant:
    """Main assistant class integrating QA, PDF search, and mBART generation."""
    
    def __init__(self, model_path: Optional[str] = None, data_dir: str = "data", pdf_data_dir: str = PDF_DATA_DIR, target_lang: str = DEFAULT_LANG):
        self.data_dir = data_dir
        self.pdf_data_dir = pdf_data_dir
        self.target_lang = target_lang
        print(f"Initializing SmartAssistant on device: {DEVICE}")
        
        # --- Load mBART Model (Optional, for translation and generation fallback) ---
        self.mbart_model = None
        self.mbart_tokenizer = None
        self.forced_bos_token_id = None
        try:
            # Attempt to load from specified path or default name
            load_path = model_path if (model_path and os.path.exists(model_path)) else MODEL_NAME
            print(f"Attempting to load mBART model/tokenizer from: {load_path}")
            # Check if it's the default model name to avoid redundant path check message
            if load_path == MODEL_NAME and not model_path:
                 print(f"Loading base mBART model: {MODEL_NAME}")
            elif model_path and os.path.exists(model_path):
                 print(f"Loading pre-trained mBART model from {model_path}")
            # else: # Case where model_path provided but doesn't exist - handled by from_pretrained error

            self.mbart_model = MBartForConditionalGeneration.from_pretrained(load_path)
            self.mbart_tokenizer = MBartTokenizer.from_pretrained(load_path)
            self.mbart_model = self.mbart_model.to(DEVICE)
            # Set the target language for generation
            if target_lang in self.mbart_tokenizer.lang_code_to_id:
                 self.forced_bos_token_id = self.mbart_tokenizer.lang_code_to_id.get(target_lang)
                 print(f"mBART Target language: {target_lang}, BOS token ID: {self.forced_bos_token_id}")
            else:
                 print(f"Warning: Target language '{target_lang}' not valid for mBART tokenizer. Using default.")
                 self.forced_bos_token_id = self.mbart_tokenizer.lang_code_to_id.get(DEFAULT_LANG)
        except ImportError as ie:
             print(f"ImportError loading mBART: {ie}. SentencePiece likely missing. Translation/generation features unavailable.")
        except Exception as e:
            print(f"Error loading mBART model from {load_path}: {e}. Translation/generation features might be limited.")
        
        # --- Initialize Semantic Searcher --- 
        print("Initializing SemanticSearcher...")
        # Pass loaded mBART components (or None) to the searcher
        self.searcher = SemanticSearcher(mbart_tokenizer=self.mbart_tokenizer, mbart_model=self.mbart_model, dataset_lang="en_XX") # Assuming QA data is English
        
        # --- Load Datasets --- 
        print(f"Loading datasets from JSON dir: {data_dir} and PDF dir: {pdf_data_dir}")
        self._load_datasets(data_dir, pdf_data_dir)
        print("SmartAssistant initialization complete.")
    
    def _load_datasets(self, json_data_dir: str, pdf_data_dir: str):
        """Load data from JSON files (for QA) and PDF files (for Hybrid Scoring)."""
        
        # --- 1. Load JSON QA Pairs --- 
        qa_pairs = []
        if os.path.exists(json_data_dir):
            files_loaded = 0
            print(f"Loading QA pairs from JSON files in {json_data_dir}...")
            for filename in os.listdir(json_data_dir):
                if filename.endswith(".json"):
                    file_path = os.path.join(json_data_dir, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            # Use QADataset's processing logic by creating a temporary instance
                            # Pass tokenizer only if available, QADataset handles None
                            temp_dataset = QADataset(json_data_dir, self.mbart_tokenizer)
                            temp_dataset.data = [] # Reset internal data
                            temp_dataset._process_data(data) # Process the loaded JSON data
                            qa_pairs.extend([item for item in temp_dataset.data]) # Extract processed items
                        files_loaded += 1
                    except Exception as e:
                        print(f"Error loading or processing JSON {file_path}: {e}")
            print(f"Loaded {len(qa_pairs)} QA pairs from {files_loaded} JSON files.")
            # Add loaded QA pairs to the searcher
            if qa_pairs:
                self.searcher.add_qa_pairs(qa_pairs)
        else:
            print(f"JSON data directory '{json_data_dir}' not found. Skipping JSON loading.")

        # --- 2. Load PDF Paragraphs --- 
        pdf_paragraphs = []
        if os.path.exists(pdf_data_dir):
            # Find PDF files matching the pattern 'unsup*.pdf'
            pdf_files = glob.glob(os.path.join(pdf_data_dir, "unsup*.pdf"))
            print(f"Found {len(pdf_files)} PDF files matching 'unsup*.pdf' in {pdf_data_dir}. Processing...")
            pdfs_processed = 0
            for pdf_path in pdf_files:
                try:
                    with pdfplumber.open(pdf_path) as pdf:
                        # print(f"Processing PDF: {os.path.basename(pdf_path)}") # Optional progress print
                        full_text = ""
                        for i, page in enumerate(pdf.pages):
                            # Extract text from the page, ignoring visual elements like images/tables
                            # x_tolerance helps merge words split across small gaps
                            # y_tolerance helps separate lines correctly
                            page_text = page.extract_text(x_tolerance=1, y_tolerance=3) 
                            if page_text:
                                full_text += page_text + "\n" # Add newline between pages for potential paragraph breaks
                        
                        # Split the full text into paragraphs based on newline characters
                        # Filter out empty strings resulting from multiple newlines
                        paragraphs = [p.strip() for p in full_text.split('\n') if p.strip()]
                        pdf_paragraphs.extend(paragraphs)
                        pdfs_processed += 1
                except Exception as e:
                    # Catch errors during PDF processing (e.g., corrupted file, password protected)
                    print(f"Error processing PDF {os.path.basename(pdf_path)}: {e}")
            print(f"Extracted {len(pdf_paragraphs)} paragraphs from {pdfs_processed} PDF files.")
            # Add extracted paragraphs to the searcher
            if pdf_paragraphs:
                self.searcher.add_pdf_paragraphs(pdf_paragraphs)
        else:
            print(f"PDF data directory '{pdf_data_dir}' not found. Skipping PDF loading.")

    def train(self, data_dir: str = None, output_dir: str = "model", epochs: int = EPOCHS):
        """Train the mBART model on the loaded QA dataset (JSON only)."""
        # Check if mBART model is available for training
        if self.mbart_model is None or self.mbart_tokenizer is None:
            print("mBART model/tokenizer not available. Skipping training.")
            return
        
        # Use default data directory if not specified
        if data_dir is None:
            data_dir = self.data_dir
        
        # Load dataset using QADataset class
        print(f"Loading training data from: {data_dir}")
        dataset = QADataset(data_dir, self.mbart_tokenizer)
        if len(dataset) == 0:
            print("No training data found in QADataset. Skipping training.")
            return
        
        print(f"Starting mBART training on {len(dataset)} examples for {epochs} epochs...")
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        optimizer = torch.optim.AdamW(self.mbart_model.parameters(), lr=LEARNING_RATE)
        
        self.mbart_model.train() # Set model to training mode
        for epoch in range(epochs):
            total_loss = 0
            batch_count = 0
            for batch in dataloader:
                try:
                    # Move batch data to the configured device (CPU or GPU)
                    input_ids = batch["input_ids"].to(DEVICE)
                    attention_mask = batch["attention_mask"].to(DEVICE)
                    labels = batch["labels"].to(DEVICE)
                    
                    optimizer.zero_grad() # Clear previous gradients
                    
                    # Forward pass
                    outputs = self.mbart_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels # Provide labels for loss calculation
                    )
                    loss = outputs.loss # Get the loss
                    total_loss += loss.item()
                    
                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()
                    
                    batch_count += 1
                    # Print progress periodically
                    if batch_count % 10 == 0:
                        print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_count}/{len(dataloader)}, Loss: {loss.item():.4f}")
                        
                except Exception as e:
                    print(f"Error during training batch {batch_count}: {e}")
                    # Consider skipping the batch or adding more specific error handling
                    continue 
                    
            avg_loss = total_loss / max(1, batch_count) # Avoid division by zero
            print(f"Epoch {epoch+1}/{epochs} finished. Average Loss: {avg_loss:.4f}")
        
        # --- Save the trained model --- 
        try:
            print(f"Saving trained model to {output_dir}...")
            os.makedirs(output_dir, exist_ok=True)
            self.mbart_model.save_pretrained(output_dir)
            self.mbart_tokenizer.save_pretrained(output_dir)
            print(f"Model and tokenizer saved successfully.")
        except Exception as e:
            print(f"Error saving model to {output_dir}: {e}")
    
    def answer(self, question: str, similarity_threshold: float = 0.7, source_lang: str = "auto", use_pdf: bool = True) -> str:
        """Provide an answer by searching PDF (Hybrid), then QA (Semantic), then generating (mBART)."""
        
        if not question or not question.strip():
            return "Please provide a question."
        
        print(f"\nProcessing question: '{question[:100]}...' (Source lang: {source_lang}, PDF search: {use_pdf})")
        
        # --- 1. PDF Hybrid Search --- 
        if use_pdf and self.searcher.pdf_paragraphs:
            print("Attempting PDF Hybrid Search...")
            pdf_results = self.searcher.search_pdf(question, top_k=1)
            if pdf_results:
                # User requested only the paragraph, no score threshold applied here by default
                # Can add threshold: if pdf_results[0]["hybrid_score"] > 0.5:
                print(f"  Found relevant paragraph in PDF (Score: {pdf_results[0]['hybrid_score']:.4f})")
                return pdf_results[0]["paragraph"]
            else:
                print("  No relevant paragraph found in PDFs.")
        elif use_pdf:
             print("PDF search enabled, but no PDF paragraphs loaded.")

        # --- 2. QA Semantic Search (Fallback) --- 
        if self.searcher.qa_pairs:
            print("Attempting QA Semantic Search...")
            qa_search_results = self.searcher.search_qa(question, top_k=1, threshold=similarity_threshold, source_lang=source_lang)
            if qa_search_results: # Already filtered by threshold in search_qa
                result = qa_search_results[0]
                answer = result["answer"]
                evidence = result.get("evidence", []) # Use .get for safety
                print(f"  Found relevant QA pair (Similarity: {result['similarity']:.4f})")
                # Format answer with evidence if available
                if evidence:
                    evidence_texts = [e.get("text", "") for e in evidence]
                    sources = [e.get("source", "") for e in evidence]
                    # Filter out empty evidence/sources before joining
                    valid_evidence = [f"reason: {text}\n{source}" for text, source in zip(evidence_texts, sources) if text and source]
                    if valid_evidence:
                         evidence_output = "\n".join(valid_evidence)
                         return f"{answer}\n{evidence_output}"
                return answer # Return answer without evidence if none provided
            else:
                print(f"  No relevant QA pair found above threshold {similarity_threshold}.")
        else:
             print("No QA pairs loaded for search.")

        # --- 3. mBART Generation (Final Fallback) --- 
        print("Falling back to mBART generation...")
        return self._generate_mbart_answer(question, source_lang=source_lang)
    
    def _generate_mbart_answer(self, question: str, source_lang: str = "auto") -> str:
        """Generate an answer using the mBART model (if available)."""
        # Check if mBART is loaded
        if self.mbart_model is None or self.mbart_tokenizer is None:
            return "I'm sorry, the generation model is not available to answer that question."
        
        try:
            # Prepare input for generation
            # Set source language for tokenizer if needed (mBART needs src_lang)
            if source_lang == "auto":
                 source_lang = self.target_lang # Assume question is in target lang if auto
            if source_lang not in self.mbart_tokenizer.lang_code_to_id:
                 print(f"Warning: Source language '{source_lang}' invalid for mBART generation. Using target '{self.target_lang}'.")
                 source_lang = self.target_lang
            self.mbart_tokenizer.src_lang = source_lang
            
            inputs = self.mbart_tokenizer(
                question, # Use the original question for generation
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH
            ).to(DEVICE)
            
            self.mbart_model.eval() # Set to evaluation mode
            with torch.no_grad(): # Disable gradients
                output_ids = self.mbart_model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=MAX_LENGTH, # Max length of the generated answer
                    forced_bos_token_id=self.forced_bos_token_id, # Force output in target language
                    no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE, # Prevent repetitive phrases
                    num_beams=5, # Use beam search for better quality generation
                    length_penalty=1.0,
                    early_stopping=True
                )
            # Decode the generated IDs to text
            output = self.mbart_tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return output if output else "mBART generated an empty response."
            
        except Exception as e:
            print(f"Error during mBART generation for question '{question[:50]}...': {e}")
            return "I'm sorry, an error occurred while trying to generate an answer."
    
    def set_target_language(self, lang_code: str):
        """Set the target language for mBART generation."""
        # Check if mBART tokenizer is loaded and supports the language code
        if self.mbart_tokenizer and lang_code in self.mbart_tokenizer.lang_code_to_id:
            self.target_lang = lang_code
            self.forced_bos_token_id = self.mbart_tokenizer.lang_code_to_id[lang_code]
            print(f"Target language for generation updated to: {lang_code}")
            # Note: This does NOT change the assumed language of the QA dataset (dataset_lang in searcher)
            return True
        elif not self.mbart_tokenizer:
             print("Cannot set target language: mBART tokenizer not loaded.")
             return False
        else:
            print(f"Invalid or unsupported language code for mBART: '{lang_code}'. Keeping current: {self.target_lang}")
            return False
    
    def save_research(self, data: str, filename: str = "research_output.txt") -> str:
        """Append research data (Q/A pair) to a file."""
        try:
            output_dir = "research"
            # Create the directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, filename)
            # Append to the file
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(data + "\n" + "-"*20 + "\n") # Add separator
            return f"Research saved successfully to {file_path}"
        except Exception as e:
            return f"Error saving research to {filename}: {e}"

def main():
    print("Initializing Marca Smart Assistant...")
    # Initialize with both JSON and PDF data directories
    # Assumes 'data' for JSON and '/home/ubuntu/pdf_data' for PDFs
    assistant = SmartAssistant(data_dir="data", pdf_data_dir=PDF_DATA_DIR)
    
    print("\n===================================")
    print("Marca Assistant (Type 'exit' to quit)")
    print("Commands: /lang <lang_code>")
    print("===================================\n")
    
    while True:
        try:
            user_input = input("Question: ").strip()
            
            if not user_input:
                continue # Ask again if input is empty
                
            if user_input.lower() == 'exit':
                print("Thank you for using Marca Assistant. Goodbye!")
                break
            
            # Handle language setting command
            if user_input.startswith("/lang "):
                parts = user_input.split(" ", 1)
                if len(parts) == 2:
                    lang_code = parts[1].strip()
                    if assistant.set_target_language(lang_code):
                         print(f"Generation language set to {lang_code}.")
                    else:
                         print(f"Failed to set language to {lang_code}.")
                else:
                    print("Usage: /lang <language_code> (e.g., /lang es_XX)")
                continue # Ask for next question
            
            # --- Process normal question --- 
            question_text = user_input
            source_lang = "auto" # Default source language
            
            # Optional: Check for language prefix like "fr_FR: Quelle heure est-il?"
            if ":" in user_input:
                parts = user_input.split(":", 1)
                lang_part = parts[0].strip()
                # Check if it's a valid language code known to the tokenizer
                if assistant.mbart_tokenizer and lang_part in assistant.mbart_tokenizer.lang_code_to_id:
                    source_lang = lang_part
                    question_text = parts[1].strip()
                    print(f"(Detected source language: {source_lang})" ) 
                # else: treat the colon as part of the question
            
            # Get the answer using the assistant's logic (PDF -> QA -> mBART)
            answer = assistant.answer(question_text, source_lang=source_lang, use_pdf=True)
            print(f"\nMarca: {answer}\n")
            
            # --- Save research option --- 
            # save_option = input("Do you want to save this research? (y/n): ").lower()
            # if save_option == 'y':
            #     research_data = f"Question: {user_input}\nAnswer: {answer}"
            #     save_result = assistant.save_research(research_data)
            #     print(save_result)
            # print("") # Add a blank line for spacing
                
        except KeyboardInterrupt:
            print("\nExiting Marca Assistant. Goodbye!")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            # Log the full traceback for debugging if needed
            # import traceback
            # traceback.print_exc()
            print("Please try again or ask a different question.")

if __name__ == "__main__":
    # --- Setup: Ensure data directories exist --- 
    # Ensure JSON data directory exists
    if not os.path.exists("data"):
        print("Creating JSON data directory: data")
        os.makedirs("data")
        # Optional: Create a dummy JSON file if needed for testing
        # dummy_qa = {'question': 'What is the capital of France?', 'answer': 'Paris'}
        # with open('data/dummy_qa.json', 'w') as f: json.dump(dummy_qa, f)
        
    # Ensure PDF data directory exists
    if not os.path.exists(PDF_DATA_DIR):
        print(f"Creating PDF data directory: {PDF_DATA_DIR}")
        os.makedirs(PDF_DATA_DIR)
        
    # --- Optional: Create a dummy PDF for testing if PDF dir was just created --- 
    dummy_pdf_path = os.path.join(PDF_DATA_DIR, "unsup1.pdf")
    if not os.path.exists(dummy_pdf_path):
        print(f"Creating dummy PDF: {dummy_pdf_path}")
        try:
            # Use reportlab to create a simple PDF
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            c = canvas.Canvas(dummy_pdf_path, pagesize=letter)
            textobject = c.beginText(50, 750) # Position text
            textobject.setFont('Helvetica', 12) # Set font
            # Add paragraphs separated by empty lines
            textobject.textLine("This is the first paragraph of the dummy PDF document.")
            textobject.textLine("It contains some sample text specifically for testing the hybrid scoring mechanism.")
            textobject.textLine("") 
            textobject.textLine("This constitutes the second paragraph.")
            textobject.textLine("The system should compare user input against these text segments.")
            textobject.textLine("Another sentence in the second paragraph.")
            textobject.textLine("")
            textobject.textLine("Finally, the third paragraph provides more context.")
            c.drawText(textobject)
            c.save() # Save the PDF file
            print(f"Successfully created dummy PDF.")
        except ImportError:
            # Handle case where reportlab is not installed
            print("Skipping dummy PDF creation: reportlab library not found. Please install it (`pip install reportlab`) if needed.")
        except Exception as e:
            # Catch other potential errors during PDF creation
            print(f"Error creating dummy PDF: {e}")

    # --- Run the main application loop --- 
    main()