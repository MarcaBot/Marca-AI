import os
import json
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from transformers import (
    MBartForConditionalGeneration,
    MBartTokenizer,
    PreTrainedTokenizer
)
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import pdfplumber
from multiprocessing import Pool
import logging
import re # Added for paragraph splitting

# Setup logging untuk edukasi
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
MAX_LENGTH = 384
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_LANG = "en_XX"
NO_REPEAT_NGRAM_SIZE = 3
MIN_PARAGRAPH_WORDS = 10 # Minimum words for a valid paragraph
DEBUG_SIMILARITY_THRESHOLD = 0.2 # Lower threshold for debugging

class QADataset(Dataset):
    """Dataset for QA tasks from JSON files with support for diverse formats."""

    def __init__(self, data_dir: str, tokenizer: PreTrainedTokenizer, max_length: int = MAX_LENGTH):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for filename in os.listdir(data_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(data_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self._process_data(data)
                except Exception as e:
                    logging.error(f"Error loading {file_path}: {e}")

    def _process_data(self, data: Any):
        # Existing JSON processing logic remains the same
        if isinstance(data, list) and data and "paragraphs" in data[0]:
            for item in data:
                if "paragraphs" in item:
                    for paragraph in item["paragraphs"]:
                        context = paragraph.get("context", "")
                        if "qas" in paragraph:
                            for qa in paragraph["qas"]:
                                question = qa.get("question", "")
                                if not qa.get("is_impossible", True) and qa.get("answers", []):
                                    answer = qa["answers"][0].get("text", "")
                                    self.data.append({
                                        "question": question,
                                        "answer": answer,
                                        "context": context,
                                        "evidence": []
                                    })
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "question" in item:
                    question = item.get("question", "")
                    answer = item.get("answer", "")
                    evidence = item.get("evidence", [])
                    if question and answer:
                        self.data.append({
                            "question": question,
                            "answer": answer,
                            "context": "",
                            "evidence": evidence
                        })
        elif isinstance(data, dict) and "question" in data: # Handle single dict case
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

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
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

def segment_text(text: str, max_len: int = 100) -> List[str]:
    """Segment long text into smaller chunks while preserving context."""
    words = text.split()
    segments = []
    current_segment = []
    current_len = 0

    for word in words:
        if current_len + len(word) + 1 <= max_len:
            current_segment.append(word)
            current_len += len(word) + 1
        else:
            segments.append(" ".join(current_segment))
            current_segment = [word]
            current_len = len(word) + 1
    if current_segment:
        segments.append(" ".join(current_segment))
    return segments

class SemanticSearcher:
    """Enhanced semantic searcher with typo correction and long input handling."""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 mbart_tokenizer=None, mbart_model=None, dataset_lang: str = "en_XX"):
        try:
            self.model = SentenceTransformer(model_name)
            self.model.to(DEVICE)
            logging.info(f"Loaded SentenceTransformer: {model_name}")
        except Exception as e:
            logging.error(f"Error loading embedding model: {e}")
            # Use a different fallback if the primary fails
            self.model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
            self.model.to(DEVICE)

        self.mbart_tokenizer = mbart_tokenizer
        self.mbart_model = mbart_model
        self.dataset_lang = dataset_lang
        self.qa_pairs = []
        self.embeddings = None
        self.embedding_cache = {}
        self.word_dict = set()

    def add_qa_pairs(self, qa_pairs: List[Dict[str, Any]]):
        """Add QA pairs and build word dictionary for typo correction."""
        self.qa_pairs = qa_pairs
        self._build_word_dict()
        self._compute_embeddings()

    def _build_word_dict(self):
        """Build a dictionary of words from questions for typo correction."""
        for qa in self.qa_pairs:
            words = qa["question"].lower().split()
            self.word_dict.update(words)

    def _translate_to_english(self, text: str, source_lang: str = "auto") -> str:
        if not self.mbart_model or not self.mbart_tokenizer:
            logging.warning("mBART model/tokenizer not available for translation.")
            return text
        try:
            # Ensure source language is set correctly
            if source_lang != "auto" and source_lang in self.mbart_tokenizer.lang_code_to_id:
                 self.mbart_tokenizer.src_lang = source_lang
            else:
                 # Attempt to detect or default
                 self.mbart_tokenizer.src_lang = self.dataset_lang # Use dataset lang as default source

            inputs = self.mbart_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH
            ).to(DEVICE)
            self.mbart_model.eval()
            with torch.no_grad():
                translated_ids = self.mbart_model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=MAX_LENGTH,
                    forced_bos_token_id=self.mbart_tokenizer.lang_code_to_id.get("en_XX"), # Always translate to English for embeddings
                    num_beams=4,
                    length_penalty=1.0
                )
            return self.mbart_tokenizer.decode(translated_ids[0], skip_special_tokens=True)
        except Exception as e:
            logging.error(f"Error translating text from {self.mbart_tokenizer.src_lang}: {e}")
            return text

    def _correct_typos(self, text: str) -> str:
        """Correct typos in text using Levenshtein distance."""
        words = text.lower().split()
        corrected_words = []
        for word in words:
            if word in self.word_dict:
                corrected_words.append(word)
                continue
            min_dist = float('inf')
            best_match = word
            # Limit dictionary search for performance if needed
            for dict_word in self.word_dict:
                dist = levenshtein_distance(word, dict_word)
                if dist < min_dist and dist <= 2: # Allow up to 2 edits
                    min_dist = dist
                    best_match = dict_word
            corrected_words.append(best_match)
        return " ".join(corrected_words)

    def _compute_embeddings(self):
        """Compute embeddings with caching for efficiency."""
        if not self.qa_pairs:
            logging.warning("No QA pairs to compute embeddings for.")
            return

        questions = [qa_pair["question"] for qa_pair in self.qa_pairs]
        # Translate questions to English before embedding
        logging.info(f"Translating {len(questions)} questions from {self.dataset_lang} to English for embedding...")
        translated_questions = [self._translate_to_english(q, source_lang=self.dataset_lang) for q in questions]

        self.embeddings = []
        successful_encodings = 0
        for i, question in enumerate(translated_questions):
            if not question: # Skip empty translations
                logging.warning(f"Skipping empty translated question for original: {questions[i]}")
                # Add a zero vector or handle appropriately
                self.embeddings.append(np.zeros(self.model.get_sentence_embedding_dimension()))
                continue

            if question in self.embedding_cache:
                self.embeddings.append(self.embedding_cache[question])
                successful_encodings += 1
            else:
                try:
                    embedding = self.model.encode(question, convert_to_numpy=True)
                    self.embedding_cache[question] = embedding
                    self.embeddings.append(embedding)
                    successful_encodings += 1
                except Exception as e:
                    logging.error(f"Error encoding question '{question}': {e}")
                    self.embeddings.append(np.zeros(self.model.get_sentence_embedding_dimension()))

        self.embeddings = np.array(self.embeddings)
        logging.info(f"Computed embeddings for {successful_encodings}/{len(translated_questions)} questions")

    def _combine_scores(self, query_embedding: np.ndarray, question_embeddings: np.ndarray,
                       query_tokens: List[str], question_tokens: List[List[str]]) -> np.ndarray:
        """Combine semantic and token-based similarity scores."""
        if question_embeddings.size == 0 or query_embedding.size == 0:
            return np.array([])
        # Ensure embeddings have compatible dimensions
        if question_embeddings.ndim == 1: question_embeddings = question_embeddings.reshape(1, -1)
        if query_embedding.ndim == 1: query_embedding = query_embedding.reshape(-1) # Keep query 1D for dot product
        if question_embeddings.shape[1] != query_embedding.shape[0]:
             logging.error(f"Embedding shape mismatch: Questions {question_embeddings.shape} vs Query {query_embedding.shape}")
             return np.array([])

        # Calculate cosine similarity
        norm_q = np.linalg.norm(query_embedding)
        norm_qs = np.linalg.norm(question_embeddings, axis=1)
        # Avoid division by zero
        valid_indices = (norm_q > 1e-8) & (norm_qs > 1e-8)
        cosine_scores = np.zeros(question_embeddings.shape[0])
        if np.any(valid_indices):
             q_emb_norm = query_embedding / norm_q if norm_q > 1e-8 else query_embedding
             qs_emb_norm = question_embeddings[valid_indices] / norm_qs[valid_indices, np.newaxis]
             cosine_scores[valid_indices] = np.dot(qs_emb_norm, q_emb_norm)

        # Calculate token overlap score
        token_scores = []
        query_token_set = set(query_tokens)
        for q_tokens in question_tokens:
            q_token_set = set(q_tokens)
            denominator = max(len(q_token_set), 1)
            common_tokens = len(query_token_set & q_token_set) / denominator
            token_scores.append(common_tokens)

        cosine_scores = np.nan_to_num(cosine_scores)
        token_scores_np = np.array(token_scores)

        if cosine_scores.shape != token_scores_np.shape:
             logging.warning(f"Score shape mismatch: Cosine {cosine_scores.shape}, Token {token_scores_np.shape}. Using only cosine.")
             return cosine_scores

        # Weighted combination
        return 0.7 * cosine_scores + 0.3 * token_scores_np

    def search(self, query: str, top_k: int = 3, threshold: float = 0.6, source_lang: str = "auto") -> List[Dict[str, Any]]:
        """Enhanced semantic search with typo correction and long input handling."""
        if self.embeddings is None or len(self.embeddings) == 0:
            logging.warning("No embeddings available for search.")
            return []

        try:
            corrected_query = self._correct_typos(query)
            logging.info(f"Corrected query: {corrected_query}")

            query_segments = segment_text(corrected_query, max_len=100)
            # Translate query segments to English for embedding comparison
            translated_segments = [self._translate_to_english(seg, source_lang=source_lang) for seg in query_segments]

            segment_embeddings = []
            for seg in translated_segments:
                if not seg: continue # Skip empty translations
                if seg in self.embedding_cache:
                    segment_embeddings.append(self.embedding_cache[seg])
                else:
                    try:
                        embedding = self.model.encode(seg, convert_to_numpy=True)
                        self.embedding_cache[seg] = embedding
                        segment_embeddings.append(embedding)
                    except Exception as e:
                         logging.error(f"Error encoding query segment '{seg}': {e}")

            if not segment_embeddings:
                 logging.warning("Could not generate embeddings for query segments.")
                 return []

            query_embedding = np.mean(segment_embeddings, axis=0)
            query_tokens = corrected_query.lower().split()
            question_tokens = [qa["question"].lower().split() for qa in self.qa_pairs]

            similarities = self._combine_scores(query_embedding, self.embeddings, query_tokens, question_tokens)

            if similarities.size == 0:
                 logging.warning("No similarity scores computed.")
                 return []

            actual_top_k = min(top_k, len(similarities))
            # Get indices sorted by similarity DESCENDING
            top_indices = np.argsort(-similarities)[:actual_top_k]

            results = []
            logging.info(f"Top similarities: {[similarities[i] for i in top_indices]}") # Log top scores
            for idx in top_indices:
                similarity = similarities[idx]
                # Ensure similarity is a standard float, not numpy float
                similarity_float = float(similarity)
                if similarity_float >= threshold:
                    results.append({
                        "question": self.qa_pairs[idx]["question"],
                        "answer": self.qa_pairs[idx]["answer"],
                        "context": self.qa_pairs[idx].get("context", ""),
                        "evidence": self.qa_pairs[idx].get("evidence", []),
                        "similarity": similarity_float
                    })

            return results
        except Exception as e:
            logging.error(f"Error during semantic search: {e}")
            return []

class SmartAssistant:
    """Enhanced smart assistant with robust context handling and typo correction."""

    def __init__(self, model_path: Optional[str] = None, data_dir: str = "data", target_lang: str = DEFAULT_LANG):
        self.data_dir = data_dir
        self.target_lang = target_lang
        self.model = None # Initialize model as None
        self.tokenizer = None # Initialize tokenizer as None
        logging.info(f"Initializing SmartAssistant on device: {DEVICE}")

        try:
            if model_path and os.path.exists(model_path):
                logging.info(f"Loading pre-trained model from {model_path}")
                self.model = MBartForConditionalGeneration.from_pretrained(model_path)
                self.tokenizer = MBartTokenizer.from_pretrained(model_path)
            else:
                logging.info(f"Loading base mBART model: {MODEL_NAME}")
                self.model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
                self.tokenizer = MBartTokenizer.from_pretrained(MODEL_NAME)

            self.model = self.model.to(DEVICE)
            # Set target language ID for mBART generation
            self.forced_bos_token_id = self.tokenizer.lang_code_to_id.get(target_lang)
            if self.forced_bos_token_id is None:
                 logging.warning(f"Target language {target_lang} not supported by mBART. Defaulting generation language to en_XX.")
                 self.target_lang = "en_XX" # Update target lang if fallback
                 self.forced_bos_token_id = self.tokenizer.lang_code_to_id.get("en_XX")

            logging.info(f"Target language for generation: {self.target_lang}, BOS token ID: {self.forced_bos_token_id}")
            logging.info("mBART model and tokenizer loaded successfully.") # Added confirmation log

        except Exception as e:
            logging.error(f"Error loading mBART model: {e}")
            self.model = None # Ensure model is None on error
            self.tokenizer = None

        logging.info("Initializing semantic searcher...")
        # Pass potentially None model/tokenizer to searcher
        self.searcher = SemanticSearcher(mbart_tokenizer=self.tokenizer, mbart_model=self.model, dataset_lang="id_ID")
        logging.info(f"Loading dataset from {data_dir}")
        self._load_dataset(data_dir)
        logging.info("SmartAssistant initialization complete")

    def _extract_paragraphs_from_pdf(self, pdf_path: str) -> List[str]:
        """Extracts text from a PDF and splits it into more granular paragraphs."""
        paragraphs = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                full_text = ""
                for page in pdf.pages:
                    # Use extract_text with layout preservation if possible, or adjust tolerances
                    page_text = page.extract_text(x_tolerance=2, y_tolerance=3, layout=False) # Try layout=False
                    if page_text:
                        cleaned_page_text = page_text.replace("\f", "\n").replace("\r\n", "\n").replace("\r", "\n")
                        full_text += cleaned_page_text + "\n"

            # Split into lines first
            lines = full_text.split("\n")
            current_paragraph = []

            for line in lines:
                stripped_line = line.strip()

                # Skip empty lines and potential page numbers
                if not stripped_line or (stripped_line.isdigit() and len(stripped_line) < 3):
                    if current_paragraph:
                        para_text = " ".join(current_paragraph).strip()
                        # Check word count and avoid adding just headings as paragraphs
                        is_likely_heading = (
                            re.match(r"^BAB\s+[IVXLCDM]+", para_text) or
                            re.match(r"^[A-Z]\.\s+", para_text) or
                            re.match(r"^\d+\.\s+", para_text) or
                            re.match(r"^APA ITU[\.\s]*$", para_text, re.IGNORECASE) or
                            (len(para_text.split()) < 5 and para_text.isupper())
                        )
                        if len(para_text.split()) >= MIN_PARAGRAPH_WORDS and not is_likely_heading:
                            paragraphs.append(para_text)
                        elif is_likely_heading and len(para_text.split()) < MIN_PARAGRAPH_WORDS:
                             logging.debug(f"Skipping likely heading as paragraph: {para_text}")
                        current_paragraph = []
                    continue

                # Check for potential headings/subheadings to force a paragraph break
                is_heading = (
                    re.match(r"^BAB\s+[IVXLCDM]+", stripped_line) or
                    re.match(r"^[A-Z]\.\s+", stripped_line) or
                    re.match(r"^\d+\.\s+", stripped_line) or
                    re.match(r"^APA ITU[\.\s]*$", stripped_line, re.IGNORECASE) or
                    (len(stripped_line.split()) < 5 and stripped_line.isupper() and not stripped_line.endswith(".")) # Refined heading check
                )

                if is_heading and current_paragraph:
                    para_text = " ".join(current_paragraph).strip()
                    if len(para_text.split()) >= MIN_PARAGRAPH_WORDS:
                         paragraphs.append(para_text)
                    # Start new paragraph *with* the heading line
                    current_paragraph = [stripped_line]
                else:
                    current_paragraph.append(stripped_line)

            # Add the last paragraph if it meets criteria
            if current_paragraph:
                para_text = " ".join(current_paragraph).strip()
                is_likely_heading = (
                    re.match(r"^BAB\s+[IVXLCDM]+", para_text) or
                    re.match(r"^[A-Z]\.\s+", para_text) or
                    re.match(r"^\d+\.\s+", para_text) or
                    re.match(r"^APA ITU[\.\s]*$", para_text, re.IGNORECASE) or
                    (len(para_text.split()) < 5 and para_text.isupper())
                )
                if len(para_text.split()) >= MIN_PARAGRAPH_WORDS and not is_likely_heading:
                    paragraphs.append(para_text)

            logging.info(f"Extracted {len(paragraphs)} paragraphs from {os.path.basename(pdf_path)} using refined segmentation.")
        except Exception as e:
            logging.error(f"Error extracting text from {pdf_path}: {e}")
        return paragraphs

    def _load_dataset(self, data_dir: str):
        qa_pairs = []
        if not os.path.exists(data_dir):
            logging.info(f"Data directory {data_dir} not found. Creating it.")
            os.makedirs(data_dir, exist_ok=True)
            return

        files_loaded = 0
        cache_file = os.path.join(data_dir, "pdf_paragraphs_refined.json") # Use a new cache file name
        pdf_files_to_process = [f for f in os.listdir(data_dir) if f.startswith("unsup") and f.endswith(".pdf")]

        # Load cached PDF paragraphs if cache exists
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    qa_pairs.extend(json.load(f))
                files_loaded += len(pdf_files_to_process) # Assume all PDFs were cached
                logging.info(f"Loaded {len(qa_pairs)} QA pairs from refined cache: {cache_file}")
            except Exception as e:
                logging.error(f"Error loading refined cache: {e}. Will re-process PDFs.")
                qa_pairs = []
                if os.path.exists(cache_file):
                     os.remove(cache_file)

        # Process JSON files (existing logic)
        for filename in os.listdir(data_dir):
            if filename.endswith(".json") and filename != os.path.basename(cache_file):
                file_path = os.path.join(data_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # --- Existing JSON processing logic --- 
                        if isinstance(data, list) and data and "paragraphs" in data[0]:
                            for item in data:
                                if "paragraphs" in item:
                                    for paragraph in item["paragraphs"]:
                                        context = paragraph.get("context", "")
                                        if "qas" in paragraph:
                                            for qa in paragraph["qas"]:
                                                question = qa.get("question", "")
                                                if not qa.get("is_impossible", False) and qa.get("answers", []):
                                                    answer = qa["answers"][0].get("text", "")
                                                    qa_pairs.append({
                                                        "question": question,
                                                        "answer": answer,
                                                        "context": context,
                                                        "evidence": []
                                                    })
                        elif isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and "question" in item:
                                    question = item.get("question", "")
                                    answer = item.get("answer", "")
                                    evidence = item.get("evidence", [])
                                    if question and answer:
                                        qa_pairs.append({
                                            "question": question,
                                            "answer": answer,
                                            "context": "",
                                            "evidence": evidence
                                        })
                        elif isinstance(data, dict) and "question" in data:
                            question = data.get("question", "")
                            answer = data.get("answer", "")
                            evidence = data.get("evidence", [])
                            if question and answer:
                                qa_pairs.append({
                                    "question": question,
                                    "answer": answer,
                                    "context": "",
                                    "evidence": evidence
                                })
                        # --- End of existing JSON processing logic --- 
                    files_loaded += 1
                    logging.info(f"Loaded JSON file: {filename}")
                except Exception as e:
                    logging.error(f"Error loading {file_path}: {e}")

        # Process PDF files using refined text extraction if cache wasn't loaded or needs update
        if not os.path.exists(cache_file):
            pdf_qa_pairs = []
            if pdf_files_to_process:
                logging.info(f"Processing {len(pdf_files_to_process)} PDF files using refined text extraction...")
                for i, filename in enumerate(pdf_files_to_process):
                    pdf_path = os.path.join(data_dir, filename)
                    paragraphs = self._extract_paragraphs_from_pdf(pdf_path)
                    for para in paragraphs:
                        pdf_qa_pairs.append({
                            "question": para, # Use paragraph as question (for retrieval)
                            "answer": para,   # Use paragraph as answer
                            "context": f"{filename}",
                            "evidence": []
                        })
                    logging.info(f"Completed {i+1}/{len(pdf_files_to_process)} PDFs: {filename}")

                # Save extracted paragraphs to the new cache file
                try:
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(pdf_qa_pairs, f, ensure_ascii=False, indent=2)
                    logging.info(f"Saved {len(pdf_qa_pairs)} QA pairs from PDFs to refined cache: {cache_file}")
                    qa_pairs.extend(pdf_qa_pairs)
                    files_loaded += len(pdf_files_to_process)
                except Exception as e:
                     logging.error(f"Error saving PDF QA pairs to refined cache: {e}")
            else:
                logging.info("No PDF files found to process")

        logging.info(f"Loaded a total of {len(qa_pairs)} QA pairs from {files_loaded} files")
        if qa_pairs:
            self.searcher.dataset_lang = "id_ID" # Confirm dataset language for searcher
            self.searcher.add_qa_pairs(qa_pairs)
        else:
             logging.warning("No QA pairs loaded. Search functionality will be limited.")

    def train(self, data_dir: str = None, output_dir: str = "model", epochs: int = EPOCHS):
        # Training logic remains the same
        if self.model is None or self.tokenizer is None:
            logging.error("mBART model not available for training.")
            return

        if data_dir is None:
            data_dir = self.data_dir

        # Note: QADataset currently only loads from JSON.
        # If training on PDF paragraphs is needed, QADataset needs modification.
        dataset = QADataset(data_dir, self.tokenizer)
        if len(dataset) == 0:
            logging.warning("No JSON training data found in QADataset. Skipping training.")
            return

        logging.info(f"Training on {len(dataset)} examples for {epochs} epochs")
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            batch_count = 0
            for batch in dataloader:
                try:
                    input_ids = batch["input_ids"].to(DEVICE)
                    attention_mask = batch["attention_mask"].to(DEVICE)
                    labels = batch["labels"].to(DEVICE)
                    optimizer.zero_grad()
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    batch_count += 1
                    if batch_count % 10 == 0:
                        logging.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_count}/{len(dataloader)}, Loss: {loss.item():.4f}")
                except Exception as e:
                    logging.error(f"Error in training batch: {e}")
                    continue
            avg_loss = total_loss / max(1, batch_count)
            logging.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

        try:
            os.makedirs(output_dir, exist_ok=True)
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            logging.info(f"Model saved to {output_dir}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")

    def answer(self, question: str, similarity_threshold: float = DEBUG_SIMILARITY_THRESHOLD, source_lang: str = "auto") -> str:
        # Using DEBUG_SIMILARITY_THRESHOLD defined globally
        if not question.strip():
            return "Please ask a question."

        # Assume question is Indonesian if source_lang is auto or id_ID
        effective_source_lang = source_lang if source_lang != "auto" else "id_ID"

        logging.info(f"Searching with threshold: {similarity_threshold}") # Log the threshold being used
        search_results = self.searcher.search(question, top_k=1, threshold=similarity_threshold, source_lang=effective_source_lang)

        if search_results and search_results[0]["similarity"] >= similarity_threshold:
            result = search_results[0]
            answer = result["answer"] # The answer is the retrieved paragraph
            context_info = result.get("context", "Unknown PDF")
            similarity_score = result.get("similarity", 0.0)
            # Return the matched paragraph as the answer
            return f"{answer} (Source: {context_info}, Similarity: {similarity_score:.2f})"

        logging.info(f"No suitable paragraph found (threshold {similarity_threshold}). Falling back to mBART generation.")
        return self._generate_mbart_answer(question, source_lang=effective_source_lang)

    def _generate_mbart_answer(self, question: str, source_lang: str) -> str:
        # Added check for model and tokenizer availability
        if self.model is None or self.tokenizer is None:
            logging.error("mBART model or tokenizer is not available for generation.")
            return "I'm sorry, the generation model is not available."

        logging.info(f"Attempting mBART generation. Model type: {type(self.model)}, Tokenizer type: {type(self.tokenizer)}")

        try:
            # Set source language for tokenizer
            self.tokenizer.src_lang = source_lang

            question_segments = segment_text(question, max_len=100)
            outputs = []
            for seg in question_segments:
                inputs = self.tokenizer(
                    seg,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH
                ).to(DEVICE)
                self.model.eval()
                with torch.no_grad():
                    output_ids = self.model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=MAX_LENGTH,
                        forced_bos_token_id=self.forced_bos_token_id, # Use the target language ID set during init
                        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                        num_beams=4,
                        length_penalty=1.0
                    )
                    output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    outputs.append(output)

            # Combine outputs, ensuring target language is generated
            combined_output = " ".join(outputs)
            # Basic check if output seems reasonable (not empty, not just source lang code)
            if combined_output and combined_output != self.target_lang:
                 logging.info(f"mBART generated response: {combined_output}")
                 return combined_output
            else:
                 logging.warning(f"mBART generated an empty or invalid response: '{combined_output}'")
                 return "No response generated by mBART."

        except Exception as e:
            logging.error(f"Error generating mBART answer: {e}")
            return "I'm sorry, an error occurred while generating the response."

    def set_target_language(self, lang_code: str):
        if self.tokenizer and lang_code in self.tokenizer.lang_code_to_id:
            self.target_lang = lang_code
            self.forced_bos_token_id = self.tokenizer.lang_code_to_id[lang_code]
            logging.info(f"Target language for generation updated to: {lang_code}")
            return True
        else:
            logging.error(f"Invalid language code: {lang_code} or tokenizer not available. Keeping current: {self.target_lang}")
            return False

    def save_research(self, data: str, filename: str = "research_output.txt") -> str:
        # Save research logic remains the same
        try:
            output_dir = "research"
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, filename)
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(data + "\n\n")
            return f"Research saved successfully to {file_path}"
        except Exception as e:
            return f"Error saving research: {e}"

def main():
    logging.info("Initializing Marca Smart Assistant...")
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        logging.info(f"Created data directory: {data_dir}")

    # Set target language to Indonesian for mBART generation
    assistant = SmartAssistant(data_dir=data_dir, target_lang="id_ID")

    # Check if assistant initialized correctly (model loaded)
    if assistant.model is None or assistant.tokenizer is None:
        print("\nError: Failed to initialize the assistant model. Exiting.")
        return

    print("\n===================================")
    print("Marca Assistant (Type 'exit' to quit)")
    print("===================================\n")

    while True:
        try:
            user_input = input("Question: ")
            if user_input.lower() == 'exit':
                print("Thank you for using Marca Assistant. Goodbye!")
                break
            if user_input.startswith("/lang "):
                lang_code = user_input.split(" ", 1)[1].strip()
                assistant.set_target_language(lang_code)
                continue

            # Assume input is Indonesian unless specified otherwise with lang_code:
            source_lang = "id_ID"
            if ":" in user_input:
                lang_part, question = user_input.split(":", 1)
                lang_part = lang_part.strip()
                if assistant.tokenizer and lang_part in assistant.tokenizer.lang_code_to_id:
                    source_lang = lang_part
                    user_input = question.strip()
                else:
                     logging.warning(f"Unrecognized language code {lang_part}, assuming id_ID")
                     user_input = user_input # Keep original input if lang code invalid

            # Use the globally defined debug threshold for testing
            answer = assistant.answer(user_input, similarity_threshold=DEBUG_SIMILARITY_THRESHOLD, source_lang=source_lang)
            print(f"\nMarca: {answer}\n")

        except EOFError:
             print("\nExiting...")
             break
        except KeyboardInterrupt:
             print("\nExiting...")
             break
        except Exception as e:
            logging.error(f"An error occurred in the main loop: {e}")
            print("An unexpected error occurred. Please try again.")

if __name__ == "__main__":
    main()
