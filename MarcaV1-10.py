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
            return text
        try:
            if source_lang == "auto":
                source_lang = DEFAULT_LANG
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
                    forced_bos_token_id=self.mbart_tokenizer.lang_code_to_id.get(DEFAULT_LANG),
                    num_beams=4,
                    length_penalty=1.0
                )
            return self.mbart_tokenizer.decode(translated_ids[0], skip_special_tokens=True)
        except Exception as e:
            logging.error(f"Error translating text: {e}")
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
            for dict_word in self.word_dict:
                dist = levenshtein_distance(word, dict_word)
                if dist < min_dist and dist <= 2:
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
        # Translate only if dataset language is not English
        if self.dataset_lang != "en_XX":
             translated_questions = [self._translate_to_english(q, source_lang=self.dataset_lang) for q in questions]
        else:
             translated_questions = questions # Assume questions are already in English if lang is en_XX
        
        self.embeddings = []
        for question in translated_questions:
            if question in self.embedding_cache:
                self.embeddings.append(self.embedding_cache[question])
            else:
                try:
                    embedding = self.model.encode(question, convert_to_numpy=True)
                    self.embedding_cache[question] = embedding
                    self.embeddings.append(embedding)
                except Exception as e:
                    logging.error(f"Error encoding question '{question}': {e}")
                    self.embeddings.append(np.zeros(self.model.get_sentence_embedding_dimension()))
        
        self.embeddings = np.array(self.embeddings)
        logging.info(f"Computed embeddings for {len(self.embeddings)} questions")
    
    def _combine_scores(self, query_embedding: np.ndarray, question_embeddings: np.ndarray, 
                       query_tokens: List[str], question_tokens: List[List[str]]) -> np.ndarray:
        """Combine semantic and token-based similarity scores."""
        # Ensure embeddings are not empty and have compatible shapes
        if question_embeddings.size == 0 or query_embedding.size == 0:
            return np.array([])
        if question_embeddings.shape[1] != query_embedding.shape[0]:
             logging.error(f"Embedding shape mismatch: {question_embeddings.shape} vs {query_embedding.shape}")
             return np.array([])

        cosine_scores = np.dot(question_embeddings, query_embedding) / (
            np.linalg.norm(question_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        token_scores = []
        query_token_set = set(query_tokens)
        for q_tokens in question_tokens:
            q_token_set = set(q_tokens)
            common_tokens = len(query_token_set & q_token_set) / max(len(q_token_set), 1)
            token_scores.append(common_tokens)
        
        # Ensure scores are numpy arrays before combining
        cosine_scores = np.nan_to_num(cosine_scores) # Handle potential NaNs
        token_scores_np = np.array(token_scores)
        
        # Ensure shapes match for broadcasting or element-wise operation
        if cosine_scores.shape != token_scores_np.shape:
             logging.warning(f"Score shape mismatch: Cosine {cosine_scores.shape}, Token {token_scores_np.shape}. Using only cosine.")
             return cosine_scores # Fallback to cosine scores if shapes mismatch

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
            # Translate only if source language is not English
            if source_lang != "en_XX":
                translated_segments = [self._translate_to_english(seg, source_lang) for seg in query_segments]
            else:
                translated_segments = query_segments # Assume query is already in English

            segment_embeddings = []
            for seg in translated_segments:
                if seg in self.embedding_cache:
                    segment_embeddings.append(self.embedding_cache[seg])
                else:
                    embedding = self.model.encode(seg, convert_to_numpy=True)
                    self.embedding_cache[seg] = embedding
                    segment_embeddings.append(embedding)
            
            if not segment_embeddings: # Handle case where translation/encoding fails
                 logging.warning("Could not generate embeddings for query segments.")
                 return []

            query_embedding = np.mean(segment_embeddings, axis=0)
            query_tokens = corrected_query.lower().split()
            question_tokens = [qa["question"].lower().split() for qa in self.qa_pairs]
            
            similarities = self._combine_scores(query_embedding, self.embeddings, query_tokens, question_tokens)
            
            if similarities.size == 0: # Handle empty similarities
                 logging.warning("No similarity scores computed.")
                 return []

            # Ensure top_k is not greater than the number of available similarities
            actual_top_k = min(top_k, len(similarities))
            top_indices = np.argsort(-similarities)[:actual_top_k]
            
            results = []
            for idx in top_indices:
                similarity = similarities[idx]
                if similarity >= threshold:
                    results.append({
                        "question": self.qa_pairs[idx]["question"],
                        "answer": self.qa_pairs[idx]["answer"],
                        "context": self.qa_pairs[idx].get("context", ""),
                        "evidence": self.qa_pairs[idx].get("evidence", []),
                        "similarity": float(similarity) # Ensure similarity is JSON serializable
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
            self.forced_bos_token_id = self.tokenizer.lang_code_to_id.get(target_lang)
            logging.info(f"Target language: {target_lang}, BOS token ID: {self.forced_bos_token_id}")
        except Exception as e:
            logging.error(f"Error loading mBART model: {e}")
            self.model = None
            self.tokenizer = None
        
        logging.info("Initializing semantic searcher...")
        # Assuming PDF content is in Indonesian (id_ID), set dataset_lang accordingly for translation
        self.searcher = SemanticSearcher(mbart_tokenizer=self.tokenizer, mbart_model=self.model, dataset_lang="id_ID") 
        logging.info(f"Loading dataset from {data_dir}")
        self._load_dataset(data_dir)
        logging.info("SmartAssistant initialization complete")
    
    # Removed process_pdf method as it's replaced by text extraction below

    def _extract_paragraphs_from_pdf(self, pdf_path: str) -> List[str]:
        """Extracts text from a PDF and splits it into paragraphs."""
        paragraphs = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                full_text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"
            
            # Split into paragraphs based on double newlines, clean up whitespace
            raw_paragraphs = re.split(r'\n\s*\n', full_text)
            for para in raw_paragraphs:
                cleaned_para = re.sub(r'\s+', ' ', para).strip()
                # Filter out very short paragraphs (likely headers/footers/noise)
                if cleaned_para and len(cleaned_para.split()) > 5: 
                    paragraphs.append(cleaned_para)
            logging.info(f"Extracted {len(paragraphs)} paragraphs from {os.path.basename(pdf_path)}")
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
        cache_file = os.path.join(data_dir, "pdf_paragraphs.json")
        pdf_files_to_process = [f for f in os.listdir(data_dir) if f.startswith("unsup") and f.endswith(".pdf")]
        
        # Load cached PDF paragraphs if cache exists and no new PDFs need processing
        # (Simple check: if cache exists, assume it's up-to-date for this example)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    qa_pairs.extend(json.load(f))
                files_loaded += len(pdf_files_to_process) # Assume all PDFs were cached
                logging.info(f"Loaded {len(qa_pairs)} QA pairs from cache")
            except Exception as e:
                logging.error(f"Error loading cache: {e}")
                # If cache is corrupted, proceed to re-process PDFs
                qa_pairs = [] # Reset qa_pairs
                if os.path.exists(cache_file):
                     os.remove(cache_file) # Remove corrupted cache
        
        # Process JSON files (existing logic)
        for filename in os.listdir(data_dir):
            if filename.endswith(".json") and filename != "pdf_paragraphs.json":
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
        
        # Process PDF files using text extraction if cache wasn't loaded or needs update
        if not os.path.exists(cache_file): # Process if cache doesn't exist
            pdf_qa_pairs = []
            if pdf_files_to_process:
                logging.info(f"Processing {len(pdf_files_to_process)} PDF files using text extraction...")
                for i, filename in enumerate(pdf_files_to_process):
                    pdf_path = os.path.join(data_dir, filename)
                    paragraphs = self._extract_paragraphs_from_pdf(pdf_path)
                    for para in paragraphs:
                        pdf_qa_pairs.append({
                            "question": para, # Use paragraph as question
                            "answer": para,   # Use paragraph as answer
                            "context": f"{filename}", # Add filename as context
                            "evidence": []
                        })
                    logging.info(f"Completed {i+1}/{len(pdf_files_to_process)} PDFs: {filename}")
                
                # Save extracted paragraphs to cache
                try:
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(pdf_qa_pairs, f, ensure_ascii=False, indent=2)
                    logging.info(f"Saved {len(pdf_qa_pairs)} QA pairs from PDFs to cache: {cache_file}")
                    qa_pairs.extend(pdf_qa_pairs) # Add newly processed pairs
                    files_loaded += len(pdf_files_to_process) # Update count
                except Exception as e:
                     logging.error(f"Error saving PDF QA pairs to cache: {e}")
            else:
                logging.info("No PDF files found to process")
        
        logging.info(f"Loaded a total of {len(qa_pairs)} QA pairs from {files_loaded} files")
        if qa_pairs:
            # Set dataset language hint for searcher (assuming PDF is Indonesian)
            self.searcher.dataset_lang = "id_ID" 
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
        
        dataset = QADataset(data_dir, self.tokenizer)
        if len(dataset) == 0:
            logging.warning("No training data found. Skipping training.")
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
    
    def answer(self, question: str, similarity_threshold: float = 0.6, source_lang: str = "auto") -> str:
        # Answer logic remains largely the same, relies on searcher
        if not question.strip():
            return "Please ask a question."
        
        # Assume question is Indonesian if source_lang is auto or id_ID
        effective_source_lang = source_lang if source_lang != "auto" else "id_ID"

        search_results = self.searcher.search(question, top_k=1, threshold=similarity_threshold, source_lang=effective_source_lang)
        
        if search_results and search_results[0]["similarity"] >= similarity_threshold:
            result = search_results[0]
            answer = result["answer"] # The answer is the paragraph itself
            context_info = result.get("context", "")
            similarity_score = result.get("similarity", 0.0)
            # Return the matched paragraph as the answer
            return f"{answer} (Source: {context_info}, Similarity: {similarity_score:.2f})"
        
        # Fallback to mBART generation if no good match found in PDF paragraphs
        logging.info(f"No suitable paragraph found (threshold {similarity_threshold}). Falling back to mBART generation.")
        return self._generate_mbart_answer(question, source_lang=effective_source_lang)
    
    def _generate_mbart_answer(self, question: str, source_lang: str) -> str:
        # mBART generation logic remains the same
        if self.model is None or self.tokenizer is None:
            return "I'm sorry, I don't have an answer for that question at the moment."
        
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
                        forced_bos_token_id=self.forced_bos_token_id,
                        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                        num_beams=4,
                        length_penalty=1.0
                    )
                    output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    outputs.append(output)
            
            return " ".join(outputs) if outputs else "No response generated."
        except Exception as e:
            logging.error(f"Error generating mBART answer: {e}")
            return "I'm sorry, I couldn't generate an appropriate response at the moment."
    
    def set_target_language(self, lang_code: str):
        # Language setting logic remains the same
        if lang_code in self.tokenizer.lang_code_to_id:
            self.target_lang = lang_code
            self.forced_bos_token_id = self.tokenizer.lang_code_to_id[lang_code]
            logging.info(f"Target language updated to: {lang_code}")
            return True
        else:
            logging.error(f"Invalid language code: {lang_code}. Keeping current: {self.target_lang}")
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
    # Ensure data directory exists
    if not os.path.exists("data"):
        os.makedirs("data")
        logging.info("Created data directory.")
        # You might want to copy the PDF here if it's not already there
        # import shutil
        # shutil.copy("/path/to/your/unsup1.pdf", "data/unsup1.pdf")

    assistant = SmartAssistant(data_dir="data", target_lang="id_ID") # Set target lang to Indonesian
    
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
            
            # Assume input is Indonesian unless specified otherwise
            source_lang = "id_ID" 
            if ":" in user_input:
                lang_part, question = user_input.split(":", 1)
                lang_part = lang_part.strip()
                if lang_part in assistant.tokenizer.lang_code_to_id:
                    source_lang = lang_part
                    user_input = question.strip()
            
            answer = assistant.answer(user_input, source_lang=source_lang)
            print(f"\nMarca: {answer}\n")
            
            # Removed save research prompt for simplicity in debugging
            # save_option = input("Do you want to save this research? (y/n): ")
            # if save_option.lower() == 'y':
            #     research_data = f"Question: {user_input}\nAnswer: {answer}"
            #     print(assistant.save_research(research_data))
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
