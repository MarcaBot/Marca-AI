import os
import json
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
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
                    print(f"Error loading {file_path}: {e}")
    
    def _process_data(self, data: Any):
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
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            self.model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
            self.model.to(DEVICE)
        
        self.mbart_tokenizer = mbart_tokenizer
        self.mbart_model = mbart_model
        self.dataset_lang = dataset_lang
        self.qa_pairs = []
        self.embeddings = None
        self.embedding_cache = {}  # Cache for embeddings
        self.word_dict = set()  # Dictionary for typo correction
    
    def add_qa_pairs(self, qa_pairs: List[Dict[str, str]]):
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
                    length_penalty=1.0,
                    early_stopping=True
                )
            return self.mbart_tokenizer.decode(translated_ids[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error translating text: {e}")
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
                if dist < min_dist and dist <= 2:  # Allow small edit distances
                    min_dist = dist
                    best_match = dict_word
            corrected_words.append(best_match)
        return " ".join(corrected_words)
    
    def _compute_embeddings(self):
        """Compute embeddings with caching for efficiency."""
        if not self.qa_pairs:
            print("Warning: No QA pairs to compute embeddings for.")
            return
        
        questions = [qa_pair["question"] for qa_pair in self.qa_pairs]
        translated_questions = [self._translate_to_english(q, source_lang=self.dataset_lang) for q in questions]
        
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
                    print(f"Error encoding question '{question}': {e}")
                    self.embeddings.append(np.zeros(self.model.get_sentence_embedding_dimension()))
        
        self.embeddings = np.array(self.embeddings)
        print(f"Computed embeddings for {len(self.embeddings)} questions")
    
    def _combine_scores(self, query_embedding: np.ndarray, question_embeddings: np.ndarray, 
                       query_tokens: List[str], question_tokens: List[List[str]]) -> np.ndarray:
        """Combine semantic and token-based similarity scores."""
        cosine_scores = np.dot(question_embeddings, query_embedding) / (
            np.linalg.norm(question_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        token_scores = []
        for q_tokens in question_tokens:
            common_tokens = len(set(query_tokens) & set(q_tokens)) / max(len(set(q_tokens)), 1)
            token_scores.append(common_tokens)
        
        return 0.7 * cosine_scores + 0.3 * np.array(token_scores)
    
    def search(self, query: str, top_k: int = 3, threshold: float = 0.6, source_lang: str = "auto") -> List[Dict[str, Any]]:
        """Enhanced semantic search with typo correction and long input handling."""
        if self.embeddings is None or len(self.embeddings) == 0:
            print("No embeddings available for search.")
            return []
        
        try:
            # Correct typos
            corrected_query = self._correct_typos(query)
            print(f"Corrected query: {corrected_query}")
            
            # Segment long queries
            query_segments = segment_text(corrected_query, max_len=100)
            translated_segments = [self._translate_to_english(seg, source_lang) for seg in query_segments]
            
            # Compute embeddings for segments
            segment_embeddings = []
            for seg in translated_segments:
                if seg in self.embedding_cache:
                    segment_embeddings.append(self.embedding_cache[seg])
                else:
                    embedding = self.model.encode(seg, convert_to_numpy=True)
                    self.embedding_cache[seg] = embedding
                    segment_embeddings.append(embedding)
            
            query_embedding = np.mean(segment_embeddings, axis=0)  # Aggregate segment embeddings
            
            # Token-based scoring
            query_tokens = corrected_query.lower().split()
            question_tokens = [qa["question"].lower().split() for qa in self.qa_pairs]
            
            # Combine semantic and token-based scores
            similarities = self._combine_scores(query_embedding, self.embeddings, query_tokens, question_tokens)
            
            # Get top-k results
            top_indices = np.argsort(-similarities)[:top_k]
            results = []
            for idx in top_indices:
                similarity = similarities[idx]
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
            print(f"Error during semantic search: {e}")
            return []

class SmartAssistant:
    """Enhanced smart assistant with robust context handling and typo correction."""
    
    def __init__(self, model_path: Optional[str] = None, data_dir: str = "data", target_lang: str = DEFAULT_LANG):
        self.data_dir = data_dir
        self.target_lang = target_lang
        print(f"Initializing SmartAssistant on device: {DEVICE}")
        
        try:
            if model_path and os.path.exists(model_path):
                print(f"Loading pre-trained model from {model_path}")
                self.model = MBartForConditionalGeneration.from_pretrained(model_path)
                self.tokenizer = MBartTokenizer.from_pretrained(model_path)
            else:
                print(f"Loading base mBART model: {MODEL_NAME}")
                self.model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
                self.tokenizer = MBartTokenizer.from_pretrained(MODEL_NAME)
            
            self.model = self.model.to(DEVICE)
            self.forced_bos_token_id = self.tokenizer.lang_code_to_id.get(target_lang)
            print(f"Target language: {target_lang}, BOS token ID: {self.forced_bos_token_id}")
        except Exception as e:
            print(f"Error loading mBART model: {e}")
            self.model = None
            self.tokenizer = None
        
        print("Initializing semantic searcher...")
        self.searcher = SemanticSearcher(mbart_tokenizer=self.tokenizer, mbart_model=self.model, dataset_lang="en_XX")
        print(f"Loading dataset from {data_dir}")
        self._load_dataset(data_dir)
        print("SmartAssistant initialization complete")
    
    def _load_dataset(self, data_dir: str):
        qa_pairs = []
        if not os.path.exists(data_dir):
            print(f"Data directory {data_dir} not found. Creating it.")
            os.makedirs(data_dir, exist_ok=True)
            return
        
        files_loaded = 0
        for filename in os.listdir(data_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(data_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
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
                    files_loaded += 1
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        print(f"Loaded {len(qa_pairs)} QA pairs from {files_loaded} files")
        if qa_pairs:
            self.searcher.add_qa_pairs(qa_pairs)
    
    def train(self, data_dir: str = None, output_dir: str = "model", epochs: int = EPOCHS):
        if self.model is None or self.tokenizer is None:
            print("mBART model not available for training.")
            return
        
        if data_dir is None:
            data_dir = self.data_dir
        
        dataset = QADataset(data_dir, self.tokenizer)
        if len(dataset) == 0:
            print("No training data found. Skipping training.")
            return
        
        print(f"Training on {len(dataset)} examples for {epochs} epochs")
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
                        print(f"Epoch {epoch+1}/{epochs}, Batch {batch_count}/{len(dataloader)}, Loss: {loss.item():.4f}")
                except Exception as e:
                    print(f"Error in training batch: {e}")
                    continue
            avg_loss = total_loss / max(1, batch_count)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            print(f"Model saved to {output_dir}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def answer(self, question: str, similarity_threshold: float = 0.7, source_lang: str = "auto") -> str:
        if not question.strip():
            return "Please ask a question."
        
        search_results = self.searcher.search(question, top_k=1, threshold=similarity_threshold, source_lang=source_lang)
        
        if search_results and search_results[0]["similarity"] >= similarity_threshold:
            result = search_results[0]
            answer = result["answer"]
            evidence = result["evidence"]
            if evidence:
                evidence_texts = [e.get("text", "") for e in evidence]
                sources = [e.get("source", "") for e in evidence]
                evidence_output = "\n".join(
                    f"reason: {text}\n{source}" for text, source in zip(evidence_texts, sources) if text and source
                )
                return f"{answer}\n{evidence_output}"
            return answer
        
        return self._generate_mbart_answer(question)
    
    def _generate_mbart_answer(self, question: str) -> str:
        if self.model is None or self.tokenizer is None:
            return "I'm sorry, I don't have an answer for that question at the moment."
        
        try:
            # Segment long questions
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
                        length_penalty=1.0,
                        early_stopping=True
                    )
                    output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    outputs.append(output)
            
            # Combine outputs intelligently
            return " ".join(outputs) if outputs else "No response generated."
        except Exception as e:
            print(f"Error generating mBART answer: {e}")
            return "I'm sorry, I couldn't generate an appropriate response at the moment."
    
    def set_target_language(self, lang_code: str):
        if lang_code in self.tokenizer.lang_code_to_id:
            self.target_lang = lang_code
            self.forced_bos_token_id = self.tokenizer.lang_code_to_id[lang_code]
            print(f"Target language updated to: {lang_code}")
            return True
        else:
            print(f"Invalid language code: {lang_code}. Keeping current: {self.target_lang}")
            return False
    
    def save_research(self, data: str, filename: str = "research_output.txt") -> str:
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
    print("Initializing Marca Smart Assistant...")
    assistant = SmartAssistant(data_dir="data")
    
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
            
            source_lang = "auto"
            if ":" in user_input:
                lang_part, question = user_input.split(":", 1)
                lang_part = lang_part.strip()
                if lang_part in assistant.tokenizer.lang_code_to_id:
                    source_lang = lang_part
                    user_input = question.strip()
            
            answer = assistant.answer(user_input, source_lang=source_lang)
            print(f"\nMarca: {answer}\n")
            
            save_option = input("Do you want to save this research? (y/n): ")
            if save_option.lower() == 'y':
                research_data = f"Question: {user_input}\nAnswer: {answer}"
                save_result = assistant.save_research(research_data)
                print(save_result)
                
        except KeyboardInterrupt:
            print("\nExiting Marca Assistant. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Let's continue with a new question.")

if __name__ == "__main__":
    main()
