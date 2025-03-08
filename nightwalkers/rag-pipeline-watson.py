import os
import json
import numpy as np
import faiss
import requests
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, List, Union, Literal

# Pydantic models
class Question(BaseModel):
    text: str
    kind: Literal["number", "name", "boolean", "names"]

class SourceReference(BaseModel):
    pdf_sha1: str = Field(..., description="SHA1 hash of the PDF file")
    page_index: int = Field(..., description="Physical page number in the PDF file")

class Answer(BaseModel):
    question_text: Optional[str] = Field(None, description="Text of the question")
    kind: Optional[Literal["number", "name", "boolean", "names"]] = Field(None, description="Kind of the question")
    value: Union[float, str, bool, List[str], Literal["N/A"]] = Field(..., description="Answer to the question")
    references: List[SourceReference] = Field([], description="References to the source material in the PDF file")

class AnswerSubmission(BaseModel):
    team_email: str = Field(..., description="Team email")
    submission_name: str = Field(..., description="Unique submission name")
    answers: List[Answer] = Field(..., description="List of answers to the questions")

# Helper function: Truncate text to a maximum number of tokens (using whitespace split)
def truncate_to_max_tokens(text: str, max_tokens: int = 512) -> str:
    tokens = text.split()
    if len(tokens) > max_tokens:
        tokens = tokens[-max_tokens:]  # keep the last max_tokens tokens (or choose first tokens as desired)
    return " ".join(tokens)

# WatsonX Embedding endpoint helper
def get_watsonx_embeddings(inputs: List[str], token: str, model_id: str = "ibm/granite-embedding-107m-multilingual") -> np.ndarray:
    url = "https://rag.timetoact.at/ibm/embeddings"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    # Enforce a 512-token limit on each input
    max_tokens_limit = 512
    truncated_inputs = []
    for inp in inputs:
        truncated = truncate_to_max_tokens(inp, max_tokens=max_tokens_limit)
        token_count = len(truncated.split())
        print(f"Original input token count: {len(inp.split())}, truncated token count: {token_count}")
        print(f"Truncated snippet: {truncated[:100]}...")
        truncated_inputs.append(truncated)

    payload = {
        "inputs": truncated_inputs,
        "model_id": model_id
    }
    print("Payload being sent to WatsonX:", payload)
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    embeddings = np.array(data.get("embeddings"))
    return embeddings

# Functions to load and process PDF JSON pages
def load_pdf_json(folder_path):
    """Load all PDF JSON pages from the folder, flattening lists if necessary."""
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            with open(os.path.join(folder_path, filename), 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    documents.extend(data)
                else:
                    documents.append(data)
    return documents

def get_page_content(doc):
    """
    Combine available text fields from the document.
    This includes main text, table content, and figure captions/ocr_text.
    """
    content = doc.get("text") or ""
    for table in doc.get("tables", []):
        headers = " ".join([str(x) for x in (table.get("headers") or []) if x is not None])
        rows = " ".join([
            " ".join([str(item) for item in (row or []) if item is not None])
            for row in (table.get("rows") or [])
        ])
        content += " " + headers + " " + rows
    figures = doc.get("figures")
    if figures:
        if isinstance(figures, list):
            for figure in figures:
                caption = str(figure.get("caption") or "")
                ocr_text = str(figure.get("ocr_text") or "")
                content += " " + caption + " " + ocr_text
        elif isinstance(figures, dict):
            caption = str(figures.get("caption") or "")
            ocr_text = str(figures.get("ocr_text") or "")
            content += " " + caption + " " + ocr_text
    return content

def build_vector_db(documents, watson_token):
    """
    Build a FAISS vector index for the document pages using cosine similarity.
    Uses WatsonX embeddings.
    """
    contents = [get_page_content(doc) for doc in documents]
    embeddings = get_watsonx_embeddings(contents, watson_token)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms  # Normalize embeddings
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index, documents, contents

# --- Dense retrieval and re-ranking using CrossEncoder ---
def dense_retrieve(query: str, watson_token: str, index, documents, top_k: int = 20):
    """
    Retrieve top_k candidates using WatsonX embeddings and the FAISS index.
    Returns a list of (document, dense_score) tuples.
    """
    query_embedding = get_watsonx_embeddings([query], watson_token)
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    scores, indices = index.search(query_embedding, top_k)
    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        candidate = documents[idx]
        content = get_page_content(candidate)
        if len(content.split()) < 10:
            continue
        candidates.append((candidate, score))
    return candidates

def rerank_candidates(query: str, candidates: List, cross_encoder: CrossEncoder):
    """
    Re-rank candidates using a cross-encoder.
    Returns the candidate document with the highest re-ranking score.
    """
    if not candidates:
        return None
    pairs = [(query, get_page_content(doc)) for doc, _ in candidates]
    rerank_scores = cross_encoder.predict(pairs)
    best_idx = int(np.argmax(rerank_scores))
    return candidates[best_idx][0]

# Updated process_question using two-stage retrieval
def process_question(question: Question, watson_token: str, dense_index, documents, cross_encoder, generation_token):
    # Stage 1: Dense retrieval using WatsonX embeddings
    candidates = dense_retrieve(question.text, watson_token, dense_index, documents, top_k=20)
    # Stage 2: Re-rank candidates using CrossEncoder
    best_candidate = rerank_candidates(question.text, candidates, cross_encoder)
    if best_candidate is None:
        print("No relevant document found for the question:", question.text)
        answer_val = "N/A"
        references = []
    else:
        context = get_page_content(best_candidate)
        # Truncate context to 512 tokens (limit per query)
        context = truncate_to_max_tokens(context, max_tokens=512)
        text_response = call_text_generation(question.text, question.kind, context, generation_token)
        print("Text generation response:", text_response)
        results = text_response.get("results")
        if results and isinstance(results, list) and len(results) > 0:
            generated_text = results[0].get("generated_text", "N/A")
        else:
            generated_text = "N/A"
        answer_val = generated_text.strip()
        filename = best_candidate.get("filename", "")
        pdf_sha1 = filename.split(".pdf")[0] if ".pdf" in filename else filename
        page_index = best_candidate.get("page", 0)
        references = [SourceReference(pdf_sha1=pdf_sha1, page_index=page_index)]
    return answer_val, references

def call_text_generation(query, question_type, context, generation_token):
    """
    Call the text generation API (using deepseek here),
    augmenting the question with context.
    """
    text_generation_url = "https://rag.timetoact.at/ibm/text_generation"
    headers = {
        "Authorization": f"Bearer {generation_token}",
        "Content-Type": "application/json"
    }
    augmented_query = f"Context: {context}\n question: {query} question_type: {question_type}"
    # Truncate the augmented query to 512 tokens as well
    augmented_query = truncate_to_max_tokens(augmented_query, max_tokens=512)
    print("Augmented query:", augmented_query)
    payload = {
        "input": [
            {"role": "system", "content": "For type of the question number answer only a metric number. name - only name is expected as an answer. names - multiple names. boolean - true or false. Don't add any additional text."},
            {"role": "user", "content": augmented_query}
        ],
        "model_id": "deepseek/deepseek-r1-distill-llama-70b",
        "parameters": {
            "max_new_tokens": 100,
            "min_new_tokens": 1
        }
    }
    response = requests.post(text_generation_url, headers=headers, json=payload, verify=False)
    response.raise_for_status()
    return response.json()

def process_questions(questions_file: str, folder_path: str, watson_token: str, generation_token: str, team_email: str,
                      submission_name: str) -> AnswerSubmission:
    with open(questions_file, 'r') as f:
        questions_json = json.load(f)
    questions = [Question(**q) for q in questions_json]

    documents = load_pdf_json(folder_path)
    dense_index, docs, _ = build_vector_db(documents, watson_token)
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    answers = []
    for q in questions:
        value, refs = process_question(q, watson_token, dense_index, documents, cross_encoder, generation_token)
        if q.kind == "number":
            try:
                cleaned = value.replace(",", "").replace(" ", "")
                value = float(cleaned)
            except (ValueError, TypeError):
                value = "N/A"
        elif q.kind == "boolean":
            if isinstance(value, str):
                lower_val = value.strip().lower()
                if lower_val in ["true", "yes"]:
                    value = True
                elif lower_val in ["false", "no"]:
                    value = False
                else:
                    value = False
            else:
                value = bool(value)
        answer_obj = Answer(
            question_text=q.text,
            kind=q.kind,
            value=value,
            references=refs
        )
        answers.append(answer_obj)

    submission = AnswerSubmission(
        team_email=team_email,
        submission_name=submission_name,
        answers=answers
    )
    return submission

# Example usage
if __name__ == "__main__":
    load_dotenv()
    folder_path = "jsons"
    questions_file = "questions.json"

    watson_token = os.getenv("MY_SECRET_KEY", "YOUR_WATSON_TOKEN")
    generation_token = os.getenv("MY_SECRET_KEY", "YOUR_WATSON_TOKEN")
    team_email = "markdrozdov0@gmail.com"
    submission_name = "experiment_01"

    submission = process_questions(questions_file, folder_path, watson_token, generation_token, team_email, submission_name)
    print(submission)

    with open("answer_submission.json", "w") as f:
        f.write(json.dumps(submission.model_dump(), indent=2))

    print("Answer submission created and saved to answer_submission.json")
