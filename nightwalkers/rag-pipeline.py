import os
import json
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer
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


# Functions to load and process PDF JSON pages

def load_pdf_json(folder_path):
    """Load all PDF JSON pages from the folder, flattening lists if necessary."""
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            with open(os.path.join(folder_path, filename), 'r') as f:
                data = json.load(f)
                # If the file returns a list of pages, extend the documents list.
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
    # Append table data (if any)
    for table in doc.get("tables", []):
        headers = " ".join([str(x) for x in (table.get("headers") or []) if x is not None])
        rows = " ".join([
            " ".join([str(item) for item in (row or []) if item is not None])
            for row in (table.get("rows") or [])
        ])
        content += " " + headers + " " + rows
    # Append figure data (figures can be a list or a dict)
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


def build_vector_db(documents, model):
    """
    Build a FAISS vector index for the document pages using cosine similarity.
    Embeddings are normalized, and we use an inner product index.
    """
    contents = [get_page_content(doc) for doc in documents]
    embeddings = model.encode(contents, convert_to_numpy=True)
    # Normalize embeddings to unit vectors for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Use inner product for cosine similarity
    index.add(embeddings)
    return index, documents, contents


def search_vector_db(query, model, index, documents, top_k=5, similarity_threshold=0.4):
    """
    Retrieve the most relevant document page based on cosine similarity.
    Returns the candidate with the highest similarity above the threshold.
    Optionally, filters by checking that the candidate content includes key query terms and isn't too short.
    """
    query_embedding = model.encode([query], convert_to_numpy=True)
    # Normalize the query embedding
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    scores, indices = index.search(query_embedding, top_k)
    best_candidate = None
    best_score = -1
    # Retrieve top_k candidates and filter by threshold, keyword presence, and minimum length.
    for score, idx in zip(scores[0], indices[0]):
        print(score, idx)
        if score >= similarity_threshold:
            candidate = documents[idx]
            content = get_page_content(candidate).lower()
            # Skip candidate if content is almost empty (fewer than 10 words)
            if len(content.split()) < 10:
                continue
            # Require that at least one significant keyword from the query appears in the candidate.
            query_keywords = [word for word in query.lower().split() if len(word) > 3]
            if any(keyword in content for keyword in query_keywords):
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
    return best_candidate


def call_text_generation(query, question_type, context, watson_token):
    """
    Call the text generation API, augmenting the question with context.
    """
    text_generation_url = "https://rag.timetoact.at/ibm/text_generation"
    headers = {
        "Authorization": f"Bearer {watson_token}",
        "Content-Type": "application/json"
    }
    augmented_query = f"Context: {context}\n question: {query} question_type: {question_type}"
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
    response = requests.post(text_generation_url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()


def process_question(question: Question, model, index, docs, watson_token):
    relevant_doc = search_vector_db(question.text, model, index, docs, top_k=5, similarity_threshold=0.5)
    if relevant_doc is None:
        print("No relevant document found for the question:", question.text)
        answer_val = "N/A"
        references = []
    else:
        context = get_page_content(relevant_doc)
        text_response = call_text_generation(question.text, question.kind, context, watson_token)
        print("Text generation response:", text_response)
        results = text_response.get("results")
        if results and isinstance(results, list) and len(results) > 0:
            generated_text = results[0].get("generated_text", "N/A")
        else:
            generated_text = "N/A"
        answer_val = generated_text.strip()
        filename = relevant_doc.get("filename", "")
        pdf_sha1 = filename.split(".pdf")[0] if ".pdf" in filename else filename
        page_index = relevant_doc.get("page", 0)
        references = [SourceReference(pdf_sha1=pdf_sha1, page_index=page_index)]
    return answer_val, references


def process_questions(questions_file: str, folder_path: str, watson_token: str, team_email: str,
                      submission_name: str) -> AnswerSubmission:
    with open(questions_file, 'r') as f:
        questions_json = json.load(f)
    questions = [Question(**q) for q in questions_json]

    documents = load_pdf_json(folder_path)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index, docs, _ = build_vector_db(documents, model)

    answers = []
    for q in questions:
        value, refs = process_question(q, model, index, docs, watson_token)
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
    team_email = "markdrozdov0@gmail.com"
    submission_name = "experiment_01"

    submission = process_questions(questions_file, folder_path, watson_token, team_email, submission_name)
    print(submission)

    with open("answer_submission.json", "w") as f:
        f.write(json.dumps(submission.model_dump(), indent=2))

    print("Answer submission created and saved to answer_submission.json")
