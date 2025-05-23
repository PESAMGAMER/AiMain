import os
import json
import faiss
import numpy as np
import requests
from typing import List
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from flask import Flask, request, render_template, jsonify

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
FOLDER_PATH = "data"
INDEX_PATH = "faiss.index"
TEXTS_PATH = "texts.json"

app = Flask(__name__)
embedder = SentenceTransformer("intfloat/multilingual-e5-base")

def load_laws(filepath: str) -> List[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [f"{law['law_name']} มาตรา {law['section_num']}:\n{law['section_content']}" for law in data]
    return texts

def load_laws_from_folder(folder_path: str) -> List[str]:
    all_texts = []
    print(f"📚 โหลดไฟล์กฎหมายจากโฟลเดอร์ {folder_path}")
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            try:
                texts = load_laws(file_path)
                all_texts.extend(texts)
                print(f"✓ โหลด {filename} สำเร็จ ({len(texts)} มาตรา)")
            except Exception as e:
                print(f"✗ ไม่สามารถโหลด {filename}: {str(e)}")
    return all_texts

def embed_chunks(texts: List[str], batch_size: int = 16) -> np.ndarray:
    print("🔍 สร้าง Embeddings...")
    embeddings = embedder.encode(texts, batch_size=batch_size, show_progress_bar=True,
                                 convert_to_numpy=True, normalize_embeddings=True)
    return embeddings

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    print("📦 สร้าง FAISS Index...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def search_similar_contexts(query: str, texts: List[str], index: faiss.IndexFlatIP, top_k: int = 5) -> List[str]:
    print("🧠 ค้นหาข้อมูล...")
    query_embedding = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(query_embedding, top_k)

    threshold = 0.4
    filtered_results = [(texts[i], scores[0][idx]) for idx, i in enumerate(indices[0])
                        if scores[0][idx] > threshold]
    filtered_results.sort(key=lambda x: x[1], reverse=True)
    return [text for text, _ in filtered_results]

def generate_answer_ollama(context: str, query: str, max_tokens: int = 512) -> str:
    print("🧾 สร้างคำตอบผ่าน Ollama...")
    if context.strip():
        prompt = f"""คุณเป็น AI ที่ช่วยตอบคำถามทั้งเรื่องกฎหมายและคำถามทั่วไป โดยถ้ามีข้อมูลกฎหมายที่เกี่ยวข้องให้อ้างอิงด้วย

บริบทกฎหมายที่เกี่ยวข้อง:
{context}

คำถาม: {query}

โปรดตอบคำถามโดยใช้ภาษาที่เข้าใจง่าย อ้างอิงกฎหมายถ้าเกี่ยวข้อง หรือตอบตามความรู้ทั่วไปถ้าไม่เกี่ยวกับกฎหมาย"""
    else:
        prompt = f"""คุณเป็น AI ที่ช่วยตอบคำถามทั้งเรื่องกฎหมายและคำถามทั่วไป

คำถาม: {query}

โปรดตอบคำถามโดยใช้ภาษาที่เข้าใจง่าย ตอบตามความรู้ทั่วไปเนื่องจากไม่พบข้อกฎหมายที่เกี่ยวข้อง"""

    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens
        }
    })

    if response.status_code == 200:
        return response.json()["response"].strip()
    else:
        return f"เกิดข้อผิดพลาดในการติดต่อกับ Ollama: {response.status_code}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    question = request.json['question']
    top_contexts = search_similar_contexts(question, texts, index)
    context = "\n\n".join(top_contexts)
    answer = generate_answer_ollama(context, question)
    return jsonify({
        'context': context,
        'answer': answer
    })

if __name__ == "__main__":
    # โหลดจากแคชถ้ามี
    if os.path.exists(INDEX_PATH) and os.path.exists(TEXTS_PATH):
        print("📂 โหลดข้อมูลจากแคช...")
        index = faiss.read_index(INDEX_PATH)
        with open(TEXTS_PATH, "r", encoding="utf-8") as f:
            texts = json.load(f)
    else:
        texts = load_laws_from_folder(FOLDER_PATH)
        if not texts:
            print("❌ ไม่พบข้อมูลกฎหมายในโฟลเดอร์")
            exit(1)

        print(f"\n📝 พบข้อมูลกฎหมายทั้งหมด {len(texts)} มาตรา")
        embeddings = embed_chunks(texts)
        index = build_faiss_index(embeddings)

        # แคชไว้ใช้รอบหน้า
        faiss.write_index(index, INDEX_PATH)
        with open(TEXTS_PATH, "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False)

    print("\n🎯 ระบบ RAG (Ollama) พร้อมใช้งาน!\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
