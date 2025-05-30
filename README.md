# <p align="center">DEEP DIVE TO PDF KNOWLEDGE-BASED AI</p>

#### Apa itu PDF Knowledge-Based AI? saat ini mungkin kita melihat bahwa AI kita hanya belajar dari Supervised Learning dimana ketika kita mengatur dapat dimengerti bahwa ini hanyalah _klasifikasi_, bukan _clustering_. Disini, kita mencoba menggabungkan keduanya sehingga terlihat lebih modern. Spesifikasi laptop yang dibutuhkan nantinya adalah minimal RAM = 16 GB dan minimal disk = 10-20GB. Terimakasih. Apabila kamu merasa keberatan, maka instal lah versi sebelumnya.

📚 Marca Smart AI Assistant
Versi: 1.10 (Updated Version with different file)
Pengembang: Marco Julius Andreas Bakkara

⚠️ PERINGATAN HUKUM: LARANGAN KOMERSIAL TANPA IZIN
Semua hak cipta dilindungi undang-undang. Sistem Marca Smart AI Assistant ini dibuat untuk keperluan penelitian, pembelajaran, dan edukasi. Penggunaan sistem dalam bentuk promosi, iklan, penawaran produk/jasa, dan/atau kegiatan komersial lainnya tanpa izin tertulis dari pengembang adalah pelanggaran hukum.

🚫 Larangan Meliputi:
Menjual kembali sistem atau bagian dari kodenya.

Mengintegrasikan sistem dengan aplikasi pihak ketiga untuk monetisasi.

Mengklaim kepemilikan atas AI ini atau turunannya.

Menyebarkan sistem dalam bentuk layanan chatbot berbayar tanpa izin.

Setiap pelanggaran akan dikenai sanksi sesuai hukum yang berlaku berdasarkan UU Hak Cipta dan UU ITE di Indonesia atau yurisdiksi internasional lainnya.

🧠 Tentang Proyek
Marca Smart AI Assistant adalah sistem chatbot generatif berbasis model mBART (Multilingual BART) dengan dukungan pemahaman semantik, koreksi typo, dan multibahasa. Sistem ini menggabungkan pendekatan supervised learning, semantic search, dan auto-translation untuk menjawab pertanyaan pengguna dengan konteks relevan.

📂 Struktur Proyek
bash
Copy
Edit
├── supervised.py          # Model utama AI berbasis mBART + Semantic Search
├── flasken.py             # Backend Flask API
├── index.html             # Antarmuka Web (frontend modern)
├── data/                  # Folder dataset JSON
├── model/                 # Folder output model hasil training
├── research/              # Folder output hasil penelitian/simpanan pengguna
📊 Format Dataset
✅ Format Dataset SQuAD-like (support):
json
Copy
Edit
[
  {
    "title": "Judul Artikel",
    "paragraphs": [
      {
        "context": "Isi paragraf terkait pertanyaan",
        "qas": [
          {
            "question": "Apa isi pertanyaan?",
            "answers": [{"text": "Jawaban dari pertanyaan"}],
            "is_impossible": false
          }
        ]
      }
    ]
  }
]
✅ Format Dataset Custom Sederhana:
json
Copy
Edit
[
  {
    "question": "Apa itu Pematangsiantar?",
    "answer": "Pematangsiantar adalah kota di Sumatera Utara.",
    "evidence": [{"text": "Sumber info", "source": "Wikipedia"}]
  }
]
Dataset bisa ditaruh dalam folder data/ dan akan otomatis diproses.

⚙️ Persyaratan Sistem
🖥️ Minimal Spesifikasi Hardware:
Komponen	Minimum	Rekomendasi
RAM	8 GB	16 GB
GPU	Tidak wajib (CPU bisa)	NVIDIA RTX 2060 atau lebih tinggi
CPU	4 core	8 core
Disk	2 GB ruang kosong untuk model	5+ GB

Training dengan CPU bisa dilakukan, tapi akan lebih lambat.

🛠️ Instalasi
1. Kloning Repositori
bash
Copy
Edit
git clone https://github.com/marcobakkara/marca-ai-assistant.git
cd marca-ai-assistant
2. Buat Virtual Environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
3. Install Dependensi
bash
Copy
Edit
pip install -r requirements.txt
Contoh requirements.txt:

txt
Copy
Edit
flask
torch
transformers
sentence-transformers
numpy
🚀 Menjalankan Aplikasi
1. Jalankan Flask Backend
bash
Copy
Edit
python flasken.py
2. Buka di Browser
url
Copy
Edit
http://localhost:5000
🧪 Cara Melatih Model
Jika Anda menambahkan dataset baru dalam folder data/, Anda bisa melatih ulang model dengan perintah berikut:

bash
Copy
Edit
python supervised.py
Atau jalankan manual di terminal Python:

python
Copy
Edit
from supervised import SmartAssistant
assistant = SmartAssistant(data_dir="data")
assistant.train()
Output model akan disimpan otomatis di folder model/.

🌍 Dukungan Multibahasa
Sistem ini mendukung berbagai bahasa karena menggunakan mBART-50. Anda bisa mengubah bahasa target dengan:

bash
Copy
Edit
/lang id_ID
atau saat bertanya:

makefile
Copy
Edit
id_ID: Apa itu evolusi?
🧠 Fitur Utama
✅ Semantic Search dengan typo correction.

✅ Training supervised dari berbagai format dataset.

✅ Multilingual Answer Generator (mBART).

✅ Interaksi real-time via antarmuka web modern.

✅ Otomatis menyimpan jawaban ke research/.

🛡️ Legal & Etika
Sistem ini dilindungi oleh lisensi edukatif terbatas. Harap gunakan dengan bijak untuk:

Riset akademik

Eksperimen AI pribadi

Belajar pengembangan chatbot dan NLP

Tidak untuk:

Propaganda

Penyebaran hoaks

Bisnis ilegal

Monetisasi tanpa izin

📬 Kontak
Email: marco.bakkara@gmail.com

GitHub: github.com/marcobakkara

© 2025 Marco Julius Andreas Bakkara. All rights reserved.









