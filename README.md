# 🤖 PDF Text Summarization and Q&A Chatbot

![Image](https://github.com/user-attachments/assets/edbbb58c-dc59-4b8e-8967-fb5767dd1fc1)

---


## 💡 프로젝트 개요
- **B2G 입찰지원 전문 컨설팅 스타트업   'Smash-RFP'**
- **RFP 내용을 분석·요약과 즉각적인 질의응답 기능을 제공하는 AI 챗봇 서비스 구현**

>매일 수백 건의 RFP가 쏟아지지만, 수십 페이지씩 일일이 읽기는 버겁습니다. 
><br>
>**Smash-RFP**는 관련 문서를 빠르게 찾아 실시간으로 질문에 답해 문서 검토 시간을 대폭 줄여드립니다.


---


## 🛠️ 기술 스택
#### < languages >
<div>
  <img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/nodedotjs-5FA04E?style=for-the-badge&logo=nodedotjs&logoColor=white">
</div>

#### < Frameworks >
<div>
  <img src="https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/langchain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white">
  <img src="https://img.shields.io/badge/huggingface-f1a805?style=for-the-badge&logo=huggingface&logoColor=ffffff">
  <img src="https://img.shields.io/badge/openai-050A52?style=for-the-badge&logo=openai&logoColor=white">
  <img src="https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/fastapi-009688?style=for-the-badge&logo=fastapi&logoColor=white">
</div>

#### < Tool >
<div>
  <img src="https://img.shields.io/badge/googlecloud-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white">
  <img src="https://img.shields.io/badge/googlecolab-F58320?style=for-the-badge&logo=googlecolab&logoColor=white">
  <img src="https://img.shields.io/badge/anaconda-44A833?style=for-the-badge&logo=anaconda&logoColor=white">
  <img src="https://img.shields.io/badge/visualstudio-0082fb?style=for-the-badge&logo=visualstudio&logoColor=white">
  <img src="https://img.shields.io/badge/notion-f2eee9?style=for-the-badge&logo=notion&logoColor=black">
</div>


---


## 📁 프로젝트 구조

```
AI-Engineer/
│
├── main.py              
├── environment.yaml     
├── data/                
├── src/
│   ├── loader/          
│   ├── retriever/       
│   ├── generator/       
│   └── vectordb/        
└── README.md            
```

---

## 📄 디렉토리 및 파일 설명

- **`main.py`**  :   전체 RAG 파이프라인 실행 코드
- **`environment.yaml`**  :   필요한 패키지가 정의된 Conda 가상환경 설정 파일입니다.
- **`data/`**  :   원문 문서, 처리된 텍스트, 생성된 벡터DB 등이 저장되는 디렉토리입니다.
- **`src/loader/`**  :   PDF 문서를 로딩하고, 텍스트 이미지 테이블 추출 및 의미 단위 청킹을 수행합니다.
- **`src/vectordb/`**  :   추출된 텍스트를 임베딩하고 Chroma 벡터DB로 저장합니다.
- **`src/retriever/`**  :   사용자의 질의에 대해 벡터 유사도를 기반으로 관련 문서를 검색합니다.
- **`src/generator/`**  :   검색된 문서를 기반으로 LLM을 활용해 최종 응답을 생성합니다.
- **`README.md`**  :   프로젝트에 대한 개요와 사용법을 설명한 문서입니다.

---

## 🦾 모델

[![Docling](https://img.shields.io/badge/Preprocessing-Docling-f2f1f0)](https://docling-project.github.io/docling/)

[![text-embedding-3-small](https://img.shields.io/badge/OpenAI-text--embedding--3--small-768c45)](https://platform.openai.com/docs/models/text-embedding-3-small)
[![BAAI/bge-m3](https://img.shields.io/badge/HF-BAAI%2Fbge--m3-768c45)](https://huggingface.co/BAAI/bge-m3)

[![Cross Encoder](https://img.shields.io/badge/Rerank-MiniLM--L--6--v2-f2f1f0)](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)

[![GPT 4.1 nano](https://img.shields.io/badge/LLM-GPT--4.1--nano-3f7373)](https://openai.com/index/gpt-4-1/)
[![MessagesPlaceholder](https://img.shields.io/badge/Prompt-MessagesPlaceholder-3f7373)](https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.chat.MessagesPlaceholder.html)

[![Qwen3-8B](https://img.shields.io/badge/LLM-Qwen3--8B-f2f1f0)](https://huggingface.co/Qwen/Qwen3-8B)
[![SFTTrainer](https://img.shields.io/badge/Trainer-SFTTrainer-f2f1f0)](https://huggingface.co/docs/trl/en/sft_trainer)

---

## 👤 팀원 소개
|       | 이름           | Notion                                                                                                     | 역할                                | 담당 업무                                                                                                    |
| :----- | :----- | :----- | :----- | :----- |
| 🧑‍💻 | **김어진** (팀장) | [협업일지 🚀](https://efficient-saver-88c.notion.site/Daily-238880186a9f8076a5cce4d0fb21e783?source=copy_link) | LLM & Prompt Engineer             | - LLM 기반 질의응답 설계 및 프롬프트 엔지니어링<br>- FrontEnd / BackEnd 개발                                                 |
| 👩‍💻 | **김하늘**      | [협업일지 🚀](https://www.notion.so/23806aa61633804eabd8f6ec466e277d)                                          | LLM & Prompt Engineer             | - LLM 기반 질의응답 설계 <br>- 프롬프트 엔지니어링 및 응답 최적화                                                               |
| 🧑‍💻 | **이대석**      | [협업일지 🚀](https://typhoon-friend-92d.notion.site/2378dd9e8ab380d082eedca33ac04203)                         | RFP Document Structuring Engineer | - RFP 문서 데이터 EDA 및 통계 분석<br>- 텍스트/표/이미지 등 구조적 요소 추출<br>- 마크다운 기반 청킹 파이프라인 설계 및 구현<br>- 각 청크에 대한 메타데이터 생성 |
| 🧑‍💻 | **이현도**      | [협업일지 🚀](https://www.notion.so/1fc5a0ce825e80278ec2e8b670db03fe)                                          | Vector Database Engineer          | - 임베딩 및 Vector DB 구축 (FAISS/ChromaDB)<br>- 쿼리 매칭 및 효율적 검색 구현                                             |
| 👩‍💻 | **전혜정**      | [협업일지 🚀](https://www.notion.so/23926cc555b4819180a5d94818e700b3?source=copy_link)                         | Retrieval Engineer                | - HyDE, Hybrid, rerank 기반 Retriever 개발<br>- 검색 성능 평가 및 개선                                                |


---

---

## 🤖 배포 링크
[🔗 Smash-RFP 챗 서비스 바로가기](https://codeit-rfp-ai.netlify.app/)

---

## 📜 보고서 PDF 링크
[🔗 Smash-RFP 최종 보고서](https://drive.google.com/drive/u/0/folders/1GEZdSbp-1DeFFZcOvhO3g0gv_ip3Zutk)

---
