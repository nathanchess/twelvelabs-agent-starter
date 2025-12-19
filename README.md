# 🎬 TwelveLabs Video RAG Agent

> Build an AI agent that understands and searches videos using natural language — powered by TwelveLabs Marengo embeddings and LangChain.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TwelveLabs](https://img.shields.io/badge/TwelveLabs-Marengo_3.0-purple.svg)](https://twelvelabs.io/)
[![LangChain](https://img.shields.io/badge/LangChain-Agent-green.svg)](https://langchain.com/)
[![Tutorial](https://img.shields.io/badge/Tutorial-Hands--On-orange.svg)](#-tutorial-exercises)

> **📚 This is a hands-on tutorial!** The codebase includes 3 exercises for you to complete. Look for `TODO` comments in the code.

---

## 📺 Video Tutorial

<p align="center">
  <a href="https://youtu.be/hnJr6rmoa9A">
    <img src="https://img.youtube.com/vi/hnJr6rmoa9A/maxresdefault.jpg" alt="Video Tutorial" width="600">
  </a>
</p>

<p align="center">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/hnJr6rmoa9A" title="TwelveLabs Video RAG Agent Tutorial" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</p>

<p align="center">
  <em>👆 Watch the full tutorial on YouTube</em>
</p>

---

## 🌟 What is This?

This is a **hands-on tutorial** that teaches you how to build a **video semantic search agent** using TwelveLabs Marengo embeddings. By completing 3 exercises, you'll learn how to:

- 🎥 **Embed videos** into vector representations using TwelveLabs Marengo 3.0
- 🔍 **Search videos with natural language** queries (e.g., "find clips with a cheetah running")
- 🤖 **Use an AI agent** that intelligently decides which tools to call
- ⏱️ **Retrieve specific timestamps** where relevant content appears

Perfect for learning multimodal AI, building video libraries, content search, or any application where you need to find specific moments in video content.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎬 **Video Embedding** | Convert any video into 512-dimensional vectors with timestamp segments |
| 💬 **Natural Language Search** | Search your video library using plain English queries |
| 🧠 **Multimodal Understanding** | Marengo 3.0 understands visual content, motion, and context |
| 🤖 **Agentic Architecture** | LangChain agent autonomously selects and executes the right tools |
| 📍 **Timestamp Retrieval** | Get exact start/end times for relevant video segments |
| 💾 **Simple Vector Store** | JSON-based storage for easy understanding and modification |

---

## 📝 Tutorial Exercises

This project is designed as a **hands-on tutorial**. You'll implement 3 key functions to understand how TwelveLabs embeddings work.

### Exercise 1: Video Embedding
**File:** `tools/create_video_embed.py` (line 79)

Implement the TwelveLabs API call to create video embeddings:

```python
# TODO: Exercise 1 - Create video embedding using TwelveLabs API
# 
# Use twelvelabs_client.embed.v_2.create() with:
#   - input_type: 'video'
#   - model_name: 'marengo3.0'  
#   - video: VideoInputRequest with MediaSource containing base64 string
#
# Hint: The base64 video string is in variable `base64_video_string`

response = None  # Replace with TwelveLabs API call
```

<details>
<summary>💡 Click to see solution</summary>

```python
response = twelvelabs_client.embed.v_2.create(
    input_type='video',
    model_name='marengo3.0',
    video=VideoInputRequest(
        media_source=MediaSource(
            base_64_string=base64_video_string
        ),
    ),
)
```

</details>

---

### Exercise 2: Text Embedding
**File:** `tools/text_rag.py` (line 21)

Implement the TwelveLabs API call to create text embeddings:

```python
# TODO: Exercise 2 - Create text embedding using TwelveLabs API
#
# Use twelvelabs_client.embed.create() with:
#   - model_name: 'marengo3.0'
#   - text: the input text parameter
#
# Return: response.text_embedding.segments[0].float_

pass  # Replace with TwelveLabs API call and return statement
```

<details>
<summary>💡 Click to see solution</summary>

```python
response = twelvelabs_client.embed.create(
    model_name='marengo3.0',
    text=text,
)

if response and response.text_embedding is not None:
    return response.text_embedding.segments[0].float_
else:
    raise ValueError("Failed to create text embed.")
```

</details>

---

### Exercise 3: Cosine Similarity
**File:** `tools/text_rag.py` (line 73)

Implement the cosine similarity calculation between text and video embeddings:

```python
# TODO: Exercise 3 - Calculate cosine similarity
#
# Cosine similarity = dot(A, B) / (||A|| * ||B||)
#
# Use numpy: np.dot(), np.linalg.norm()
# Variables: text_embedding, video_embedding_array

similarity = 0.0  # Replace with cosine similarity calculation
```

<details>
<summary>💡 Click to see solution</summary>

```python
text_embedding_norm = np.linalg.norm(text_embedding)
video_embedding_norm = np.linalg.norm(video_embedding_array)
product = np.dot(text_embedding, video_embedding_array)
similarity = product / (text_embedding_norm * video_embedding_norm)
```

</details>

---

### ✅ Completed Version

Want to see the finished code? Check out the `completed/` folder for working implementations of all exercises.

---

## 🏗️ Technical Architecture

<img width="5771" height="2259" alt="TwelveLabs Embedding Agent Diagram - Page 1" src="https://github.com/user-attachments/assets/d520685a-63b6-4c36-bb2f-d6ffb3a4409d" />

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           USER INPUT                                 │
│                    "Find videos about cheetahs"                      │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        LANGCHAIN AGENT                               │
│                   (Claude 3.5 Sonnet via AWS Bedrock)                │
│                                                                      │
│   ┌─────────────────┐              ┌─────────────────┐              │
│   │  Tool Selection │───────────────│  Response Gen   │              │
│   └────────┬────────┘              └─────────────────┘              │
└────────────┼────────────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────────┐
│                         TOOLS LAYER                                 │
│                                                                     │
│  ┌──────────────────────┐      ┌──────────────────────┐            │
│  │  create_video_embed  │      │      text_rag        │            │
│  │                      │      │                      │            │
│  │  • Read video file   │      │  • Embed query text  │            │
│  │  • Encode to base64  │      │  • Load vector DB    │            │
│  │  • Call Marengo API  │      │  • Cosine similarity │            │
│  │  • Store embeddings  │      │  • Return top-k      │            │
│  └──────────┬───────────┘      └──────────┬───────────┘            │
└─────────────┼──────────────────────────────┼───────────────────────┘
              │                              │
              ▼                              ▼
┌─────────────────────────┐    ┌─────────────────────────────────────┐
│    TWELVELABS API       │    │         VECTOR DATABASE             │
│                         │    │          (embed.json)               │
│  ┌───────────────────┐  │    │                                     │
│  │   Marengo 3.0     │  │    │  {                                  │
│  │                   │  │    │    "video_0.00_8.00": {             │
│  │  • Video Embed    │  │    │      "embedding": [512 floats],     │
│  │  • Text Embed     │  │    │      "start_time": 0.0,             │
│  │  • 512-dim output │  │    │      "end_time": 8.0                │
│  └───────────────────┘  │    │    }                                │
└─────────────────────────┘    │  }                                  │
                               └─────────────────────────────────────┘
```

### Key Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **LLM** | Claude 3.5 Sonnet (AWS Bedrock) | Reasoning, tool selection, response generation |
| **Agent Framework** | LangChain + LangGraph | Tool orchestration and conversation management |
| **Embedding Model** | TwelveLabs Marengo 3.0 | Multimodal video & text embeddings |
| **Vector Store** | JSON file | Simple, transparent embedding storage |
| **Similarity Search** | NumPy (cosine similarity) | Find relevant video segments |

---

## 📋 Prerequisites

Before you begin, ensure you have:

- **Python 3.10+** installed
- **TwelveLabs API Key** — [Get one free at twelvelabs.io](https://twelvelabs.io/)
- **AWS Account** with Bedrock access (for Claude 3.5 Sonnet)
- Some video files to embed (MP4 format recommended)

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/twelvelabs-video-rag-agent.git
cd twelvelabs-video-rag-agent
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
TWELVELABS_API_KEY=your_twelvelabs_api_key_here
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
VECTOR_DB_FILE=./db/embed.json
```

### 5. Run the Agent

```bash
python main.py
```

---

## 💡 Usage Examples

Once the agent is running, try these commands:

### Embed a Video
```
Enter query: Please embed videos/cheetah.mp4 and store it in the database
```

### Search for Content
```
Enter query: Find videos that show fast animals running
```

### Semantic Search
```
Enter query: Which video segments contain wildlife in nature?
```

### Exit
```
Enter query: exit
```

---

## 📁 Project Structure

```
twelvelabs-video-rag-agent/
├── main.py                 # Entry point - agent setup and CLI loop
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── .gitignore
│
├── tools/                  # LangChain tools (with TODO exercises)
│   ├── __init__.py
│   ├── create_video_embed.py   # Exercise 1: Video embedding
│   └── text_rag.py             # Exercise 2 & 3: Text embedding + similarity
│
├── completed/              # ✅ Completed solutions (reference)
│   └── tools/
│       ├── create_video_embed.py
│       └── text_rag.py
│
├── db/
│   └── embed.json          # Vector database (JSON)
│
└── videos/                 # Sample videos (add your own)
    ├── cheetah.mp4
    └── rhino.mp4
```

---

## 🔧 How It Works

> **Note:** The code snippets below show the completed implementations. In the tutorial, you'll implement these yourself!

### 1. Video Embedding (`create_video_embed.py`) — Exercise 1

```python
response = twelvelabs_client.embed.v_2.create(
    input_type='video',
    model_name='marengo3.0',
    video=VideoInputRequest(
        media_source=MediaSource(base_64_string=base64_video_string)
    ),
)
```

- Reads video file and encodes to base64
- Sends to TwelveLabs Marengo 3.0 API
- Receives multiple 512-dimensional embeddings (one per segment)
- Stores embeddings with timestamps in JSON database

### 2. Text-to-Video Search (`text_rag.py`) — Exercise 2 & 3

```python
# Embed the search query
text_embedding = twelvelabs_client.embed.create(
    model_name='marengo3.0',
    text=query,
)

# Calculate cosine similarity with all video embeddings
similarity = np.dot(text_embedding, video_embedding) / (
    np.linalg.norm(text_embedding) * np.linalg.norm(video_embedding)
)
```

- Converts text query to same 512-dim embedding space
- Computes cosine similarity against all stored video segments
- Returns top-k most similar video segments with timestamps

### 3. Agent Orchestration (`main.py`)

The LangChain agent uses Claude 3.5 Sonnet to:
- Understand user intent from natural language
- Select appropriate tool(s) to call
- Execute tools and format responses
- Maintain conversation context

---

## 🎯 Use Cases

- **Media Asset Management** — Search large video libraries with natural language
- **Content Moderation** — Find specific types of content in video archives
- **Video Summarization** — Identify key moments and segments
- **Educational Platforms** — Help students find relevant lecture segments
- **Sports Analysis** — Locate specific plays or moments in game footage

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [TwelveLabs](https://twelvelabs.io/) for the incredible Marengo embedding model
- [LangChain](https://langchain.com/) for the agent framework
- [AWS Bedrock](https://aws.amazon.com/bedrock/) for Claude 3.5 Sonnet access

---

<p align="center">
  Made with ❤️ for the TwelveLabs Developer Community
</p>

