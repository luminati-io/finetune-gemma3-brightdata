# ✨ finetune-gemma3-bright

> Automate the collection, processing, and transformation of Trustpilot customer reviews into high-quality QA datasets for fine-tuning [Gemma 3](https://ai.google.dev/gemma) and similar LLMs.

---

## 🚀 Overview

This repo provides a full workflow to:

1. **Collect customer reviews** from Trustpilot using the [Bright Data API](https://docs.brightdata.com/api-reference/introduction).
2. **Convert reviews to Markdown** for easy inspection or sharing.
3. **Chunk and refine reviews** with OpenAI LLMs for coherence and quality.
4. **Generate insightful QA pairs** suitable for fine-tuning customer service or product feedback models.
5. **Upload the dataset** directly to [Hugging Face Hub](https://huggingface.co/) with a train/test split.

---

## 🏗️ Directory Structure

- `collect_reviews.py` : Download reviews from Trustpilot via Bright Data.
- `json_to_md.py` : Convert review JSON to Markdown format.
- `md_chunker.py` : Split/clean markdown into chunks and enhance with LLM.
- `generate_qa_pairs.py` : Use OpenAI to generate question-answer pairs from review chunks.
- `upload_to_hf.py` : Validate and upload results as Hugging Face dataset.

---

## 📦 Installation

**Requires:** Python 3.8+  
**Recommended:** [virtualenv](https://docs.python.org/3/tutorial/venv.html)

```bash
git clone https://github.com/youraccount/finetune-gemma3-bright.git
cd finetune-gemma3-bright
pip install -r requirements.txt
```

### 🔑 Environment Setup

1. Create a `.env` file and add:

   ```
   OPENAI_API_KEY=your-openai-key
   HF_TOKEN=your-huggingface-key
   ```

2. Update your Bright Data API key and dataset ID in `collect_reviews.py`.

---

## 📚 Key Dependencies

- [requests](https://docs.python-requests.org/) — robust HTTP requests
- [python-dotenv](https://github.com/theskumar/python-dotenv) — manage secrets
- [openai](https://platform.openai.com/docs/api-reference) — LLM completion and chat API
- [tenacity](https://tenacity.readthedocs.io/) — retry mechanisms
- [langchain-text-splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/) — advanced text chunking
- [datasets](https://huggingface.co/docs/datasets/) — fast data handling for ML/NLP
- [huggingface-hub](https://huggingface.co/docs/huggingface_hub/) — seamless model/data uploads

---

## 🔄 Review Collection Flow

1. **Start by collecting reviews**

   ```bash
   python collect_reviews.py
   ```

   - Triggers a Bright Data [snapshot collection](https://docs.brightdata.com/api-reference/web-scraper-api/trigger-a-collection) of a Trustpilot page.
   - Polls until reviews are ready, then saves to `trustpilot_reviews.json`.

2. **Convert JSON reviews to pretty Markdown**

   ```bash
   python json_to_md.py --input trustpilot_reviews.json --output trustpilot_reviews.md
   ```

3. **Chunk markdown, improve coherence with GPT, save as JSONL**

   ```bash
   python md_chunker.py --input trustpilot_reviews.md --output trustpilot_reviews_chunks.jsonl
   ```

4. **Generate Question-Answer pairs using OpenAI**

   ```bash
   python generate_qa_pairs.py --input trustpilot_reviews_chunks.jsonl --output trustpilot_qa_pairs.json
   ```

5. **Upload dataset to Hugging Face Hub 🚀**

   ```bash
   python upload_to_hf.py --input trustpilot_qa_pairs.json --repo yourusername/trustpilot-reviews-qa-dataset
   ```

---

## 📝 Example Workflow

```bash
# Step 1: Collect data
python collect_reviews.py

# Step 2: Markdown conversion
python json_to_md.py

# Step 3: Chunk and enhance
python md_chunker.py

# Step 4: QA generation
python generate_qa_pairs.py

# Step 5: Upload to Hugging Face
python upload_to_hf.py
```

---

## 💡 Extending & Customizing

- **Change Target Company:**  
  Edit `TARGET_URL` in `collect_reviews.py`.

- **Replace Keys:**  
  Put your API keys/secrets in `.env`.

- **Adjust Chunk/Overlap Sizes:**  
  Use `--chunk-size` and `--chunk-overlap` flags in chunker script.

- **Customize QA Prompt:**  
  Modify `SYSTEM_PROMPT` in `generate_qa_pairs.py`.

---

## 🌎 Helpful Links

- Bright Data API Docs: [API Reference](https://docs.brightdata.com/api-reference/introduction)
- OpenAI API Docs: [Reference](https://platform.openai.com/docs)
- Hugging Face Hub: [Upload Dataset](https://huggingface.co/docs/datasets/upload_dataset)
- LangChain Text Splitters: [Docs](https://python.langchain.com/docs/how_to/#text-splitters)

---

## 📝 License

MIT

---

**Happy Finetuning!** 🎉
