# 🧠 Scientific Paper Hypothesis Generator

## 🔬 Overview
This Python application fetches recent papers from [medRxiv](https://www.medrxiv.org/), filters them based on author H-index using the [Semantic Scholar API](https://api.semanticscholar.org/), extracts the core research problem using Azure-hosted LLaMA 3 8B, and submits hypotheses to the [SeeChat](https://seechat.ai/) platform.

## 🔁 Features
- Filters based on author H-index threshold (≥10)
- Uses LLaMA 3 8B to extract concise problem statements
- Automatically submits hypotheses to SeeChat
- Saves processed titles to avoid duplication

## 🛠 Setup Locally
1. Clone or download the project repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a .env file:
    ```bash
    SEECHAT_API_KEY=your_api_key_here
    SEMANTIC_SCHOLAR_API_KEY=your_semantic_key_here
    AZURE_LLAMA_KEY=your_llama_key_here
    AZURE_LLAMA_ENDPOINT=https://your-azure-endpoint/
    ```
4. Ensure the `processed_titles.txt` file is writable in the project directory for storing processed paper titles.
5. Run the script:
    ```bash
      python medRxiv_Bot.py
    ```  

## 📄 License
This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.

## 📬 Contact
For issues or inquiries, please contact the project maintainer via the repository's issue tracker.
