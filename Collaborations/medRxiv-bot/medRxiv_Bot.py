import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

import requests
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

# ---------------- CONFIG ---------------- #

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("SEECHAT_API_KEY")

# Azure LLaMA 3 8B configuration
AZURE_LLAMA_KEY = os.getenv("AZURE_LLAMA_KEY")
AZURE_LLAMA_ENDPOINT = os.getenv("AZURE_LLAMA_ENDPOINT")

# Initialize the Azure client
llama_client = ChatCompletionsClient(
    endpoint=AZURE_LLAMA_ENDPOINT,
    credential=AzureKeyCredential(AZURE_LLAMA_KEY)
)


PROCESSED_TITLES_FILE = "processed_titles.txt"
processed_titles = set()
# ---------------------------------------- #


# ---------- STORAGE ---------- #
def load_titles() -> Set[str]:
    global processed_titles
    if os.path.exists(PROCESSED_TITLES_FILE):
        with open(PROCESSED_TITLES_FILE, "r", encoding="utf-8") as f:
            processed_titles = set(line.strip() for line in f)
    return processed_titles

def save_title(title) -> None:
    with open(PROCESSED_TITLES_FILE, "a", encoding="utf-8") as f:
        f.write(title.strip() + "\n")
# --------------------------------------- #


# ---------- LlaMA Extractors ---------- #        
# Function to extract problem statement using LLaMA 3 8B
def extract_problem(abstract: str) -> Optional[str]:
    try:
        system_prompt = """Act like a scientific research assistant. You are an expert in analyzing research abstracts and identifying core research problems.
                            Your task is to extract only the core problem or main issue the paper addresses.
                            Do not mention any methods, solutions, or results.
                            Use plain, simple language that any reader can understand.
                            Limit your output to a single sentence of no more than 20 words.
                            Do not include any explanation, intro, or extra text. Output only the problem statement.
                            """
        user_prompt = f"""Abstract: \"\"\"{abstract.strip()}\"\"\"
                             Extract the core research problem in ≤20 words.
                            Take a deep breath and work on this problem step-by-step."""

        response = llama_client.complete(
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt)
            ],
            max_tokens=100,
            stream=False
        )
        if not response.choices:
            logging.error("Empty response from LLaMA.")
            return None
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"LLaMA problem extraction error: {e}")
        return None


    try:
        system_prompt = f"""Act like a scientific research classifier. You are an expert at assigning research papers to standard academic disciplines based on their abstract content.
                            Your task is to classify the paper into one of the predefined categories below. Use the abstract and the original Catefory to make your decision.
                            If no category fits, return 'None'.
                            Return only one item: either a category from the list or 'None'. Do not explain.Do not list reasons. Do not include any bullet points.
                            Valid target categories:
                            {category_str}"""

        user_prompt = f"""Abstract: \"\"\"{abstract.strip()}\"\"\"
                            Original Category: {category}
                            Based on the abstract, map this paper to the most appropriate field from the target category list.
                            If no match is appropriate, return 'None'.
                            """

        response = llama_client.complete(
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt)
            ],
            max_tokens=30,
            stream=False
        )
        if not response.choices:
            logging.error("Empty category response from LLaMA.")
            return None
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"LLaMA category extraction error: {e}")
        return None
# --------------------------------------- #


# ---------- See Chat API INTEGRATION ---------- #
def create_hypothesis(problem:str, field1:str, category:str, source:str) -> Optional[str]:
    try:
        url = "https://api.staging.seechat.ai/idea/create_hypothesis"
        headers = {
            "X-API-Key": API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "problem": problem,
            "research_topics": [category],
            "field_of_study_1": field1,
            "data_source": source,
            "is_private": False
        }
        hypothesis_id = None
        with requests.post(url, headers=headers, json=payload, stream=True) as response:
            buffer = ""
            for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                buffer += chunk
                matches = re.findall(r'\{.*?\}(?=\{|\Z)', buffer)
                for match in matches:
                    try:
                        data = json.loads(match)
                        if data.get("data", {}).get("type") == "metadata":
                            hypothesis_id = data["data"]["content"]["hypothesis_id"]
                    except Exception:
                        continue
                buffer = ""
    except Exception as e:
        logging.error(f"Create hypothesis failed: {e}")
    return hypothesis_id

def edit_hypothesis(hypothesis_id:str, title:str, summary:str) -> bool:
    try:
        url = "https://api.staging.seechat.ai/idea/edit_hypothesis"
        headers = {
            "X-API-Key": API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "hypothesis_id": hypothesis_id,
            "title": title,
            "idea_summary": summary,
            "is_private": False
        }
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            logging.info("Hypothesis edited successfully.")
            return True
        logging.error(f"Edit failed ({response.status_code}): {response.text}")
    except Exception as e:
        logging.error(f"Edit hypothesis error: {e}")
    return False

def process_paper(title:str, abstract:str, summary:str, category:str, source:str) -> None:
    if title in processed_titles:
        logging.info(f"[Skipped] Already processed: {title}")
        return
    try:
        problem = extract_problem(abstract)

        print(f'Problem Statement: {problem}')

        if not problem:
            return
        field1 = "Medicine" 
        hypothesis_id = create_hypothesis(problem, field1, category, source)
        if hypothesis_id and edit_hypothesis(hypothesis_id, title, summary):
            print(f'Idea URL: https://staging.seechat.ai/idea/{hypothesis_id}')
            save_title(title)
            processed_titles.add(title)
            logging.info(f"[✓] Processed: {title}")
    except Exception as e:
        logging.error(f"Process paper error: {e}")
# ------------------------------------- #


# ---------- FETCH FUNCTIONS ---------- #
def fetch_medrxiv_by_date_range(start_date:Optional[str]=None, end_date:Optional[str]=None) -> List[Dict]:
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

    url = f"https://api.biorxiv.org/details/medrxiv/{start_date}/{end_date}"
    params = {"limit": 1000}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get("collection", [])
    except Exception as e:
        logging.error(f"medRxiv fetch error: {e}")
        return []
# ------------------------------------- #

def format_results_paragraph(text):
   
    sentences = re.split(r'(?<=[.?!])(?:\s+|$)|\n+', text)
    # Use a fixed emoji or cycle through a few
    emojis = ['- 📌', '- 🔹', '- ➤', '- ✅', '- 📝',  '- 📉']
    print("DEBUG SENTENCES:", sentences) 
    bullets = []
    for i, sentence in enumerate(sentences):
        if sentence:
            emoji = emojis[i % len(emojis)]
            bullets.append(f"{emoji} {sentence.strip()}")
    
    return "\n".join(bullets)


def format_abstract_paragraphs(abstract_text):
    section_labels = [
        "Background", "Outcomes of interest", "Objectives", "Objective",
        "Methods and Results", "Methods", "Results", "Conclusions", "Conclusion", "Findings"
    ]
    sentences = re.split(r'(?<=[.?!])\s+', abstract_text)
    formatted = []
    i = 0
    n = len(sentences)
    while i < n:
        sentence = sentences[i]
        # Check for section label at the start (with or without colon)
        match_label = re.match(r'^(' + '|'.join(section_labels) + r')(:)?(\s*)(.*)', sentence, re.IGNORECASE)
        if match_label:
            label = match_label.group(1)
            bolded = f"**{label}:**"
            rest = match_label.group(4).strip()
            formatted.append("")
            formatted.append(bolded)
            formatted.append("")
            if label.lower() == "results":
                # Collect all sentences after "Results:" until next label or end
                results_block = []
                if rest:
                    results_block.append(rest)
                i += 1
                while i < n:
                    next_sentence = sentences[i]
                    next_label = re.match(r'^(' + '|'.join(section_labels) + r')(:)?(\s*)', next_sentence, re.IGNORECASE)
                    if next_label:
                        break
                    results_block.append(next_sentence)
                    i += 1
                formatted.extend(format_results_paragraph(' '.join(results_block)).splitlines())
                continue
            else:
                if rest:
                    formatted.append(rest)
            i += 1
            continue

        # Check for phrase ending with ':'
        match_colon = re.match(r'^(.*?:)(\s*)(.*)', sentence)
        match_upper = re.match(r'^([A-Z]{2,})(\s+)(.*)', sentence)
        if match_colon:
            bolded = f"**{match_colon.group(1)}**"
            rest = match_colon.group(3).strip()
           
            formatted.append(bolded)
            
            if rest:
                formatted.append(rest)
        elif match_upper:
            bolded = f"**{match_upper.group(1)}**"
            rest = match_upper.group(3).strip()
           
            formatted.append(bolded)
           
            if rest:
                formatted.append(rest)
        else:
            formatted.append(sentence)
        i += 1
    return '\n'.join(formatted)



# ---------- CONSOLE APP ---------- #
def main() -> None:
    load_titles()
    logging.info("Scientific Paper Hypothesis Assistant")
    logging.info("Fetching medRxiv papers...")
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")
    papers = fetch_medrxiv_by_date_range(start_date, end_date)
    success_count = 0
    for item in papers:
        title = item['title']
        abstract = item['abstract']
        abstract_md = f"## 📄 Abstract\n\n{format_abstract_paragraphs(abstract)}"
        doi = item['doi']
        link = f"https://www.medrxiv.org/content/{item['doi']}v{item['version']}"
        date = item['date']
        category = item['category']
        authors = [name.strip() for name in item['authors'].split(';')]
        authors_list = "\n".join(f"- {author}" for author in authors)
        summary = f"""# 🧪 {title}\n\n---\n\n{abstract_md}\n\n---\n\n## 👩‍🔬 Authors\n{authors_list}\n\n## 🧷 DOI\n[{doi}](https://doi.org/{doi})\n\n## 🗓️ Published Date\n**{date}**\n\n## 🔗 [View Full Article]({link})\n"""
        print('-' * 60)
        logging.info(f"Title: {title}")
        previous_count = len(processed_titles)
        process_paper(title, abstract, summary, category, "medRxiv")
        if len(processed_titles) > previous_count:
            success_count += 1

    print(f"\n✅ Successfully processed {success_count} new Papers.\n")


if __name__ == "__main__":
    main()
