import json
import logging
import re
import os
import time
from datetime import datetime, timedelta
from typing import Optional, List, Set, Dict
from tqdm import tqdm
import requests
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

# ---------------- CONFIG ---------------- #
# Load environment variables from .env file

load_dotenv()
API_KEY = os.getenv("SEECHAT_API_KEY")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

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

# Minimum H-index required for an author for the paper to be processed
MIN_H_INDEX = 10 

# Maximum hypotheses to generate
max_hypotheses = 10 
# ---------------------------------------- #


# ---------- STORAGE ---------- #
def load_titles() -> Set[str]:
    """Loads previously processed paper titles from a file."""
    global processed_titles
    if os.path.exists(PROCESSED_TITLES_FILE):
        with open(PROCESSED_TITLES_FILE, "r", encoding="utf-8") as f:
            processed_titles = set(line.strip() for line in f)
    return processed_titles

def save_title(title) -> None:
    """Saves a processed paper title to a file."""
    with open(PROCESSED_TITLES_FILE, "a", encoding="utf-8") as f:
        f.write(title.strip() + "\n")
# --------------------------------------- #


# ---------- LlaMA Extractors ---------- #        
# Function to extract problem statement using LLaMA 3 8B
def extract_problem(abstract: str) -> Optional[str]:
    """
    Extracts the core problem statement from a paper abstract using LLaMA 3 8B.
    Limits the output to a single sentence of no more than 20 words.
    """
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
# --------------------------------------- #


# ---------- Semantic Scholar API for H-index ---------- #
def get_authors_from_doi(doi: str) -> Optional[List[Dict]]:
    """
    Fetches authors' basic information (including authorId) for a given DOI.
    Returns a list of author dictionaries or None if not found/error.
    """
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY}
    params = {
        "fields": "authors"
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status() # Raise an exception for HTTP errors
        data = response.json()
        time.sleep(1)  # Sleep for 1 second after each successful request
        return data.get("authors", [])
    except requests.exceptions.RequestException as e:
        logging.warning(f"Error fetching authors for DOI {doi} from Semantic Scholar: {e}")
        return None
    except json.JSONDecodeError:
        logging.warning(f"Error decoding JSON for DOI {doi} from Semantic Scholar.")
        return None

def get_author_hindex(author_id: str) -> Optional[int]:
    """
    Fetches the H-index for a given Semantic Scholar author ID.
    Returns the H-index as an integer or None if not found/error.
    """
    url = f"https://api.semanticscholar.org/graph/v1/author/{author_id}"
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY}
    params = {
        "fields": "hIndex"
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status() # Raise an exception for HTTP errors
        data = response.json()
        time.sleep(1)  # Sleep for 1 second after each successful request
        return data.get("hIndex")
    except requests.exceptions.RequestException as e:
        logging.warning(f"Error fetching H-index for author ID {author_id} from Semantic Scholar: {e}")
        return None
    except json.JSONDecodeError:
        logging.warning(f"Error decoding JSON for author ID {author_id} from Semantic Scholar.")
        return None

def rank_valid_papers_by_hindex(papers: List[Dict], min_h_index: int, max_papers: int) -> List[Dict]:
    """
    Filters papers where at least one author has H-index ≥ min_h_index,
    and returns the top N papers sorted by highest author H-index.
    """
    valid_papers = []
    print(f"\n🔍 Checking H-index for {len(papers)} papers...")
    for paper in tqdm(papers, desc="Processing papers"):
        title = paper.get("title")
        doi = paper.get("doi")

        if not title or not doi or title in processed_titles:
            continue

        authors_data = get_authors_from_doi(doi)
        if not authors_data:
            continue

        max_h = -1
        for author in authors_data:
            author_id = author.get("authorId")
            if author_id:
                h = get_author_hindex(author_id)
                if h is not None:
                    if h >= min_h_index:
                        paper["max_h_index"] = h
                        valid_papers.append(paper)
                        break  # Stop checking more authors as soon as one passes            

    # Sort by H-index descending
    valid_papers.sort(key=lambda x: x["max_h_index"], reverse=True)

    # Return top N
    return valid_papers[:max_papers]
# ---------------------------------------------------- #


# ---------- See Chat API INTEGRATION ---------- #
def create_hypothesis(problem:str, field1:str, category:str, source:str) -> Optional[str]:
    """Creates a new hypothesis in the Seechat API."""
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
    """Edits an existing hypothesis in the Seechat API."""
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

def process_paper(title:str, abstract:str, summary:str, doi:str, category:str, source:str) -> None:
    """
    Processes a single paper: checks H-index, extracts problem, creates/edits hypothesis.
    """
    if title in processed_titles:
        logging.info(f"[Skipped] Already processed: {title}")
        return
    
    logging.info(f"Proceeding to process paper '{title}'.") 
    try:
        problem = extract_problem(abstract)
        print(f'Problem Statement: {problem}')
        if not problem:
            logging.warning(f"Skipping paper '{title}' due to failure in problem statement extraction.")
            return
        field1 = "Medicine" 
        hypothesis_id = create_hypothesis(problem, field1, category, source)
        if hypothesis_id and edit_hypothesis(hypothesis_id, title, summary):
            print(f'Idea URL: https://staging.seechat.ai/idea/{hypothesis_id}')
            save_title(title)
            processed_titles.add(title)
            logging.info(f"[✓] Processed: {title}")
        else:
            logging.error(f"Failed to create or edit hypothesis for '{title}'.")     
    except Exception as e:
        logging.error(f"Process paper error: {e}")
# ------------------------------------- #


#------------ MARKDOWN FORMATTING ----------------#
def format_results_paragraph(text):
   
    sentences = re.split(r'(?<=[.?!])(?:\s+|$)|\n+', text)
    # Use a fixed emoji or cycle through a few
    emojis = ['- 📌', '- 🔹', '- ➤', '- ✅', '- 📝',  '- 📉']
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
#-------------------------------------------------#


# ---------- FETCH FUNCTIONS ---------- #
def fetch_medrxiv(start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict]:
    """
    Fetches papers from medRxiv within a specified period or date range,
    optionally filtered by a search term. Handles API pagination.
    """
    base_url = f"https://api.biorxiv.org/details/medrxiv"
    # Use date range if no period is specified
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        # Default to last 7 days if no period or start_date is provided
        start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    interval = f"{start_date}/{end_date}"

    # The API might paginate, so we need to fetch multiple pages
    all_fetched_papers = []
    cursor = 0

    logging.info(f"Fetching papers from medRxiv using interval: {interval}...")

    while True:
        url = f"{base_url}/{interval}/{cursor}"    
        
        try:
            response = requests.get(url, timeout=30) # Increased timeout for potentially large responses
            response.raise_for_status()
            data = response.json()

            current_page_papers = data.get("collection", [])
            if not current_page_papers:
                break # No more papers on this page, or end of collection

            all_fetched_papers.extend(current_page_papers)

            # Check for pagination info in 'messages'
            messages = data.get('messages')
            if messages:
                # The cursor in the message seems to indicate the total fetched so far
                # Or the start of the next batch if there are more results
                message = messages[0] # Assuming there's only one message with cursor info
                total_records = message.get('total')
                next_cursor_start = message.get('cursor') # This seems to be the current cursor position

                if total_records is not None and next_cursor_start is not None:
                    try:
                        total_records = int(total_records)
                        next_cursor_start = int(next_cursor_start)

                        if (next_cursor_start + len(current_page_papers)) >= total_records:
                            # We've fetched all available papers
                            break
                        else:
                            cursor = next_cursor_start + len(current_page_papers) # Move cursor to next page's start
                    except ValueError:
                        logging.warning(f"Could not convert total_records ({total_records}) or next_cursor_start ({next_cursor_start}) to int. Stopping pagination.")
                        break # Stop if conversion fails
                else:
                    # If 'total' or 'cursor' not in message, assume no more pages
                    break
            else:
                # If no messages, assume no more pages
                break

        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching data from medrxiv ({url}): {e}")
            break
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON response from medrxiv ({url}).")
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred while fetching from medrxiv: {e}")
            break
    
    logging.info(f"Returning all {len(all_fetched_papers)} fetched papers.")
    return all_fetched_papers
# ------------------------------------- #


# ---------- CONSOLE APP ---------- #
def main() -> None:
    """Main function to run the paper fetching and processing application."""
    load_titles()
    logging.info("Scientific Paper Hypothesis Assistant")
    logging.info("Fetching medRxiv papers...")
    # Offer period or date range
    start_date_input = None 
    end_date_input = None 
   
    papers = fetch_medrxiv(start_date=start_date_input if start_date_input else None,
                           end_date=end_date_input if end_date_input else None)
        
    if not papers:
        logging.info("No papers found matching your criteria.")
        return
    
    success_count = 0
    papers = rank_valid_papers_by_hindex(papers, MIN_H_INDEX, max_hypotheses)
    print(f"\nFound {len(papers)} top papers eligible for hypothesis generation.")

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
        process_paper(title, abstract, summary, doi, category, "medRxiv")
        if len(processed_titles) > previous_count:
            success_count += 1

    print(f"\n✅ Successfully processed {success_count} new Papers.\n")


if __name__ == "__main__":
    main()