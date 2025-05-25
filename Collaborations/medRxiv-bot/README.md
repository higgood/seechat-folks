# Scientific Paper Hypothesis Generator

## Overview
This Python application fetches recent research papers from the medRxiv API, extracts core research problems from their abstracts using Azure's LLaMA 3 8B model, and generates hypotheses via the SeeChat API. It processes papers within a specified date range, stores processed titles to avoid duplication, and logs activities for transparency.

## Features
- **Paper Fetching**: Retrieves research papers from medRxiv based on user-defined start and end dates.
- **Problem Extraction**: Uses Azure's LLaMA 3 8B model to extract concise problem statements from paper abstracts.
- **Hypothesis Generation**: Integrates with the SeeChat API to create and edit hypotheses based on extracted problems.
- **Duplicate Prevention**: Tracks processed paper titles in a local file to avoid redundant processing.
- **Logging**: Implements detailed logging for monitoring and debugging.

## Prerequisites
- **Python**: Version 3.8 or higher.
- **Dependencies**: Install required packages using:
  ```bash
  pip install requests azure-ai-inference azure-core python-dotenv
  ```
- **Environment Variables**:
  - `SEECHAT_API_KEY`: API key for the SeeChat API.
  - `AZURE_LLAMA_KEY`: Azure credential key for LLaMA 3 8B.
  - `AZURE_LLAMA_ENDPOINT`: Endpoint URL for the Azure LLaMA model.
  Create a `.env` file in the project root with the following format:
  ```plaintext
  SEECHAT_API_KEY=your_seechat_api_key
  AZURE_LLAMA_KEY=your_azure_llama_key
  AZURE_LLAMA_ENDPOINT=your_azure_llama_endpoint
  ```

## Installation
1. Clone or download the project repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up the `.env` file with the required API keys and endpoint.
4. Ensure the `processed_titles.txt` file is writable in the project directory for storing processed paper titles.

## Usage
1. Run the script:
   ```bash
   python main.py
   ```
2. Enter the start and end dates (format: `YYYY-MM-DD`) when prompted to fetch papers from medRxiv.
3. The application will:
   - Fetch papers within the specified date range.
   - Extract problem statements using the LLaMA model.
   - Generate and edit hypotheses via the SeeChat API.
   - Display the problem statement and a URL to the generated hypothesis for each successfully processed paper.
   - Log progress and errors to the console.
4. Successfully processed papers are saved to `processed_titles.txt` to prevent reprocessing.

## Code Structure
- **Configuration**: Loads environment variables and initializes the Azure LLaMA client.
- **Storage**: Manages a local file (`processed_titles.txt`) to track processed paper titles.
- **LLaMA Extractors**: Uses Azure's LLaMA 3 8B model to extract problem statements from abstracts.
- **SeeChat API Integration**: Creates and edits hypotheses using the SeeChat API.
- **Fetch Functions**: Retrieves paper metadata from the medRxiv API.
- **Main Console App**: Orchestrates the workflow, prompting for dates and processing papers.

## Logging
- The application uses Python's `logging` module to output informational and error messages.
- Logs are formatted as `LEVEL - MESSAGE` (e.g., `INFO - Processed: [Paper Title]`).
- Azure HTTP logging is suppressed to reduce noise.

## Error Handling
- Robust error handling is implemented for API calls, file operations, and LLaMA responses.
- Errors are logged with descriptive messages to aid debugging.
- The application skips papers with failed processing and continues with the next paper.

## Limitations
- Requires stable internet access for API calls.
- Dependent on the availability of the medRxiv, Azure, and SeeChat APIs.
- Problem extraction is limited to 20 words due to the LLaMA prompt configuration.
- The field of study is currently hardcoded to "Medicine" for hypothesis creation.

## Future Improvements
- Support for multiple fields of study in hypothesis creation.
- Enhanced category classification using LLaMA for more accurate discipline mapping.
- Parallel processing of papers to improve performance.
- Configurable output formats for hypothesis summaries.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For issues or inquiries, please contact the project maintainer via the repository's issue tracker.