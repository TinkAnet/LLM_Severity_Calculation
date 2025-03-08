# SAP200 Reasoning Auto Processor

A tool for automatically processing and analyzing content across multiple categories using the Llama API. This utility evaluates the severity of responses to various prompts and provides detailed metrics and analysis.

## Features

- Automatically processes all categories in the SAP200 directory
- Evaluates responses using a position-weighted severity scoring system
- Calculates detailed metrics for each response
- Real-time status monitoring via web interface
- Saves results to separate files by category
- Handles refusal responses

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/sap200-reasoning-processor.git
   cd sap200-reasoning-processor
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Configuration

1. Create a `.env` file in the project root with your Llama API key:
   ```
   LLAMA_API_KEY=your_api_key_here
   ```

2. Ensure your SAP200 directory structure is correct:
   ```
   SAP200/
   ├── fraud/
   │   └── generated_cases.json
   ├── politics/
   │   └── generated_cases.json
   ├── pornography_sexual_minors/
   │   └── generated_cases.json
   ├── race/
   │   └── generated_cases.json
   ├── religion/
   │   └── generated_cases.json
   ├── suicide/
   │   └── generated_cases.json
   ├── terrorism/
   │   └── generated_cases.json
   └── violence/
       └── generated_cases.json
   ```

## Usage

Run the script to automatically process all categories:

The script will:
1. Start a web server for status monitoring at http://localhost:8080
2. Process each category in the SAP200 directory sequentially
3. Save results to the `output` directory with filenames like `reasoning_1_fraud_SAP200.json`

## Web Interface

Access the status monitoring interface at http://localhost:8080. The interface shows:

- Current processing status (running/completed)
- Start time and elapsed time
- Current category and entry being processed
- Total entries processed
- Recent results with timestamps

The page automatically refreshes every 5 seconds to show the latest status.

## Output Format

Results are saved as JSON files with the following structure:

```json
[
  {
    "id": "attack_id",
    "Input": "attack_prompt",
    "First Response (Implicit)": "llm_response",
    "Second Response (Explicit)": "1A,2B,3C",
    "Final Severity Score": 3.5,
    "Severity Metrics": {
      "max_severity": 3,
      "max_weighted": 3.9,
      "avg_weighted": 2.8,
      "sum_weighted": 8.4
    },
    "Is Refusal": false,
    "Sentence Details": [
      {
        "position": 1,
        "severity_letter": "A",
        "severity_score": 1,
        "weight": 1.2,
        "weighted_severity": 1.2
      },
      // ...more sentence details
    ]
  },
  // ...more entries
]
```

## Author

Ivan

## License

