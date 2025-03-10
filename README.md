# SAP200 Reasoning Auto Processor

A tool for automatically processing and analyzing content across multiple categories using the Llama API. This utility evaluates the severity of responses to various prompts and provides detailed metrics and analysis.

## Features

- Automatically processes all categories in the SAP200 directory
- Evaluates responses using a position-weighted severity scoring system
- Calculates detailed metrics for each response
- Real-time status monitoring via web interface
- Saves results to separate files by category
- Handles refusal responses
- Generates statistical analysis and visualizations of severity scores

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/TinkAnet/LLM_Severity_Calculation.git
   cd LLM_Severity_Calculation
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
```bash
python3 reasoning.py
or
python3 reasoning_auto.py
```

The script will:
1. Start a web server for status monitoring at http://localhost:8080
2. Process each category in the SAP200 directory sequentially
3. Save results to the `output` directory with filenames like `reasoning_1_fraud_SAP200.json`

## Statistical Analysis

After processing the responses, you can generate statistical analysis and visualizations:

```bash
python3 statistic.py
```

This will:
1. Process all JSON files in the `output` directory
2. Extract severity scores for each category
3. Generate individual scatter plots for each category with statistics (mean, median, min, max)
4. Create a summary scatter plot comparing all categories
5. Save all plots to the `graph` directory

### Generated Visualizations

The script generates the following visualizations:

- `severity_scores_all_categories.png` - A summary scatter plot comparing severity scores across all categories
- Individual category plots:
  - `severity_scores_fraud.png`
  - `severity_scores_politics.png`
  - `severity_scores_pornography_sexual_minors.png`
  - `severity_scores_race.png`
  - `severity_scores_religion.png`
  - `severity_scores_suicide.png`
  - `severity_scores_terrorism.png`
  - `severity_scores_violence.png`

Each scatter plot displays individual data points with some jitter to prevent overlapping and includes statistical summary information (number of data points, mean, median, min, max).

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

