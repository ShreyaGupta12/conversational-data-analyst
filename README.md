# Conversational Data Analyst Tool

A Python-based **Conversational Data Analysis Agent** that allows users to perform exploratory data analysis using **natural language commands**. The system leverages **LangChain**, **local LLMs via Ollama**, and custom Python analysis tools to make data insights accessible to non-programmers.

---

## Project Overview

Traditional data analysis often requires writing repetitive code and technical expertise. This project addresses that limitation by enabling users to interact with datasets conversationally.

Users can ask questions such as:
- *"Show me the distribution of funding"*
- *"Calculate correlation between valuation and employees"*
- *"Clean the funding column"*

The agent interprets intent, executes the appropriate Python tools, and returns numerical or visual insights.

---

## Key Features

- Natural language interface for data analysis
- Automatic tool selection using LangChain
- Local LLM execution using Ollama (privacy-preserving)
- Modular Python analysis functions
- Visualization generation (histograms, plots)
- Robust error handling and structured outputs

---

## Tech Stack

- **Python 3.11**
- **LangChain**
- **Ollama (Local LLMs)**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **JSON-based tool calling**

---

## Dataset

The project uses a startup analytics dataset containing the following fields:

- Funding (USD)
- Valuation
- Number of Employees
- Founded Year
- Startup Stage

Dataset file:
```
data/startup_dataset.csv
```

---

## Project Structure

```
conversational-data-analyst/
│
├── data/
│   └── startup_dataset.csv
│
├── plots/
│   └── *.png
│
├── tools.py
├── main.py
├── requirements.txt
├── README.md
└── .venv/
```

---

## Core Python Tools

The system exposes analytical capabilities through well-defined Python functions:

### `clean_feature()`
- Handles missing values
- Converts numeric columns
- Removes formatting symbols

### `plot_histogram()`
- Generates histograms for selected features
- Saves plots automatically

### `calculate_correlation()`
- Computes correlation between numerical variables
- Returns interpretable numeric output

Each function includes validation and error handling to ensure reliable execution.

---

## System Architecture

1. **User Input** (Natural Language Query)
2. **LLM Intent Parsing**
3. **Tool Selection via LangChain**
4. **Python Function Execution**
5. **Structured Output / Visualization**
6. **Response Returned to User**

This pipeline enables seamless conversion of language → analytics → insight.

---

## Interactive Workflow

- User enters a query
- LLM determines required operation
- Corresponding Python tool is invoked
- Output is returned as:
  - Numerical result
  - Table
  - Saved visualization

The loop continues until the user exits.

---

## Installation

### 1. Clone the Repository

```
git clone https://github.com/your-username/conversational-data-analyst.git
cd conversational-data-analyst
```

### 2. Create Virtual Environment

```
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. Install Ollama

Follow instructions from:

https://ollama.com

Pull a model, for example:

```
ollama pull llama3
```

---

## Running the Application

```
python main.py
```

Then interact using natural language queries.

---

## Example Queries

- "Show histogram of total funding"
- "Clean the valuation column"
- "Find correlation between funding and employees"
- "Describe the dataset"

---

## Evaluation Criteria Covered

- Correct tool implementation
- Effective LangChain tool binding
- Fully operational conversational loop
- Structured JSON outputs
- Visualization support

---

## Limitations

- Occasional JSON parsing errors from LLM
- Limited to basic EDA operations
- Dependent on local LLM performance
- No advanced statistics or ML models yet

---

## Future Enhancements

- Regression and statistical testing
- Time-series analysis
- Interactive dashboards (Streamlit / Gradio)
- Multi-dataset support
- SQL-based querying
- Improved prompt optimization

---

## Author

**Shreya Gupta**  

---

## License

This project is intended for academic and research purposes.

---

## Acknowledgements

- LangChain
- Ollama
- Open-source Python ecosystem

