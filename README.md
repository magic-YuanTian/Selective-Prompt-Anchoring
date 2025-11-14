<h1 align="center">SQLsynth</h1>
<p align="center">
  <em>Text-to-SQL Domain Adaptation via Human-LLM Collaborative Data Annotation</em>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2502.15980"><img src="https://img.shields.io/badge/arXiv-2408.09121-b31b1b.svg" alt="arXiv"></a>
  <a href="https://dl.acm.org/doi/10.1145/3708359.3712083"><img src="https://img.shields.io/badge/ACM-IUI'25-0085CA.svg" alt="ACM"></a>
</p>

<img width="1000"  alt="teaser" src="https://github.com/user-attachments/assets/cfb9613e-abfb-48cb-9b48-1525c003ec07" />


This is the repo for IUI'25 paper, [Text-to-SQL Domain Adaptation via Human-LLM Collaborative Data Annotation](https://arxiv.org/abs/2502.15980).

- *Note: This repo serves as the latest and backup version of the [official Adobe repo](https://github.com/adobe/nl_sql_analyzer).*
  
SQLsynth is not only an **interactive data annotation** but also **automated data synthesis** tool designed for quickly creating highly customized (e.g., schema, DB records, distribution) text-to-SQL datasets. 


## 🌟 Features

- **Database Schema Customization**
  - Freely create, edit, annotate (use NL to label the semantics of database fields, useful for LLMs) in the canvas.
  - 📦 --> A highly customized database schema, with meaningful descriptions
- **Database Records Population**
  - Given a database schema, populate it with concrete records
  - Rule-based method (No LLM calling)
  - Recognized for different datatype
  - Distribution is configurable
  - 📦 --> A complete, customized database full of records
- **SQL Query Sampling**
  - Given a database, randomly sample SQL queries.
  - Based on PCFG (Probability Context-Free Grammar) and other rules to extract records from a specified database.
  - The probability distribution is configurable (e.g., increase the number of queries with WHERE clauses or those involving a specific column).
  - Syntax is customizable (e.g., support for user-defined SQL dialect).
  - 📦 --> A large amount of SQL queries (with a customized distribution) under the provided database
- **SQL-to-Text Generation**
  - Convert SQL queries into NL questions
  - Three stages:
    1. Convert the SQL query into step-by-step NL explanations by a [grammar-based method](https://github.com/magic-YuanTian/STEPS).
    2. Conduct in-context on specified real-world data for style adaptation
    3. Generating the NL question by LLMs
  - 📦 --> A large amount of (NL, SQL) pairs under the customized database, where NL questions may be perfect (ambiguous, lack details, etc.)
- **Text-SQL Alignment**:
  - Mapping NL components (substrings) to SQL compoenents (clauses)
  - Error checking for generated NL (note that the SQL is absolutely correct)
  - Use to analyze (1) what information may be missing (the SQL component fails to map to NL components), and (2) what information may be redundant (the NL component doesn't map to any SQL compoenent)
  - Interactively highlight by visual correspondence in the UI
  - 📦 --> A large amount of *reliable* (NL, SQL) pairs under the customized database
- **Dataset statistics & visualization**:
  - Upload and analyze existing SQL query datasets
  - Assist users in tracking datasets from a dataset-level perspective
  - Comprehensive statistics dashboard with summary metrics (total queries, unique keywords, average complexity),including:
    - SQL structure distribution
    - Keyword frequency distribution
    - Clause number distribution
    - Column and table usage patterns
    - Query complexity distribution
    - Reference value distribution
  - 📦 --> Insights into dataset characteristics and qualities


<img width="2000" alt="overview" src="https://github.com/user-attachments/assets/538533f8-eebc-42f0-8f80-2822fb707847" />


## 🚀 Installation

### Backend Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SQLsynth.git
cd SQLsynth
```

2. Install Python dependencies:
```bash
cd backend
pip install flask
pip install flask_cors
pip install sql-metadata
pip install openai
pip install nltk
pip install spacy
pip install sqlparse
python -m spacy download en_core_web_sm
```

3. Configure LLM API:
   - Open `backend/openai_api.py`
   - Implement your own `get_openai_response()` function
   - The function should take a string prompt as input and return a string response


### Frontend Setup

1. Install Node.js dependencies:
```bash
cd frontend
npm install
```

2. If you encounter missing dependencies, please use `npm install` for necessary packages based on pop-up instructions.

## 🎯 Quick Start

### Running the Application

1. **Start the Backend Server**:
```bash
cd backend
python server.py
```
The backend will run on `http://localhost:5001` by default.

2. **Start the Frontend**:
```bash
cd frontend
npm start
```
The frontend will run on `http://localhost:3000` by default.

3. Open your browser and navigate to `http://localhost:3000`

4. Enjoy it! 🎉

### Basic Workflow

1. **Schema Tab**: Design or import your database schema
2. **Database Tab**: Generate synthetic records for your schema
3. **Dataset Tab**: Synthesize SQL queries and natural language pairs
4. **Analysis Tab**: Analyze alignment between SQL and natural language

## 🏗️ Architecture

### Backend (`backend/`)

- **`server.py`**: Flask server handling all API endpoints
- **`SQL_synthesizer.py`**: PCFG-based SQL query generation
- **`SQL2NL_clean.py`**: Rule-based SQL decomposition and explanation
- **`llm_analysis.py`**: LLM prompts and analysis functions
- **`records_synthesizer.py`**: Database record generation with constraint satisfaction
- **`ICL_retriever.py`**: In-context learning example retrieval
- **`db_handling.py`**: Database operations and utilities
- **`openai_api.py`**: LLM API interface (user-implemented)
- **`evaluation_steps.py`**: Evaluation utilities

### Frontend (`frontend/src/`)

- **`App.jsx`**: Main application component with global state management
- **`SchemaTab.jsx`**: Interactive schema designer
- **`DatabaseTab.jsx`**: Database record management interface
- **`DatasetTab.jsx`**: Dataset synthesis and download
- **`AnalysisTab.jsx`**: SQL-NL alignment analysis
- **`SQLSubexpressionCorrespondence.jsx`**: Visual representation of SQL components

### Configuration Files

- **`manual_config.json`**: Manual probability configuration for SQL synthesis
- **`learned_config.json`**: Learned probability distribution from existing datasets
- **`spider_example_pool.json`**: Example pool for in-context learning

## 📖 Usage

### Web Interface

#### 1. Schema Design

- **Import Schema**: Drag and drop a JSON schema file
- **Edit Schema**: Add/remove tables and columns
- **Define Relationships**: Specify primary and foreign keys
- **Add Descriptions**: Document tables and columns for better NL generation

Schema format example:
```json
{
  "users": {
    "comment": "User information table",
    "columns": [
      {
        "field": "user_id",
        "type": "text",
        "isPrimary": true,
        "comment": "Unique user identifier"
      },
      {
        "field": "username",
        "type": "text",
        "comment": "User's login name"
      }
    ]
  }
}
```
<img width="5128" height="2667" alt="page1" src="https://github.com/user-attachments/assets/dda9b1a3-c811-415a-a5b5-3853363b40e7" />

#### 2. Record Synthesis

- Click "Generate Records" to create synthetic data
- Specify the number of records to generate
- Records respect foreign key constraints and data types
- Export records to JSON
  
<img width="7343" height="4593" alt="page2" src="https://github.com/user-attachments/assets/b41292e2-1203-43bc-9349-646e8c404bc1" />

#### 3. NL-SQL pair Synthesis

- Configure query distribution (number of tables, columns, clauses)
- Generate individual queries or batch synthesis
- View step-by-step SQL decomposition
- Get suggested natural language descriptions
- Check alignment between SQL and NL

<img width="7773" height="6218" alt="page3_core" src="https://github.com/user-attachments/assets/b57c7f71-2242-4c4b-8224-97b4e7c1d8cf" />


<img width="6542" height="2690" alt="page3_2" src="https://github.com/user-attachments/assets/0f3c0f36-d3dd-40dd-84dc-8be7585d82ed" />


#### 4. Dataset Analysis

- Upload existing SQL query datasets
- View comprehensive statistics:
  - Keyword distribution
  - Query structure patterns
  - Clause complexity
  - Column and table usage
  - Query complexity metrics

<img width="7855" height="5847" alt="page4" src="https://github.com/user-attachments/assets/4cfd0ec4-92a0-4202-8727-7fe305603483" />



### Script-Based Synthesis

While human-in-the-loop guarantees the data quality, you can also opt for large-scale dataset generation without the UI:


```python
from server import auto_synthetic_data

synthetic_data = auto_synthetic_data(
    schema_path="backend/saved_frontend_schema.json",
    save_path="backend/output_data/synthetic_data.jsonl",
    config_path="backend/learned_config.json",
    synthesized_DB_records_path="backend/output_data/DB_records.json",
    example_path="backend/spider_example_pool.json",
    data_num=2000
)
```

**Parameters**:
- `schema_path`: Path to the database schema JSON file
- `save_path`: Output file path for synthetic data
- `config_path`: Configuration file for query distribution
- `synthesized_DB_records_path`: Path to save generated database records
- `example_path`: Path to example pool for in-context learning
- `data_num`: Number of SQL-NL pairs to generate

## ⚙️ Configuration

### Query Distribution Configuration

Adjust probabilities in `learned_config.json` or `manual_config.json`:

```json
{
  "sample_table_probs": [0.5, 0.3, 0.2],
  "sample_column_probs": [0.4, 0.3, 0.2, 0.1],
  "select_star_prob": 0.2,
  "where_clause_prob": 0.3,
  "group_by_clause_prob": 0.2,
  "order_by_clause_prob": 0.3,
  "having_clause_prob": 0.3,
  "limit_clause_count": 0.1
}
```

### Network Configuration

#### Change Backend Port

Edit `backend/server.py`:
```python
app.run(debug=True, host="0.0.0.0", port=YOUR_PORT)
```

#### Change Frontend Port

```bash
# macOS/Linux
PORT=4000 npm start

# Windows
set PORT=4000 && npm start
```

#### Deploy on Server

Replace `localhost` with your server IP in `frontend/src/App.jsx`:
```javascript
const ip = 'your.server.ip';  // or domain name
const port = 5001;
```

## 🔌 API Reference

### Key Endpoints

#### `POST /step_by_step_description`
Generate step-by-step explanation for a SQL query.

**Request**:
```json
{
  "sql": "SELECT name FROM users WHERE age > 18",
  "schema": {...}
}
```

**Response**:
```json
{
  "explanation_data": [...]
}
```

#### `POST /suggested_nl`
Get suggested natural language description for SQL.

**Request**:
```json
{
  "sql": "...",
  "schema": {...},
  "parsed_step_by_step_data": [...]
}
```

**Response**:
```json
{
  "nl_query": "What are the names of users older than 18?",
  "examples": [...]
}
```

#### `POST /check_alignment`
Check alignment between NL and SQL components.

**Request**:
```json
{
  "sql": "...",
  "nl": "...",
  "schema": {...},
  "parsed_step_by_step_data": [...]
}
```

**Response**:
```json
{
  "alignment_data": [...],
  "uncovered_substrings": [...]
}
```

#### `POST /synthesize_records`
Generate synthetic database records.

**Request**:
```json
{
  "schema": {...},
  "num": 100
}
```

**Response**:
```json
{
  "synthetic_records": {...}
}
```

#### `POST /synthetic_sql`
Generate a random SQL query.

**Request**:
```json
{
  "schema": {...},
  "records": {...}
}
```

**Response**:
```json
{
  "synthetic_sql": "SELECT ...",
  "config": {...}
}
```

#### `POST /analyze_dataset`
Analyze an uploaded SQL query dataset.

**Request**: Multipart form data with file upload

**Response**:
```json
{
  "totalQueries": 1000,
  "averageComplexity": 12.5,
  "keywordDistribution": {...},
  "structureDistribution": {...},
  ...
}
```


## Project Structure

```
SQLsynth_repo/
├── backend/
│   ├── server.py              # Main Flask server
│   ├── SQL_synthesizer.py     # Query synthesis engine
│   ├── SQL2NL_clean.py        # Rule-based SQL parser
│   ├── llm_analysis.py        # LLM prompts and analysis
│   ├── records_synthesizer.py # Record generation
│   ├── ICL_retriever.py       # Example retrieval
│   ├── db_handling.py         # Database utilities
│   ├── openai_api.py          # LLM API interface
│   ├── evaluation_steps.py    # Evaluation tools
│   ├── *_config.json          # Configuration files
│   ├── output_data/           # Generated datasets
│   └── temp_db/               # Temporary databases
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # Main app component
│   │   ├── SchemaTab.jsx      # Schema designer
│   │   ├── DatabaseTab.jsx    # Record management
│   │   ├── DatasetTab.jsx     # Dataset synthesis
│   │   └── AnalysisTab.jsx    # Analysis interface
│   ├── public/                # Static assets
│   └── package.json           # Dependencies
├── user_study/
│   └── spider_schemas/        # 166 Spider schemas
└── README.md
```

## 📝 Citation

If you use SQLsynth in your research, please cite:

```bibtex
@inproceedings{Tian_2025, series={IUI ’25},
   title={Text-to-SQL Domain Adaptation via Human-LLM Collaborative Data Annotation},
   url={http://dx.doi.org/10.1145/3708359.3712083},
   DOI={10.1145/3708359.3712083},
   booktitle={Proceedings of the 30th International Conference on Intelligent User Interfaces},
   publisher={ACM},
   author={Tian, Yuan and Lee, Daniel and Wu, Fei and Mai, Tung and Qian, Kun and Sahai, Siddhartha and Zhang, Tianyi and Li, Yunyao},
   year={2025},
   month=mar, pages={1398–1425},
   collection={IUI ’25} }

```

## Acknowledgments

- Adobe Property

## Contact

For questions or feedback, please open an issue on GitHub or contact by tian211@purdue.edu.


