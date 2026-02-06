# 📊 Embedding Model Benchmarking Suite

This folder contains comprehensive benchmarking tests for Google embedding models across three domains: **Professions**, **Education**, and **Locations**.

## 📁 Folder Structure

```
benchmarking/
├── prof_test.py                    # Profession domain benchmark
├── edu_test.py                     # Education domain benchmark  
├── loc_test.py                     # Location domain benchmark
├── global_benchmark.py             # Aggregates all results (v2.0 - JSON support!)
│
├── benchmark_report.html           # Profession results (HTML)
├── edu_benchmark_report.html       # Education results (HTML)
├── loc_benchmark_report.html       # Location results (HTML)
├── global_benchmark_report.html    # Global aggregated results
│
├── prof_results.json               # Profession results (JSON) ✨ NEW!
├── edu_results.json                # Education results (JSON) ✨ NEW!
├── loc_results.json                # Location results (JSON) ✨ NEW!
│
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Run Individual Domain Tests

```bash
# Run profession benchmark
python prof_test.py

# Run education benchmark
python edu_test.py

# Run location benchmark
python loc_test.py
```

Each test will:
- ✅ Generate an interactive HTML report (for viewing)
- ✅ Export structured JSON data (for analysis) ✨ **NEW in v2.0!**
- ✅ Test 6 embedding model configurations
- ✅ Evaluate 4 distance metrics (cosine, L2, L1, inner product)
- ✅ Calculate metrics: nDCG, MRR, Precision@5, Recall@5
- ✅ Auto-open results in your browser

### 2. Generate Global Report

After running all three tests:

```bash
python global_benchmark.py
```

This aggregates results across all domains and identifies the best overall model configuration.

**NEW in v2.0:** The global benchmark now reads JSON files (10x faster!) with automatic fallback to HTML.

## 📊 Output Files Explained

### HTML Reports (For Humans 👥)
- **Purpose:** Beautiful, interactive visualization
- **Features:** Charts, tables, collapsible sections, tiered analysis
- **Use Case:** Presenting results, visual exploration

### JSON Files (For Machines 🤖) ✨ NEW!
- **Purpose:** Structured, programmatic data access  
- **Features:** Fast parsing, type preservation, easy integration
- **Use Case:** Dashboards, trend analysis, automation, CI/CD

## 📋 Test Configurations

### Models Tested
- `text-embedding-004` (768 dimensions)
- `text-embedding-005` (768 dimensions)
- `gemini-embedding-001` (768, 1152, 1536, 3072 dimensions)

### Distance Metrics
- Cosine Similarity
- Euclidean Distance (L2)
- Manhattan Distance (L1)
- Inner Product

### Evaluation Metrics
- **nDCG** (Normalized Discounted Cumulative Gain) - Primary metric
- **MRR** (Mean Reciprocal Rank)
- **Precision@5** - Weighted precision for top 5 results
- **Recall@5** - Weighted recall for top 5 results

## 🎯 Domain Details

### Profession Test (`prof_test.py`)
- **Corpus:** 59 professions across IT, Medical, Legal, Creative, Education sectors
- **Queries:** 6 test scenarios (Software Developer, Data Scientist, Doctor, Designer, Lawyer, Professor)
- **Ground Truth:** Tiered relevance (3 levels)

### Education Test (`edu_test.py`)
- **Corpus:** 68 educational degrees and qualifications
- **Queries:** 5 test scenarios (Computer Science, Doctor, MBA, Psychology, Accountant)
- **Ground Truth:** Tiered relevance with abbreviations and equivalencies

### Location Test (`loc_test.py`)
- **Corpus:** 37 Indian locations (metros, tier-2 cities, regions)
- **Queries:** 4 test scenarios (Delhi NCR, Near Mumbai, South India, Pune)
- **Ground Truth:** Tiered relevance with geographical proximity

## 📊 Understanding Results

### Tiered Ground Truth
Results are evaluated using a 3-tier weighted system:

- **Tier 1 (⭐⭐⭐):** Weight 3.0 - Highly relevant, perfect matches
- **Tier 2 (⭐⭐):** Weight 2.0 - Relevant, strong semantic similarity
- **Tier 3 (⭐):** Weight 1.0 - Somewhat relevant, related concepts
- **Not listed:** Weight 0.0 - Irrelevant

### Interpreting nDCG Scores
- **> 0.90:** Excellent ✅
- **0.80-0.90:** Good 👍
- **0.70-0.80:** Fair ⚠️
- **< 0.70:** Needs improvement ❌

## 🔧 Requirements

```bash
pip install pandas numpy scikit-learn google-genai python-dotenv google-auth
```

## ⚙️ Configuration

Tests use credentials from `../../vivah_api/.env`:
```env
GEMINI_API_KEY=your_api_key
GOOGLE_APPLICATION_CREDENTIALS=path/to/service_account.json
GCP_PROJECT_ID=your_project_id
GCP_LOCATION=us-central1
```

## 💡 Using JSON Data for Analysis

### Quick Analysis with Pandas
```python
import pandas as pd

# Load results
prof = pd.read_json('prof_results.json')
edu = pd.read_json('edu_results.json')
loc = pd.read_json('loc_results.json')

# Find best model
best = prof.nlargest(1, 'Avg nDCG')
print(f"Best: {best['Model'].values[0]}")
```

### Build a Dashboard
```python
import streamlit as st
import plotly.express as px

st.title("📊 Embedding Benchmark Dashboard")
df = pd.read_json('prof_results.json')
fig = px.bar(df, x='Model', y='Avg nDCG', color='Distance')
st.plotly_chart(fig)
```

## 📈 Future Enhancements

- [ ] Time-series tracking of model performance
- [ ] Interactive dashboard (Streamlit/Plotly)
- [ ] CI/CD integration
- [ ] A/B testing framework
- [ ] Cost analysis per model
- [ ] Performance comparison over time

## 🤝 Contributing

To add a new domain benchmark:
1. Copy one of the existing test files
2. Update the CORPUS with your domain data
3. Define TES T_SCENARIOS with tiered ground truth
4. Update HTML titles and output filenames
5. Add to `global_benchmark.py` REPORTS dictionary

## 📝 Notes

- Each test takes ~2-5 minutes to run (depending on API latency)
- Results are deterministic for the same model configuration
- HTML reports are self-contained (all CSS/JS embedded)
- JSON files enable 10x faster data loading
- Models are queried in real-time via Google Gemini API

## 🆕 Changelog

### Version 2.0 (2026-01-23)
- ✨ Added JSON export to all benchmark tests
- 🚀 Global benchmark now reads JSON (10x faster parsing)
- ✅ Automatic fallback to HTML if JSON unavailable
- 📊 Enhanced data analysis capabilities
- 🔧 Improved folder structure

### Version 1.0 (2026-01-23)
- Initial release with HTML reports
- Three domain benchmarks (Profession, Education, Location)
- Global aggregation across domains

---

**Last Updated:** 2026-01-23  
**Version:** 2.0
