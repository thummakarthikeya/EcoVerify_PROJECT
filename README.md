# Eco-Verify
### A Comparative Gap Analysis Tool for Corporate and Consumer Social Responsibility Using Machine Learning-Based Topic Modeling and NLP

This project has **two parts**:
1. `Eco_Verify_Project.ipynb` — the full analysis notebook (code + explanations + results), for
   documentation/report/viva purposes.
2. `app.py` — an interactive **web frontend** (Streamlit) for the same pipeline, so you can click a
   button and see everything visually in a browser.

Both use the exact same NLP + Machine Learning pipeline described in your project PPT:
Data Collection → Text Preprocessing (NLTK) → TF-IDF → Topic Modeling (LDA/NMF) → Visualization
(word clouds, charts) → Gap Analysis.

---

## 1. Setup (do this once)

You need **Python 3.10+** installed. Then, in a terminal, inside this project folder:

```bash
pip install -r requirements.txt
```

This installs Streamlit, NLTK, scikit-learn, pandas, numpy, matplotlib, wordcloud, and Jupyter.

---

## 2. Run the Web App (Frontend)

```bash
streamlit run app.py
```

This opens a browser tab automatically (usually at `http://localhost:8501`). If it doesn't open
automatically, copy that URL into your browser.

**How to use it:**
1. In the sidebar, choose **"Use built-in sample data"** (or upload your own CSVs — see below).
2. Adjust the number of topics / algorithm (LDA or NMF) if you like.
3. Click **"🚀 Run Analysis"**.
4. Explore the tabs: Data → Preprocessing → Consumer Topics → Corporate Topics → Gap Analysis.
5. Download the gap analysis results as CSV from the last tab.

**To use your own real data instead of the built-in sample data:**
- Consumer reviews CSV needs a column named `review_text`.
- Corporate statements CSV needs a column named `statement_text`.
- Upload both files in the sidebar and click Run Analysis.

---

## 3. Run the Notebook (for your report / viva)

```bash
jupyter notebook Eco_Verify_Project.ipynb
```

Then click **"Run All"** in the Jupyter toolbar (Cell → Run All / Kernel → Restart & Run All).
It is fully self-contained — it generates its own sample data, so it runs top to bottom with no
extra setup.

---

## 4. Project Structure

```
EcoVerify_Project/
├── app.py                        # Streamlit frontend (the "UI" of the project)
├── Eco_Verify_Project.ipynb      # Full notebook: code + explanations + results
├── requirements.txt              # One-command install
├── data/
│   ├── consumer_reviews.csv      # Sample consumer review dataset
│   └── corporate_statements.csv  # Sample corporate CSR statement dataset
└── gap_analysis.csv              # Example output of the gap analysis
```

---

## 5. About the Data

No public labeled dataset of "Consumer Social Responsibility" reviews exists, so this project uses
a realistic **synthetic dataset**: templated consumer reviews across 9 CSR themes (packaging waste,
labor practices, greenwashing skepticism, carbon footprint, ethical sourcing, recycling, price of
ethical products, corporate transparency, animal testing), and templated CSR statements from 8
fictional companies. The **pipeline itself is fully real** — swap in a real CSV (via the app's
upload option) and it works unchanged.

---

## 6. What the Gap Analysis Means

For each topic discovered in consumer reviews, the tool finds the closest matching topic in
corporate CSR statements (using Jaccard similarity of top keywords). If no corporate topic is a
good match (below the gap threshold), that consumer concern is flagged as a **GAP** — meaning
companies are not talking about something consumers clearly care about. This is the core insight
Eco-Verify is built to surface for decision-makers.
