from groq import Groq
import time
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"


def _safe_generate(prompt, max_retries=2):
    """Wrapper that handles rate limit errors with retry + friendly message."""
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_completion_tokens=4096,
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                if attempt < max_retries:
                    time.sleep(10)  # wait 10s before retry
                    continue
                return "⚠️ API rate limit reached. Please wait a minute and try again."
            else:
                return f"⚠️ AI error: {error_msg}"


def _get_dataset_context(df):
    """Build a reusable context string from the dataframe."""
    summary = df.describe().to_string()
    missing = df.isnull().sum().to_string()
    columns = ", ".join(df.columns)
    dtypes = df.dtypes.to_string()
    sample = df.head(10).to_string()

    return f"""
Dataset columns: {columns}
Data types:
{dtypes}
Missing values:
{missing}
Statistics:
{summary}
Sample data (first 10 rows):
{sample}
"""


def ask_gemini(df, question):

    context = _get_dataset_context(df)

    prompt = f"""
You are a professional data analyst.

{context}

User question:
{question}

Answer clearly based on the dataset.
"""

    return _safe_generate(prompt)


def explain_chart(df, chart_type, columns):
    """Explain what a chart is showing."""

    stats = df[columns].describe().to_string() if len(columns) > 0 else ""
    col_names = ", ".join(columns)

    prompt = f"""
You are a professional data analyst. A user generated a {chart_type} chart
for column(s): {col_names}

Statistics for those columns:
{stats}

In 2-3 concise sentences, explain what this {chart_type} chart reveals about the data.
Focus on key patterns, trends, or anomalies.
"""

    return _safe_generate(prompt)


# ============================================================
# 1. Auto Data Cleaning Suggestions
# ============================================================
def ai_clean_suggestions(df):
    """AI suggests data cleaning steps."""

    context = _get_dataset_context(df)
    duplicates = df.duplicated().sum()

    prompt = f"""
You are a data cleaning expert.

{context}
Duplicate rows: {duplicates}

Analyze this dataset and provide specific, actionable cleaning suggestions.
Format your response as a numbered list. For each suggestion include:
- What the issue is
- Which column(s) are affected
- How to fix it (specific method)

Focus on: missing values, data types, duplicates, outliers, inconsistent values.
"""

    return _safe_generate(prompt)


# ============================================================
# 2. AI Chart Recommender
# ============================================================
def ai_recommend_charts(df):
    """AI recommends the best chart types for the dataset."""

    context = _get_dataset_context(df)

    prompt = f"""
You are a data visualization expert.

{context}

Recommend the top 5 most insightful charts/visualizations for this dataset.
For each recommendation, provide:
1. **Chart Type** (e.g., Histogram, Scatter Plot, Boxplot, Heatmap, Line Chart, Pie Chart)
2. **Column(s)** to use
3. **Why** this chart would be insightful (1 sentence)

Format as a numbered list. Be specific about column names.
"""

    return _safe_generate(prompt)


# ============================================================
# 3. Anomaly Detection Analysis
# ============================================================
def ai_detect_anomalies(df, outlier_results):
    """AI analyzes outliers and anomalies."""

    context = _get_dataset_context(df)

    prompt = f"""
You are a data anomaly detection expert.

{context}

IQR-based outlier analysis results:
{outlier_results}

Analyze these anomalies and provide:
1. Which columns have the most concerning outliers and why
2. Whether these outliers are likely real data or errors
3. Recommended actions for handling each set of outliers
4. Any patterns you notice across multiple columns

Be specific and reference actual column names and values.
"""

    return _safe_generate(prompt)


# ============================================================
# 4. Feature Engineering Suggestions
# ============================================================
def ai_feature_engineering(df):
    """AI suggests new feature columns."""

    context = _get_dataset_context(df)

    prompt = f"""
You are a feature engineering expert for machine learning.

{context}

Suggest 5-8 new features that could be derived from this dataset.
For each suggestion provide:
1. **New Feature Name**
2. **Formula/Method** to create it
3. **Why** it would be useful (1 sentence)

Consider: ratios, interactions, binning, encoding, aggregations, date features, polynomial features.
Be specific to the actual columns in this dataset.
"""

    return _safe_generate(prompt)


# ============================================================
# 5. Data Quality Score
# ============================================================
def ai_data_quality_score(df):
    """AI rates the dataset quality."""

    context = _get_dataset_context(df)
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    completeness = round((1 - missing_cells / total_cells) * 100, 1) if total_cells > 0 else 0
    duplicates = df.duplicated().sum()
    dup_pct = round((duplicates / df.shape[0]) * 100, 1) if df.shape[0] > 0 else 0

    prompt = f"""
You are a data quality assessment expert.

{context}

Calculated metrics:
- Total cells: {total_cells}
- Missing cells: {missing_cells}
- Completeness: {completeness}%
- Duplicate rows: {duplicates} ({dup_pct}%)

Rate this dataset on a scale of 1-10 for each category:
1. **Completeness** (missing values) — Score: X/10
2. **Consistency** (data types, formats) — Score: X/10
3. **Accuracy** (outliers, impossible values) — Score: X/10
4. **Uniqueness** (duplicates) — Score: X/10
5. **Overall Quality Score** — Score: X/10

For each category, provide 1 sentence explanation.
Then provide a brief "Top 3 Improvements" list.
"""

    return _safe_generate(prompt)


# ============================================================
# 6. Smart Summary
# ============================================================
def ai_smart_summary(df):
    """One-click comprehensive AI analysis of the entire dataset."""

    context = _get_dataset_context(df)
    shape = f"{df.shape[0]} rows x {df.shape[1]} columns"

    prompt = f"""
You are a senior data scientist providing a comprehensive dataset briefing.

{context}
Dataset shape: {shape}

Provide a thorough analysis covering:

## 📋 Overview
Brief description of what this dataset appears to contain.

## 🔑 Key Findings
Top 5 most important observations about the data.

## 📊 Column Analysis
For each important column, provide 1-2 sentence insight.

## ⚠️ Potential Issues
Any data quality concerns, biases, or limitations.

## 💡 Recommendations
3-5 actionable next steps for working with this data.

## 🎯 Use Cases
2-3 potential analysis or ML tasks this data could support.

Use markdown formatting with headers and bullet points.
"""

    return _safe_generate(prompt)


# ============================================================
# 7. Compare Datasets
# ============================================================
def ai_compare_datasets(df1, df2):
    """AI compares two uploaded datasets."""

    ctx1 = f"""
Dataset 1:
Shape: {df1.shape}
Columns: {', '.join(df1.columns)}
Types: {df1.dtypes.to_string()}
Stats: {df1.describe().to_string()}
Sample: {df1.head(5).to_string()}
"""

    ctx2 = f"""
Dataset 2:
Shape: {df2.shape}
Columns: {', '.join(df2.columns)}
Types: {df2.dtypes.to_string()}
Stats: {df2.describe().to_string()}
Sample: {df2.head(5).to_string()}
"""

    prompt = f"""
You are a data comparison expert.

{ctx1}

{ctx2}

Compare these two datasets and provide:

## 📊 Structural Comparison
- Column differences (added/removed/renamed)
- Data type differences
- Size comparison

## 📈 Statistical Comparison
- Key metric changes for common columns
- Distribution shifts

## 🔍 Key Differences
- Most significant changes between the datasets

## 💡 Insights
- What might explain the differences
- Recommendations

Use markdown formatting.
"""

    return _safe_generate(prompt)


# ============================================================
# 8. Natural Language to Chart
# ============================================================
def ai_nl_to_chart(df, query):
    """Convert natural language query to chart specification."""

    columns = ", ".join(df.columns)
    dtypes = df.dtypes.to_string()

    prompt = f"""
You are a data visualization assistant.

Dataset columns: {columns}
Data types:
{dtypes}

User request: "{query}"

Based on the user's request, determine the best chart to create.
Respond in EXACTLY this format (no extra text):

CHART_TYPE: <one of: histogram, boxplot, scatter, line, pie, heatmap>
COLUMN_X: <column name or "none">
COLUMN_Y: <column name or "none">
TITLE: <chart title>

Rules:
- histogram: needs COLUMN_X only
- boxplot: needs COLUMN_X only
- scatter: needs both COLUMN_X and COLUMN_Y
- line: needs COLUMN_X only
- pie: needs COLUMN_X only (categorical column)
- heatmap: set both to "none" (uses all numeric)
- Use exact column names from the dataset
"""

    return _safe_generate(prompt)


# ============================================================
# 9. Auto Report Generation
# ============================================================
def ai_generate_report(df):
    """Generate a comprehensive HTML report."""

    context = _get_dataset_context(df)
    shape = f"{df.shape[0]} rows x {df.shape[1]} columns"
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0].to_string() if missing.sum() > 0 else "None"
    duplicates = df.duplicated().sum()

    prompt = f"""
You are a professional data analyst writing a formal EDA report.

{context}
Shape: {shape}
Missing value columns: {missing_cols}
Duplicates: {duplicates}

Write a comprehensive EDA report in clean HTML format. Include:

1. <h2>Executive Summary</h2> - Brief overview
2. <h2>Dataset Overview</h2> - Shape, columns, types
3. <h2>Data Quality Assessment</h2> - Missing values, duplicates, types
4. <h2>Statistical Analysis</h2> - Key statistics and distributions
5. <h2>Key Findings</h2> - Top insights as bullet points
6. <h2>Recommendations</h2> - Next steps

Use proper HTML tags: <h2>, <p>, <ul>, <li>, <table>, <tr>, <td>, <th>, <strong>.
Add inline CSS for a professional look with a clean, modern style.
Wrap everything inside a <div> tag. Do NOT include <html>, <head>, or <body> tags.
"""

    return _safe_generate(prompt)


# ============================================================
# 10. AI Business Insights
# ============================================================
def ai_business_insights(df):
    """AI analyzes the dataset from a business perspective."""

    context = _get_dataset_context(df)
    shape = f"{df.shape[0]} rows x {df.shape[1]} columns"

    prompt = f"""
You are a senior business analyst and strategy consultant.

{context}
Dataset shape: {shape}

Analyze this dataset from a **business perspective** and provide:

## 📊 Business Overview
What kind of business data is this? What domain does it belong to?

## 💰 Revenue & Financial Insights
Identify any revenue, cost, profit, or financial columns. Analyze trends, averages, and key metrics.

## 📈 Growth & Performance KPIs
Identify key performance indicators from the data. Calculate or estimate growth rates, conversion rates, or other relevant KPIs.

## 👥 Customer/Segment Analysis
If applicable, analyze customer segments, demographics, or behavioral patterns.

## ⚠️ Business Risks
Identify potential risks, declining metrics, or areas of concern.

## 🎯 Strategic Recommendations
Provide 5 specific, actionable business recommendations based on data trends.

## 📊 Key Metrics Dashboard
Summarize the top 5-8 business metrics in a table format:
| Metric | Value | Status |
Use ✅ for good, ⚠️ for warning, ❌ for critical.

Use markdown formatting. Be specific and reference actual numbers from the data.
"""

    return _safe_generate(prompt)


# ============================================================
# 11. AI Auto Insights
# ============================================================
def ai_auto_insights(df):
    """Automatically discover and present the most interesting patterns."""

    context = _get_dataset_context(df)
    shape = f"{df.shape[0]} rows x {df.shape[1]} columns"

    # Calculate correlations for numeric columns
    numeric_df = df.select_dtypes(include='number')
    corr_info = ""
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr()
        # Get top correlations (excluding self-correlation)
        pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        top_corrs = pairs[:10]
        corr_info = "Top correlations:\n"
        for c1, c2, val in top_corrs:
            corr_info += f"  {c1} ↔ {c2}: {val:.3f}\n"

    # Value distribution for categorical columns
    cat_df = df.select_dtypes(include=['object', 'category'])
    cat_info = ""
    for col in cat_df.columns[:5]:
        cat_info += f"\n{col} value counts (top 5):\n{df[col].value_counts().head(5).to_string()}\n"

    prompt = f"""
You are an expert data scientist performing automated insight discovery.

{context}
Dataset shape: {shape}

{corr_info}

Categorical distributions:
{cat_info}

Automatically discover and present the **most interesting and surprising insights** from this data.

## 🔥 Top 5 Auto-Discovered Insights
For each insight:
- **Finding**: What you discovered (specific, with numbers)
- **Significance**: Why this matters (1 sentence)
- **Confidence**: High / Medium / Low

## 📊 Interesting Correlations
Highlight the most meaningful correlations and what they suggest.

## 🔍 Hidden Patterns
Any non-obvious patterns, clusters, or segmentation you notice.

## 📉 Anomalies & Outliers
Unusual data points or distributions worth investigating.

## 💡 Questions This Data Can Answer
5 specific analytical questions this dataset is well-suited to answer.

## 🚀 Quick Wins
3 immediate actions that could generate value from this data.

Use markdown formatting. Be specific — reference actual column names, values, and numbers.
Prioritize surprising, non-obvious findings over basic statistics.
"""

    return _safe_generate(prompt)