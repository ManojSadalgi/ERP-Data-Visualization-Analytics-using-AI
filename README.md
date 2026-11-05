# 🤖 AI-Powered EDA Tool

An intelligent **Exploratory Data Analysis** web application built with **Streamlit** and powered by **Llama 3.3 70B** AI. Upload any CSV dataset and get instant insights, visualizations, business analysis, predictions, and downloadable reports — all driven by AI.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![AI](https://img.shields.io/badge/AI-Llama%203.3%2070B-purple)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📋 **Data Preview** | View dataset, missing values, and statistical summary |
| ⚠️ **AI Data Quality Score** | AI rates your data on completeness, consistency, accuracy, and uniqueness |
| 🧹 **AI Data Cleaning** | AI suggests specific, actionable cleaning steps |
| 🎨 **Visualizations** | Histograms, boxplots, scatter plots, heatmaps, line charts with AI explanations |
| 📊 **Smart Charts** | AI recommends the best charts + generate charts from natural language |
| 📊 **AI Dashboard** | Auto-built interactive Plotly dashboard with KPIs, distributions, and correlations |
| ⚡ **Auto Insights** | AI auto-discovers interesting patterns, correlations, and hidden findings |
| 💼 **Business Insights** | Business-focused analysis — KPIs, revenue, growth, risks, and strategy |
| 💡 **Feature Engineering** | AI suggests new derived features for machine learning |
| 🔮 **Prediction** | Quick ML prediction using Random Forest with AI interpretation |
| 💬 **AI Chat** | Ask any question about your data in natural language |
| 📝 **Report Generation** | Generate and download a comprehensive HTML EDA report |
| 🌙 **Dark / Light Mode** | Toggle between dark and light themes |

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) — interactive web UI
- **AI Model**: [Llama 3.3 70B](https://groq.com/) via Groq API — fast AI inference
- **Data**: [Pandas](https://pandas.pydata.org/) + [NumPy](https://numpy.org/) — data manipulation
- **Static Charts**: [Matplotlib](https://matplotlib.org/) + [Seaborn](https://seaborn.pydata.org/) — statistical visualizations
- **Interactive Charts**: [Plotly](https://plotly.com/python/) — zoomable, hoverable dashboard charts
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) — Random Forest models, metrics

---

## 📁 Project Structure

```
AI-Powered EDA Tool/
├── app.py               # Main Streamlit app (UI + 12 tabs)
├── ai_insights.py       # All AI functions (Groq/Llama API)
├── eda.py               # Basic EDA helpers (stats, outliers)
├── visualizations.py    # Chart generation (matplotlib/seaborn)
├── requirements.txt     # Python dependencies
├── static/
│   ├── style.css        # Shared styles
│   ├── dark.css         # Dark theme
│   └── light.css        # Light theme
├── templates/
│   ├── header.html      # App header
│   └── landing.html     # Landing page
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- A [Groq API key](https://console.groq.com/)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AI-Powered-EDA-Tool.git
   cd AI-Powered-EDA-Tool
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/Mac
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your API key**

   Open `ai_insights.py` and replace the API key:
   ```python
   client = Groq(api_key="YOUR_GROQ_API_KEY")
   ```

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

6. Open your browser at **http://localhost:8501** and upload a CSV file to start!

---

## 📦 Dependencies

```
streamlit
pandas
numpy
matplotlib
seaborn
groq
scikit-learn
plotly
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🔄 How It Works

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Upload CSV  │────▶│  Parse with   │────▶│   DataFrame     │
│              │     │  Pandas       │     │                 │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                    ┌──────────────────────────────┼──────────────────┐
                    │                              │                  │
              ┌─────▼─────┐              ┌────────▼──────┐   ┌──────▼──────┐
              │   eda.py   │              │ ai_insights.py│   │visualizations│
              │  Stats &   │              │  AI Analysis  │   │   .py       │
              │  Outliers  │              │  (Groq API)   │   │  Charts     │
              └─────┬─────┘              └────────┬──────┘   └──────┬──────┘
                    │                              │                  │
                    └──────────────────────────────┼──────────────────┘
                                                   │
                                          ┌────────▼────────┐
                                          │  Streamlit UI   │
                                          │  (12 Tabs)      │
                                          └─────────────────┘
```

1. **User uploads a CSV** → Pandas reads it into a DataFrame
2. **Sidebar** shows basic stats (rows, columns, nulls, duplicates) via `eda.py`
3. **Each tab** performs a specific analysis:
   - Static charts via `visualizations.py` (matplotlib/seaborn)
   - Interactive charts via Plotly (AI Dashboard)
   - AI analysis via `ai_insights.py` → sends data context + prompt to Llama 3.3 70B
   - ML predictions via scikit-learn Random Forest
4. **AI Chat** allows free-form Q&A about the dataset
5. **Report** generates a downloadable HTML document

---

## 📸 Screenshots

> Upload a CSV to see the app in action with all 12 analysis tabs!

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) — for the amazing web framework
- [Groq](https://groq.com/) — for blazing fast AI inference
- [Meta AI](https://ai.meta.com/) — for the Llama 3.3 model
