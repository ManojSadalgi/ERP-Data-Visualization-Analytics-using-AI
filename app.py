import streamlit as st
import pandas as pd
import numpy as np
import os

from eda import basic_eda, detect_outliers_iqr
from visualizations import correlation_heatmap, histogram, boxplot, scatter_plot, line_chart, pie_chart
from ai_insights import (
    ask_gemini, explain_chart, ai_clean_suggestions, ai_recommend_charts,
    ai_detect_anomalies, ai_feature_engineering, ai_data_quality_score,
    ai_smart_summary, ai_compare_datasets, ai_nl_to_chart, ai_generate_report,
    ai_business_insights, ai_auto_insights,
)

# ================================================================
# Helpers — load external CSS / HTML
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_file(path):
    with open(os.path.join(BASE_DIR, path), encoding="utf-8") as f:
        return f.read()

def inject_css(path):
    st.markdown(f"<style>{load_file(path)}</style>", unsafe_allow_html=True)

def inject_html(path):
    st.markdown(load_file(path), unsafe_allow_html=True)

# ================================================================
# Page Configuration
# ================================================================
st.set_page_config(
    page_title="InsightForge AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
# Session State
# ================================================================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False
if "chat_response" not in st.session_state:
    st.session_state.chat_response = ""

dark = st.session_state.dark_mode

# ================================================================
# Inject CSS — shared + theme
# ================================================================
inject_css("static/style.css")
inject_css("static/dark.css" if dark else "static/light.css")

# ================================================================
# Header — Title  +  Dark-mode toggle
# ================================================================
inject_html("templates/header.html")

_, toggle_col = st.columns([14, 1])
with toggle_col:
    if st.button("☀️" if dark else "🌙", key="theme_toggle", use_container_width=True, help="Toggle dark / light mode"):
        st.session_state.dark_mode = not dark
        st.rerun()

# ================================================================
# File Uploader
# ================================================================
file = st.file_uploader("📁 Upload CSV dataset", type=["csv"])

if file:
    df = pd.read_csv(file)

    # ==========================
    # Sidebar — Dataset Info
    # ==========================
    with st.sidebar:
        st.markdown("### 📊 Dataset Info")
        rows, cols, null_values = basic_eda(df)
        st.metric("Rows", rows)
        st.metric("Columns", cols)
        st.metric("Null Values", null_values)
        st.metric("Duplicates", df.duplicated().sum())
        st.divider()
        with st.expander("📋 Column Details", expanded=False):
            for c in df.columns:
                st.markdown(f"**{c}** — `{df[c].dtype}`")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    all_cols = df.columns.tolist()

    # ==========================
    # Tabs — new workflow order
    # ==========================
    tabs = st.tabs([
        "📋 Preview",
        "⚠️ Data Quality",
        "🧹 Cleaning",
        "🎨 Visualizations",
        "📊 Smart Charts",
        "📊 AI Dashboard",
        "⚡ Auto Insights",
        "💼 Business Insights",
        "💡 Feature Engineering",
        "🔮 Prediction",
        "💬 AI Chat",
        "📝 Report",
    ])

    # =====================================================
    # 0 — Preview
    # =====================================================
    with tabs[0]:
        st.header("📋 Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)
        st.subheader("❓ Missing Values")
        missing = df.isnull().sum()
        missing_df = pd.DataFrame({"Column": missing.index, "Missing": missing.values,
                                   "% Missing": (missing.values / len(df) * 100).round(2)})
        st.dataframe(missing_df[missing_df["Missing"] > 0], use_container_width=True, hide_index=True)
        if missing.sum() == 0:
            st.success("No missing values found! ✅")
        st.subheader("📈 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

    # =====================================================
    # 1 — Data Quality
    # =====================================================
    with tabs[1]:
        st.header("⚠️ AI Data Quality Score")
        st.write("AI rates your dataset across completeness, consistency, accuracy, and uniqueness.")
        if st.button("📋 Assess Data Quality", key="quality_btn"):
            with st.spinner("🤖 AI is scoring your data quality..."):
                quality = ai_data_quality_score(df)
            st.markdown(quality)

    # =====================================================
    # 2 — Cleaning
    # =====================================================
    with tabs[2]:
        st.header("🧹 AI Data Cleaning")
        st.write("AI analyzes your dataset and suggests specific cleaning steps.")
        if st.button("🔍 Analyze & Suggest Cleaning", key="clean_btn"):
            with st.spinner("🤖 AI is analyzing your data for cleaning..."):
                suggestions = ai_clean_suggestions(df)
            st.markdown(suggestions)

    # =====================================================
    # 3 — Visualizations
    # =====================================================
    with tabs[3]:
        st.header("🎨 Visualizations")
        if len(numeric_cols) == 0:
            st.warning("No numeric columns found for visualization.")
        else:
            selected_cols = st.multiselect("Select columns to visualize", options=numeric_cols,
                                           default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols, key="viz_cols")
            if selected_cols:
                chart_type = st.selectbox("Choose chart type",
                                          ["Histogram (Distribution)", "Boxplot (Outliers)", "Correlation Heatmap",
                                           "Scatter Plot", "Line Chart"], key="viz_chart_type")
                if st.button("📊 Generate Charts", key="viz_btn"):
                    if chart_type == "Correlation Heatmap":
                        if len(selected_cols) < 2:
                            st.warning("Select at least 2 columns for a heatmap.")
                        else:
                            fig = correlation_heatmap(df, selected_cols)
                            if fig:
                                st.pyplot(fig)
                                with st.spinner("🤖 AI analyzing..."):
                                    insight = explain_chart(df, "Correlation Heatmap", selected_cols)
                                st.info(f"**🤖 AI Insight:** {insight}")
                    elif chart_type == "Histogram (Distribution)":
                        for col in selected_cols:
                            fig = histogram(df, col)
                            st.pyplot(fig)
                            with st.spinner(f"🤖 Analyzing {col}..."):
                                insight = explain_chart(df, "Histogram", [col])
                            st.info(f"**🤖 AI Insight for {col}:** {insight}")
                    elif chart_type == "Boxplot (Outliers)":
                        for col in selected_cols:
                            fig = boxplot(df, col)
                            st.pyplot(fig)
                            with st.spinner(f"🤖 Analyzing {col}..."):
                                insight = explain_chart(df, "Boxplot", [col])
                            st.info(f"**🤖 AI Insight for {col}:** {insight}")
                    elif chart_type == "Scatter Plot":
                        if len(selected_cols) < 2:
                            st.warning("Select at least 2 columns for a scatter plot.")
                        else:
                            for i in range(len(selected_cols)):
                                for j in range(i + 1, len(selected_cols)):
                                    fig = scatter_plot(df, selected_cols[i], selected_cols[j])
                                    st.pyplot(fig)
                                    with st.spinner("🤖 Analyzing..."):
                                        insight = explain_chart(df, "Scatter Plot", [selected_cols[i], selected_cols[j]])
                                    st.info(f"**🤖 AI Insight:** {insight}")
                    elif chart_type == "Line Chart":
                        for col in selected_cols:
                            fig = line_chart(df, col)
                            st.pyplot(fig)
                            with st.spinner(f"🤖 Analyzing {col}..."):
                                insight = explain_chart(df, "Line Chart", [col])
                            st.info(f"**🤖 AI Insight for {col}:** {insight}")
            else:
                st.warning("Please select at least one column.")

    # =====================================================
    # 4 — Smart Charts (AI recommends + NL chart generation)
    # =====================================================
    with tabs[4]:
        st.header("📊 Smart Chart Generation")
        st.write("AI recommends the best charts **and** generates charts from natural language.")

        rec_tab, nl_tab = st.tabs(["🤖 AI Recommendations", "🗣️ Describe a Chart"])

        with rec_tab:
            if st.button("🤖 Recommend Charts", key="recommend_btn"):
                with st.spinner("🤖 AI is analyzing your dataset..."):
                    recommendations = ai_recommend_charts(df)
                st.markdown(recommendations)

        with nl_tab:
            nl_query = st.text_input("Describe the chart you want", placeholder="e.g., Show me the distribution of age", key="nl_input")
            if nl_query and st.button("🎨 Generate", key="nl_btn"):
                with st.spinner("🤖 AI is creating your chart..."):
                    result = ai_nl_to_chart(df, nl_query)
                try:
                    spec = {}
                    for line in result.strip().split("\n"):
                        if ":" in line:
                            k, v = line.split(":", 1)
                            spec[k.strip().upper()] = v.strip()
                    chart = spec.get("CHART_TYPE", "").lower()
                    col_x, col_y = spec.get("COLUMN_X", "none"), spec.get("COLUMN_Y", "none")
                    fig = None
                    if chart == "histogram" and col_x in df.columns:   fig = histogram(df, col_x)
                    elif chart == "boxplot" and col_x in df.columns:   fig = boxplot(df, col_x)
                    elif chart == "scatter" and col_x in df.columns and col_y in df.columns: fig = scatter_plot(df, col_x, col_y)
                    elif chart == "line" and col_x in df.columns:      fig = line_chart(df, col_x)
                    elif chart == "pie" and col_x in df.columns:       fig = pie_chart(df, col_x)
                    elif chart == "heatmap":                           fig = correlation_heatmap(df, numeric_cols)
                    if fig:
                        st.pyplot(fig)
                        with st.spinner("🤖 Explaining chart..."):
                            used = [c for c in [col_x, col_y] if c != "none" and c in df.columns] or numeric_cols[:3]
                            insight = explain_chart(df, chart, used)
                        st.info(f"**🤖 AI Insight:** {insight}")
                    else:
                        st.warning("Could not generate chart. Try rephrasing your request.")
                except Exception as e:
                    st.error(f"Error parsing AI response: {e}")

    # =====================================================
    # 5 — AI Dashboard
    # =====================================================
    with tabs[5]:
        st.header("📊 AI Dashboard Builder")
        st.write("AI auto-builds an interactive visual dashboard from your data — like Power BI / Tableau.")
        if st.button("🚀 Build My Dashboard", key="dash_btn"):
            import plotly.express as px

            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            # ---- KPI Cards ----
            st.markdown("### 📈 Key Performance Indicators")
            kpi_cols = numeric_cols[:5]
            if kpi_cols:
                kpi_row = st.columns(len(kpi_cols))
                for i, col_name in enumerate(kpi_cols):
                    with kpi_row[i]:
                        avg_val = df[col_name].mean()
                        max_val = df[col_name].max()
                        st.metric(col_name, f"{avg_val:,.2f}", delta=f"Max {max_val:,.0f}")
            st.divider()

            # ---- Row 1: Distribution + Composition ----
            st.markdown("### 📊 Distributions & Composition")
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                if numeric_cols:
                    fig = px.histogram(df, x=numeric_cols[0], nbins=30,
                        title=f"Distribution of {numeric_cols[0]}",
                        color_discrete_sequence=["#6C5CE7"], template="plotly_dark")
                    fig.update_layout(height=350, margin=dict(t=40, l=20, r=20, b=20))
                    st.plotly_chart(fig, use_container_width=True)
            with r1c2:
                if cat_cols and df[cat_cols[0]].nunique() <= 10:
                    fig = px.pie(df, names=cat_cols[0],
                        title=f"Composition: {cat_cols[0]}",
                        color_discrete_sequence=px.colors.qualitative.Prism, template="plotly_dark")
                    fig.update_layout(height=350, margin=dict(t=40, l=20, r=20, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                elif len(numeric_cols) > 1:
                    fig = px.histogram(df, x=numeric_cols[1], nbins=30,
                        title=f"Distribution of {numeric_cols[1]}",
                        color_discrete_sequence=["#a855f7"], template="plotly_dark")
                    fig.update_layout(height=350, margin=dict(t=40, l=20, r=20, b=20))
                    st.plotly_chart(fig, use_container_width=True)

            # ---- Row 2: Scatter + Correlation ----
            st.markdown("### 🔗 Relationships & Correlations")
            r2c1, r2c2 = st.columns(2)
            with r2c1:
                if len(numeric_cols) >= 2:
                    color_col = cat_cols[0] if cat_cols and df[cat_cols[0]].nunique() <= 8 else None
                    fig = px.scatter(df.sample(min(1000, len(df))), x=numeric_cols[0], y=numeric_cols[1],
                        color=color_col, title=f"{numeric_cols[0]} vs {numeric_cols[1]}",
                        color_discrete_sequence=px.colors.qualitative.Prism,
                        template="plotly_dark", opacity=0.6)
                    fig.update_layout(height=350, margin=dict(t=40, l=20, r=20, b=20))
                    st.plotly_chart(fig, use_container_width=True)
            with r2c2:
                if len(numeric_cols) >= 2:
                    corr = df[numeric_cols[:8]].corr()
                    fig = px.imshow(corr, text_auto=".2f", title="Correlation Matrix",
                        color_continuous_scale="RdBu_r", template="plotly_dark")
                    fig.update_layout(height=350, margin=dict(t=40, l=20, r=20, b=20))
                    st.plotly_chart(fig, use_container_width=True)

            # ---- Row 3: Box + Bar ----
            st.markdown("### 📉 Outliers & Categories")
            r3c1, r3c2 = st.columns(2)
            with r3c1:
                if numeric_cols:
                    fig = px.box(df, y=numeric_cols[:4], title="Outlier Analysis",
                        color_discrete_sequence=px.colors.qualitative.Prism, template="plotly_dark")
                    fig.update_layout(height=350, margin=dict(t=40, l=20, r=20, b=20))
                    st.plotly_chart(fig, use_container_width=True)
            with r3c2:
                if cat_cols:
                    top_vals = df[cat_cols[0]].value_counts().head(10).reset_index()
                    top_vals.columns = [cat_cols[0], "Count"]
                    fig = px.bar(top_vals, x="Count", y=cat_cols[0], orientation='h',
                        title=f"Top {cat_cols[0]} Categories",
                        color="Count", color_continuous_scale="Purples", template="plotly_dark")
                    fig.update_layout(height=350, margin=dict(t=40, l=20, r=20, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                elif len(numeric_cols) > 2:
                    fig = px.histogram(df, x=numeric_cols[2], nbins=30,
                        title=f"Distribution of {numeric_cols[2]}",
                        color_discrete_sequence=["#ec4899"], template="plotly_dark")
                    fig.update_layout(height=350, margin=dict(t=40, l=20, r=20, b=20))
                    st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.markdown("### 🤖 AI Dashboard Summary")
            with st.spinner("🤖 AI analyzing your dashboard..."):
                dash_summary = ai_smart_summary(df)
            st.markdown(dash_summary)

    # =====================================================
    # 6 — Auto Insights
    # =====================================================
    with tabs[6]:
        st.header("⚡ AI Auto Insights")
        st.write("AI automatically discovers the most interesting patterns, correlations, and hidden findings.")
        if st.button("⚡ Discover Insights", key="auto_btn"):
            with st.spinner("🤖 AI is discovering insights..."):
                auto = ai_auto_insights(df)
            st.markdown(auto)

    # =====================================================
    # 7 — Business Insights
    # =====================================================
    with tabs[7]:
        st.header("💼 AI Business Insights")
        st.write("AI analyzes your data from a business perspective — KPIs, revenue, growth, strategy, and risks.")
        if st.button("💼 Generate Business Insights", key="biz_btn"):
            with st.spinner("🤖 AI is performing business analysis..."):
                biz = ai_business_insights(df)
            st.markdown(biz)

    # =====================================================
    # 8 — Feature Engineering
    # =====================================================
    with tabs[8]:
        st.header("💡 AI Feature Engineering")
        st.write("AI will suggest new features you can derive from your dataset.")
        if st.button("💡 Suggest Features", key="feature_btn"):
            with st.spinner("🤖 AI is thinking about new features..."):
                features = ai_feature_engineering(df)
            st.markdown(features)

    # =====================================================
    # 9 — Prediction
    # =====================================================
    with tabs[9]:
        st.header("🔮 Prediction Preview")
        st.write("Select a target column and get a quick ML prediction accuracy.")
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for prediction.")
        else:
            target = st.selectbox("Select target column", numeric_cols, key="pred_target")
            feature_cols = [c for c in numeric_cols if c != target]
            if st.button("🚀 Run Prediction", key="pred_btn"):
                with st.spinner("Training model..."):
                    from sklearn.model_selection import train_test_split
                    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                    from sklearn.metrics import (r2_score, mean_absolute_error,
                                                 accuracy_score, f1_score)
                    import matplotlib.pyplot as plt

                    pred_df = df[feature_cols + [target]].dropna()
                    if len(pred_df) < 10:
                        st.error("Not enough data after removing missing values.")
                    else:
                        X, y = pred_df[feature_cols], pred_df[target]
                        n_unique = y.nunique()
                        is_classification = n_unique <= 15
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                        if is_classification:
                            st.info(f"🎯 Detected **classification** task ({n_unique} classes)")
                            model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
                            model_rf.fit(X_train, y_train)
                            y_pred = model_rf.predict(X_test)
                            acc = accuracy_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred, average="weighted")
                            c1, c2 = st.columns(2)
                            c1.metric("Accuracy", f"{acc:.4f}")
                            c2.metric("F1 Score (weighted)", f"{f1:.4f}")
                            metrics_text = f"Accuracy={acc:.4f}, F1={f1:.4f}"
                        else:
                            st.info("📈 Detected **regression** task (continuous target)")
                            model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
                            model_rf.fit(X_train, y_train)
                            y_pred = model_rf.predict(X_test)
                            r2 = r2_score(y_test, y_pred)
                            mae = mean_absolute_error(y_test, y_pred)
                            c1, c2 = st.columns(2)
                            c1.metric("R² Score", f"{r2:.4f}")
                            c2.metric("Mean Absolute Error", f"{mae:.4f}")
                            metrics_text = f"R²={r2:.4f}, MAE={mae:.4f}"

                        st.subheader("📊 Feature Importance")
                        importances = model_rf.feature_importances_
                        feat_imp = pd.DataFrame({"Feature": feature_cols, "Importance": importances}).sort_values("Importance", ascending=True)
                        fig, ax = plt.subplots(figsize=(8, max(3, len(feature_cols) * 0.4)))
                        ax.barh(feat_imp["Feature"], feat_imp["Importance"], color="#6C5CE7")
                        ax.set_xlabel("Importance")
                        ax.set_title(f"Feature Importance for Predicting '{target}'")
                        st.pyplot(fig)

                        task_type = "classification" if is_classification else "regression"
                        with st.spinner("🤖 AI analyzing results..."):
                            interpretation = ask_gemini(df, f"I trained a Random Forest {task_type} model to predict '{target}' using {feature_cols}. {metrics_text}. Top features: {feat_imp.to_string()}. Interpret these results and suggest improvements.")
                        st.markdown(f"**🤖 AI Interpretation:**\n\n{interpretation}")

    # =====================================================
    # 10 — AI Chat
    # =====================================================
    with tabs[10]:
        st.header("💬 Ask AI About Your Data")
        st.write("Ask any question about your dataset and get AI-powered answers.")

        # Chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if prompt := st.chat_input("Ask anything about your dataset..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🤖 Thinking..."):
                    response = ask_gemini(df, prompt)
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    # =====================================================
    # 11 — Report
    # =====================================================
    with tabs[11]:
        st.header("📝 Auto Report Generation")
        st.write("Generate a comprehensive, downloadable HTML report of your dataset.")
        if st.button("📝 Generate Report", key="report_btn"):
            with st.spinner("🤖 AI is writing your report..."):
                report_html = ai_generate_report(df)
            full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>EDA Report</title>
<style>body{{font-family:'Segoe UI',sans-serif;max-width:900px;margin:40px auto;padding:20px;background:#f8f9fa;color:#333}}
h2{{color:#6C5CE7;border-bottom:2px solid #6C5CE7;padding-bottom:8px}}
table{{border-collapse:collapse;width:100%;margin:15px 0}}th,td{{border:1px solid #ddd;padding:10px;text-align:left}}
th{{background:#6C5CE7;color:#fff}}ul{{padding-left:20px}}li{{margin:5px 0}}</style>
</head><body><h1>🤖 InsightForge AI Report</h1>{report_html}</body></html>"""
            st.components.v1.html(full_html, height=600, scrolling=True)
            st.download_button("📥 Download HTML Report", data=full_html, file_name="eda_report.html", mime="text/html", key="download_report")

else:
    inject_html("templates/landing.html")