import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def correlation_heatmap(df, columns):

    numeric_df = df[columns].select_dtypes(include=np.number)

    if numeric_df.shape[1] < 2:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")

    return fig


def histogram(df, col):

    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax, color="#6C5CE7")
    ax.set_title(f"Distribution of {col}")

    return fig


def boxplot(df, col):

    fig, ax = plt.subplots()
    sns.boxplot(x=df[col], ax=ax, color="#00B894")
    ax.set_title(f"Boxplot of {col}")

    return fig


def scatter_plot(df, col_x, col_y):

    fig, ax = plt.subplots()
    ax.scatter(df[col_x], df[col_y], alpha=0.6, color="#E17055", edgecolors="w")
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    ax.set_title(f"{col_x} vs {col_y}")

    return fig


def line_chart(df, col):

    fig, ax = plt.subplots()
    ax.plot(df[col].values, color="#0984E3", linewidth=1.5)
    ax.set_xlabel("Index")
    ax.set_ylabel(col)
    ax.set_title(f"Line Chart of {col}")

    return fig


def pie_chart(df, col, top_n=8):

    counts = df[col].value_counts().head(top_n)
    fig, ax = plt.subplots()
    colors = sns.color_palette("Set2", len(counts))
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=colors, startangle=90)
    ax.set_title(f"Distribution of {col}")

    return fig
