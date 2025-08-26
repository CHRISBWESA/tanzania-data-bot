# analysis.py
import streamlit as st # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import pandas as pd # type: ignore

sns.set_theme(style="whitegrid")

# Example internal dataset for demonstration
DATA = pd.DataFrame({
    "Year": [2018, 2019, 2020, 2021, 2022],
    "Population": [55, 56, 57, 58, 59],
    "GDP": [60, 62, 61, 63, 65],
    "Region": ["North", "South", "East", "West", "Central"],
    "Income": [1000, 1200, 1100, 1300, 1250]
})

def trends_line_chart(x_col, y_col):
    st.subheader(f"📈 Trend of {y_col} over {x_col}")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(data=DATA, x=x_col, y=y_col, marker="o", ax=ax)
    st.pyplot(fig)

def category_bar_chart(cat_col, val_col):
    st.subheader(f"📊 Comparison of {val_col} by {cat_col}")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(data=DATA, x=cat_col, y=val_col, palette="viridis", ax=ax)
    st.pyplot(fig)

def proportion_pie_chart(cat_col):
    st.subheader(f"🥧 Proportion of {cat_col}")
    fig, ax = plt.subplots(figsize=(6,6))
    DATA[cat_col].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
    st.pyplot(fig)

def summary_statistics(num_col):
    st.subheader(f"📌 Summary statistics for {num_col}")
    stats = DATA[num_col].describe()
    st.write(stats)
