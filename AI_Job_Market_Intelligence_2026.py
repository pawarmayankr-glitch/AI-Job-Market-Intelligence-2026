# AI Job Market Intelligence 2026
# Data Analytics Project
# Generated from the Google Colab notebook

## Title
# AI Job Market Intelligence 2026

### Automation vs Traditional Roles — Skills, Salaries, Demand & Market Trends

#Project Type:** Data Analytics
#Dataset Source:** Kaggle
#Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn
#Platform:** Google Colab

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('ai_job_market_dataset.csv')

df.head()

print("Number of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])

df.info()

df.columns.tolist()

df.isnull().sum()

print("Number of duplicate rows:", df.duplicated().sum())

df.dtypes

# Fill missing categorical values
df['company'] = df['company'].fillna('Unknown')
df['contract_type'] = df['contract_type'].fillna('Unknown')
df['contract_time'] = df['contract_time'].fillna('Unknown')

# Calculate missing average salary using minimum and maximum salary
df['salary_avg'] = df['salary_avg'].fillna(
    (df['salary_min'] + df['salary_max']) / 2
)

# Check remaining missing values
df.isnull().sum()

print("Total duplicate rows:", df.duplicated().sum())
print("\nRemaining missing values:")
print(df.isnull().sum())

print("Duplicate rows:", df.duplicated().sum())

print("\nMissing values:")
print(df.isnull().sum())

ai_counts = df['is_ai_related'].value_counts()

print(ai_counts)

plt.figure(figsize=(7,5))

sns.countplot(data=df, x='is_ai_related')

plt.title('AI-Related vs Traditional Job Postings')
plt.xlabel('AI Related')
plt.ylabel('Number of Job Postings')

plt.show()

ai_percentage = df['is_ai_related'].value_counts(normalize=True) * 100

print(ai_percentage.round(2))

category_counts = df['category'].value_counts()

print(category_counts)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=category_counts.values,
    y=category_counts.index
)

plt.title('Job Postings by Category')
plt.xlabel('Number of Job Postings')
plt.ylabel('Job Category')

plt.show()

company_counts = df['company'].value_counts().head(10)

print(company_counts)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=company_counts.values,
    y=company_counts.index
)

plt.title('Top 10 Companies by Job Postings')
plt.xlabel('Number of Job Postings')
plt.ylabel('Company')

plt.show()

country_counts = df['country_name'].value_counts()

print(country_counts)

plt.figure(figsize=(8, 5))

sns.barplot(
    x=country_counts.values,
    y=country_counts.index
)

plt.title('Job Postings by Country')
plt.xlabel('Number of Job Postings')
plt.ylabel('Country')

plt.show()

employment_counts = df['contract_time'].value_counts()

print(employment_counts)

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    y='contract_time',
    order=df['contract_time'].value_counts().index
)

plt.title('Job Postings by Employment Type')
plt.xlabel('Number of Job Postings')
plt.ylabel('Employment Type')

plt.show()

print(df[['salary_min', 'salary_max', 'salary_avg', 'currency']].describe())

print(df['currency'].value_counts())

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x='salary_avg',
    bins=30,
    kde=True
)

plt.title('Distribution of Average Salary')
plt.xlabel('Average Salary')
plt.ylabel('Number of Job Postings')

plt.show()

df['currency'].value_counts()

df.groupby('currency')['salary_avg'].agg(['count', 'mean', 'median', 'min', 'max']).round(2)

salary_by_ai = df.groupby(
    ['currency', 'is_ai_related']
)['salary_avg'].agg(['count', 'mean', 'median']).round(2)

salary_by_ai

plt.figure(figsize=(10, 6))

sns.barplot(
    data=df,
    x='currency',
    y='salary_avg',
    hue='is_ai_related'
)

plt.title('Average Salary: AI-Related vs Traditional Jobs')
plt.xlabel('Currency')
plt.ylabel('Average Salary')

plt.show()

keyword_counts = df['search_keyword'].value_counts().head(15)

print(keyword_counts)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=keyword_counts.values,
    y=keyword_counts.index
)

plt.title('Top 15 Job Search Keywords')
plt.xlabel('Number of Job Postings')
plt.ylabel('Search Keyword')

plt.show()

location_counts = df['location'].value_counts().head(15)

print(location_counts)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=location_counts.values,
    y=location_counts.index
)

plt.title('Top 15 Job Locations')
plt.xlabel('Number of Job Postings')
plt.ylabel('Location')

plt.show()

import re

skill_patterns = {
    'Python': r'\bpython\b',
    'SQL': r'\bsql\b',
    'Excel': r'\bexcel\b',
    'Power BI': r'\bpower\s*bi\b',
    'Tableau': r'\btableau\b',
    'Machine Learning': r'\bmachine learning\b',
    'Artificial Intelligence': r'\bartificial intelligence\b',
    'Deep Learning': r'\bdeep learning\b',
    'AWS': r'\baws\b',
    'Azure': r'\bazure\b',
    'Google Cloud': r'\bgoogle cloud\b',
    'Java': r'\bjava\b',
    'JavaScript': r'\bjavascript\b',
    'C++': r'\bc\+\+\b',
    'R': r'(?<![A-Za-z])R(?![A-Za-z])',
    'Docker': r'\bdocker\b',
    'Kubernetes': r'\bkubernetes\b',
    'Git': r'\bgit\b',
    'TensorFlow': r'\btensorflow\b',
    'PyTorch': r'\bpytorch\b'
}

skill_counts = {}

for skill, pattern in skill_patterns.items():
    skill_counts[skill] = df['description'].str.contains(
        pattern,
        case=False,
        na=False,
        regex=True
    ).sum()

skill_counts = pd.Series(skill_counts).sort_values(ascending=False)

print(skill_counts)

top_skills = skill_counts[skill_counts > 0].head(15)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_skills.values,
    y=top_skills.index
)

plt.title('Top Technical Skills Mentioned in Job Descriptions')
plt.xlabel('Number of Job Postings')
plt.ylabel('Skill')

plt.show()

ai_skill_counts = {}

for skill, pattern in skill_patterns.items():
    if skill == 'R':
        continue

    ai_skill_counts[skill] = {
        'AI-Related': df.loc[
            df['is_ai_related'] == True, 'description'
        ].str.contains(pattern, case=False, na=False, regex=True).sum(),

        'Traditional': df.loc[
            df['is_ai_related'] == False, 'description'
        ].str.contains(pattern, case=False, na=False, regex=True).sum()
    }

ai_skill_df = pd.DataFrame(ai_skill_counts).T

ai_skill_df = ai_skill_df.sort_values(
    by='AI-Related',
    ascending=False
)

ai_skill_df

ai_skill_df.head(10).plot(
    kind='bar',
    figsize=(12, 6)
)

plt.title('Skill Demand: AI-Related vs Traditional Jobs')
plt.xlabel('Technical Skill')
plt.ylabel('Number of Job Postings')
plt.xticks(rotation=45)
plt.legend(title='Job Type')

plt.show()

# ==========================================
# KEY FINDINGS - DATA SUMMARY
# ==========================================

print("========== DATASET SUMMARY ==========")
print("Total Job Postings:", len(df))

# AI vs Traditional
ai_counts = df['is_ai_related'].value_counts(dropna=False)
print("\n========== AI vs TRADITIONAL ==========")
print(ai_counts)

ai_percentage = df['is_ai_related'].mean() * 100
print(f"AI-related Jobs: {ai_percentage:.2f}%")
print(f"Traditional Jobs: {100 - ai_percentage:.2f}%")

# Job Category
print("\n========== TOP JOB CATEGORIES ==========")
print(df['category'].value_counts().head(5))

# Companies
print("\n========== TOP COMPANIES ==========")
print(df['company'].value_counts().head(5))

# Countries
print("\n========== TOP COUNTRIES ==========")
print(df['country_name'].value_counts().head(5))

# Employment Type
print("\n========== EMPLOYMENT TYPES ==========")
print(df['contract_type'].value_counts().head(10))

# Salary
print("\n========== SALARY ANALYSIS ==========")
print("Average Salary:", df['salary_avg'].mean())
print("Median Salary:", df['salary_avg'].median())
print("Minimum Salary:", df['salary_avg'].min())
print("Maximum Salary:", df['salary_avg'].max())

# AI vs Traditional Salary
print("\n========== AI vs TRADITIONAL SALARY ==========")
print(
    df.groupby('is_ai_related')['salary_avg']
      .agg(['count', 'mean', 'median'])
      .round(2)
)

# Job Locations
print("\n========== TOP JOB LOCATIONS ==========")
print(df['location'].value_counts().head(5))

# Search Keywords
print("\n========== TOP SEARCH KEYWORDS ==========")
print(df['search_keyword'].value_counts().head(10))

