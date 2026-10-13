# Salary Fairness Analysis

**Machine learning and data mining analysis of salary fairness across companies, occupations, and geographic locations in the United States.**

This project develops a machine learning pipeline for analyzing compensation fairness using job-posting data, occupational wage benchmarks, company information, and state-level economic data. The goal is to identify whether listed salaries are relatively **underpaid, fair, or generous** compared with typical compensation for the same occupation and state, and to explore the factors and patterns associated with salary differences.

The project combines data cleaning, feature engineering, exploratory data analysis, regression, dimensionality reduction, and clustering.

## Project Overview

Salaries vary substantially across occupations, companies, and locations, and a higher nominal salary does not necessarily mean that a position is more competitive relative to its local labor market.

To make salaries more comparable, this project combines:

- **100,000+ U.S. job postings** containing salary, company, occupation, skills, and location information
- **Bureau of Labor Statistics (BLS)** occupational wage benchmarks by occupation and state
- **Bureau of Economic Analysis (BEA)** Personal Consumption Expenditure data by state
- Company-level attributes including sector, employee size, revenue size, company score, and review count

The resulting dataset is used for both supervised and unsupervised machine learning experiments.

## Salary Fairness Index

A standardized fairness index is calculated by comparing the average listed salary for a job with the BLS average annual wage for the corresponding occupation and state:

\[
\text{Fairness Index} =
\frac{\text{Listed Annual Mean Salary}}
{\text{BLS Annual Mean Wage}}
\]

Jobs are then categorized as:

| Fairness Index | Classification |
|---|---|
| `< 0.90` | Underpaid |
| `0.90 – 1.10` | Fair |
| `> 1.10` | Generous |

This creates a common benchmark for comparing compensation across different occupations and geographic locations.

## Machine Learning Methods

### Regression

The fairness index is treated as a continuous target variable. The project evaluates:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net Regression

An **80/20 train-test split** is used for model evaluation. Missing values are imputed using statistics from the training data, and numerical features are standardized while avoiding data leakage.

### Dimensionality Reduction

Two dimensionality-reduction techniques are evaluated:

- Principal Component Analysis (**PCA**)
- Truncated Singular Value Decomposition (**SVD**)

PCA and SVD are combined with Elastic Net regression to investigate whether a lower-dimensional representation can preserve predictive performance while reducing model complexity.

### Clustering

The project also investigates natural groupings within the salary data using:

- K-Means
- DBSCAN

Clustering quality is evaluated using metrics including:

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score
- Sum of Squared Error (SSE)

PCA is additionally evaluated as a preprocessing step before clustering.

## Key Results

### Regression

The linear models were able to explain approximately **39% of the variation in the salary fairness index**.

| Model | Approx. R² | Approx. RMSE |
|---|---:|---:|
| Linear Regression | 0.3897 | 0.2455 |
| Elastic Net | **0.3898** | **0.2455** |

Ridge and Lasso produced similar results, while **Elastic Net achieved the best overall regression performance**, although the differences among the linear models were small.

### PCA and SVD

Dimensionality reduction reduced the number of input features but did not outperform the original feature set for regression.

- Best PCA + Elastic Net R²: approximately **0.3860** with 70 components
- Best SVD + Elastic Net R²: approximately **0.3835**

This suggests that some predictive information was lost when compressing the feature space.

### Clustering

For K-Means, the highest silhouette score occurred at **k = 2**.

Applying PCA before K-Means improved the silhouette score from approximately:

**0.144 → 0.158**

DBSCAN performed substantially better:

- **DBSCAN Silhouette Score:** ~0.425
- **DBSCAN Davies-Bouldin Score:** ~0.802

These results suggest that the salary data is better represented by **density-based groupings** than by the spherical clusters assumed by K-Means.

## Data Processing Pipeline

The project uses a multi-stage data pipeline:

1. **Clean BLS wage data**
   - Retain relevant state-level occupation records
   - Handle suppressed or unavailable wage values
   - Convert hourly wages to annual estimates where necessary

2. **Clean BEA expenditure data**
   - Retain state-level Personal Consumption Expenditure information

3. **Clean company data**
   - Standardize company names
   - Convert company size and revenue categories into numerical values
   - Aggregate duplicate company entries

4. **Clean job-posting data**
   - Remove extreme salary outliers
   - Standardize locations and company names
   - Process skills
   - Map job groups to BLS occupation codes

5. **Merge datasets**
   - Join jobs with company information
   - Match jobs to BLS occupation/state wage benchmarks
   - Merge state-level expenditure information

6. **Engineer salary fairness features**
   - Calculate the fairness index
   - Assign underpaid, fair, and generous labels

7. **Preprocess features**
   - One-hot encode categorical variables
   - Convert the most common skills into binary features
   - Handle missing values
   - Standardize numerical variables

8. **Run regression and clustering experiments**

Approximately **93.4% of job titles were successfully mapped** to BLS occupational classifications.

## Repository Structure

```text
Salary-Fairness-Analysis/
│
├── part1-emily/
│   ├── cleaning.ipynb
│   ├── merging.ipynb
│   ├── preprocessing.ipynb
│   ├── split_and_scale.ipynb
│   └── scale.ipynb
│
├── part2-robert/
│   └── Robert_DataAnalysis.ipynb
│
├── part3-ryan/
│   ├── regression.ipynb
│   └── clustering.ipynb
│
├── data/
│   ├── raw/
│   └── cleaned/
│       └── clustering/
│
├── models/
└── README.md
```

The submitted project archive also includes the final written report, presentation slides, and recorded presentation.

## Notebook Workflow

The preprocessing notebooks should be run in approximately the following order:

```text
cleaning.ipynb
      ↓
merging.ipynb
      ↓
preprocessing.ipynb
      ↓
 ┌─────────────────────┐
 ↓                     ↓
split_and_scale.ipynb  scale.ipynb
 ↓                     ↓
Regression             Clustering
```

- `split_and_scale.ipynb` prepares the dataset for regression.
- `scale.ipynb` prepares the dataset for clustering.
- `Robert_DataAnalysis.ipynb` contains exploratory data analysis and visualization.
- `regression.ipynb` evaluates regression and dimensionality-reduction methods.
- `clustering.ipynb` evaluates K-Means, DBSCAN, and PCA-based clustering.

> **Note:** The notebooks use relative paths such as `data/raw/`, `data/cleaned/`, and `models/`. If the notebooks are kept inside the contributor subdirectories shown above, these paths may need to be adjusted depending on the Jupyter working directory.

## Datasets

### 1. U.S. Data Jobs Salaries Dataset

Job postings and company information, including salary, occupation, skills, company, and location data.

Source:  
https://www.kaggle.com/datasets/juanmerinobermejo/data-jobs-dataset

### 2. Bureau of Labor Statistics Occupational Employment and Wage Statistics

State-level employment and occupational wage benchmarks.

Source:  
https://www.bls.gov/oes/tables.htm

The project uses the **May 2023 state-level Occupational Employment and Wage Statistics** data.

### 3. Bureau of Economic Analysis Personal Consumption Expenditures

State-level Personal Consumption Expenditure data.

Source:  
https://www.bea.gov/data/consumer-spending/state

The raw datasets are not included directly in the project archive and should be downloaded from their respective sources.

## Technologies

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

Machine learning techniques include:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net
- PCA
- Truncated SVD
- K-Means
- DBSCAN

## Challenges and Limitations

One of the primary challenges was mapping real-world job titles to standardized BLS occupational classifications. Job titles can vary substantially between companies even when they describe similar positions, while BLS classifications may group jobs with different responsibilities and compensation levels.

The high dimensionality created by categorical variables and skill features also introduced additional computational complexity and potential overfitting.

Finally, salary fairness is influenced by factors that are not fully represented in the available datasets. The resulting fairness index should therefore be interpreted as a **relative compensation benchmark**, rather than a definitive determination of whether an individual employee is being fairly compensated.

## Future Work

Potential extensions include incorporating additional factors such as:

- Years of professional experience
- Education level
- More detailed company characteristics
- More granular geographic cost-of-living information
- Job-specific responsibilities
- Additional compensation such as bonuses and equity
- More advanced nonlinear machine learning models

These features could improve predictive accuracy and provide a more comprehensive measure of compensation fairness.

## Project Team

- **Emily Chen** — Data cleaning, merging, preprocessing, and feature preparation
- **Robert Rossi** — Exploratory data analysis and visualization
- **Ryan Vattuone** — Regression, dimensionality reduction, and clustering analysis

Developed as a **CSEN 140 Machine Learning & Data Mining** project.
