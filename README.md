# Hull Tactical Market Prediction (Team Project 4)

**CS 53744 Machine Learning Project**  
**Instructor:** Jongmin Lee  
**Due Date:** December 4, 2025

## Team Information
* **Team ID:** [Team ID]
* **Members:**
  * 20201876 나상현
  * 20222446 김현수
  * 20203009 박재현

## Project Overview
This project aims to predict daily excess returns of the S&P 500 and design a betting strategy that outperforms the benchmark while adhering to a 120% volatility constraint. The project is based on the [Hull Tactical - Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction/overview) Kaggle competition.

**Goal:**
* Predict daily excess returns.
* Allocate portfolio weights between 0 and 2.
* Maximize the Modified Sharpe Ratio.
* Maintain volatility within 1.2x of the market volatility.

## Dataset
The dataset is provided by the Kaggle competition. It includes historical market data used to train and validate the models.
* **Source:** [Kaggle Competition Link](https://www.kaggle.com/competitions/hull-tactical-market-prediction/overview)
* **Files:** `train.csv`, `test.csv` (Ensure these are placed in the `input/` directory or updated paths in the scripts).

## Project Structure

The project is organized into sequential steps corresponding to the assignment requirements:

*   **`step1.ipynb` (Baseline Model)**:
    *   Implements a simple baseline model (e.g., Regression or Momentum).
    *   Verifies the submission pipeline and metric calculation.

*   **`step2.ipynb` (Model Development)**:
    *   Explores Machine Learning algorithms (Random Forest).
    *   Implements Time-Series Cross-Validation.
    *   Focuses on predicting daily allocation weights.

*   **`step3.ipynb` (Feature Engineering)**:
    *   Exploratory Data Analysis (EDA).
    *   Generation of technical indicators (RSI, MACD, Bollinger Bands, Volatility ratios).
    *   Analysis of feature importance and selection.

*   **`step4.ipynb` (Evaluation & Backtesting)**:
    *   Comprehensive backtesting framework.
    *   Evaluation of Local Sharpe-variant scores.
    *   Visualization of cumulative returns, volatility ratios, and drawdowns.

*   **`step5.ipynb` (Final Model)**:
    *   **The Final Solution**.
    *   **Architecture:** Multi-Scale Ensemble with SE-Blocks.
    *   **Optimization:** SAM (Sharpness-Aware Minimization) and Online Learning.
    *   **Risk Management:** Volatility Scaling and Dynamic Weight Adjustment.
    *   Includes the full pipeline from data loading to inference.

*   **`nasdaq.ipynb` (Bonus Task)**:
    *   **Cross-Market Extension**: Applies the developed pipeline to the NASDAQ market.
    *   Demonstrates the robustness and transferability of the strategy.

*   **`dataset_crawler.py` (NASDAQ Data Builder)**:
    *   Downloads NASDAQ Composite and macro/ETF signals from Yahoo Finance.
    *   Converts price-based series into returns, engineers 50+ technical/macro features, and exports `nasdaq_new.csv` for the bonus analysis.

## Usage

1.  **Setup**: Download the dataset from Kaggle and place it in the appropriate directory (e.g., `./` or `/kaggle/input/hull-tactical-market-prediction`).
2.  **Run Baseline**: Open `step1.ipynb` in Jupyter/VS Code and execute all cells (or run it headlessly):
    ```bash
    jupyter nbconvert --to notebook --execute step1.ipynb
    ```
3.  **Run ML Model**: Open `step2.ipynb` and execute all cells (or run headlessly):
    ```bash
    jupyter nbconvert --to notebook --execute step2.ipynb
    ```
4.  **Explore & Train**: Open `step3.ipynb`, `step4.ipynb`, or `step5.ipynb` in Jupyter Notebook or VS Code to run the interactive analysis and training loops.
5.  **Final Inference**: `step5.ipynb` contains the final `predict` function and inference loop for generating the submission file.

## Methodology (Final Model)

Our final model (`step5.ipynb`) employs a sophisticated approach combining deep learning and online adaptation:

*   **Ensemble Learning**: Uses multiple seeds for stability.
*   **Multi-Scale Architecture**: Processes features with different time horizons (Short, Mid, Long) separately before fusion.
*   **SE-Block**: Squeeze-and-Excitation blocks for feature attention.
*   **SharpeHybridLoss**: A custom loss function that optimizes for both MSE and Sharpe Ratio directly.
*   **Online Learning**: Continuously updates the model weights and scaling parameters based on new incoming data to adapt to changing market regimes.

## Bonus Task: NASDAQ Prediction
We extended our analysis to the NASDAQ index (`nasdaq.ipynb`). By collecting relevant data and applying our core pipeline, we demonstrated that our volatility-constrained betting strategy is effective across different market indices.

## License
This project is for educational purposes as part of the CS 53744 course.
