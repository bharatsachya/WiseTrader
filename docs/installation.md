# Installation Guide

This guide will walk you through setting up the WiseTrader project locally.

## Prerequisites

*   Python 3.8+ (Recommended 3.9 or 3.10)
*   `pip` (Python package installer)

## Steps

1.  **Clone the Repository**

    First, clone the WiseTrader repository to your local machine:
    ```bash
    git clone https://github.com/bharatsachya/WiseTrader.git
    cd WiseTrader
    ```

2.  **Create a Virtual Environment (Recommended)**

    It's highly recommended to use a virtual environment to manage project dependencies. This prevents conflicts with other Python projects.
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment**

    *   **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install Dependencies**

    Install all required Python packages using `pip`. You might need to create a `requirements.txt` file first if one doesn't exist, by listing the following packages:
    
    *A `requirements.txt` file should contain:*
    ```
    streamlit
    yfinance
    numpy
    pandas
    scikit-learn
    tensorflow # or keras
    matplotlib
    seaborn
    ```

    Then install them:
    ```bash
    pip install -r requirements.txt
    ```
    If you don't have a `requirements.txt`, you can install them manually:
    ```bash
    pip install streamlit yfinance numpy pandas scikit-learn tensorflow matplotlib seaborn
    ```

    *(Note: TensorFlow is a large dependency. If you encounter issues, consider installing the CPU-only version `tensorflow-cpu` if you don't require GPU acceleration.)*

Your environment is now set up and ready to run the WiseTrader application.
