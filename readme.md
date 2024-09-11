# Retrieval-Augmented Generation (RAG) Project

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system that combines document retrieval and language generation to provide relevant and accurate responses to user queries.

## Components

- **Document Preprocessing**: Handles text extraction, normalization, and paragraph segmentation.
- **Document Store**: Uses Redis for in-memory storage and retrieval of preprocessed documents.
- **Firework API Client**: Connects to Firework LLM service for language generation.
- **Hybrid RAG System**: Integrates retrieval and generation components.
- **RAG Tester**: Provides testing and evaluation capabilities.

## Usage

1. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2. **Set Up Firework API Key**:

    Create an account and get a free limited API key from Firework.
    Update the Firework API key value in the .streamlit/secrets.toml file:

    ``` toml
    fireworks_api_key = "YOUR_FIREWORK_API_KEY"
    ```   


3. **Run the project**:
    ```bash
    python -m streamlit run app.py
    ```
4. **Upload Documents**: Upload documents in PDF, DOCX, CSV, or TXT format using the Streamlit interface.

5. **Submit Queries**: Enter your query in the provided text input and submit to receive a generated response.

Evaluation

    Testing Procedures: Comprehensive unit tests were performed on the document preprocessing, retrieval, and generation components. Testing also included performance evaluation using simulated queries and document inputs.
    Performance Evaluation: Metrics such as accuracy, relevance scores, and user feedback were analyzed to gauge the system's effectiveness and reliability.
    Simulated User Feedback: Feedback was collected from simulated users to assess the system’s usability and performance under typical use cases.

Requirements

    Software Requirements:
        Streamlit
        PyPDF2
        Docx
        LlamaIndex
        Torch
        Fireworks (Fireworks LLM API key)
        Sentence-transformers
        Plotly
        HDBSCAN

