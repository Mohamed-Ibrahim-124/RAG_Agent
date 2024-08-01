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


2. **Run the project**:
    ```bash
    python -m streamlit run app.py
    ```

## Task Breakdown

**Project Duration:** 4 Working Days

**Time Spent on Each Task:**

* **Day 1: Initial Setup and Planning**
    * Time Spent: 2 hours (5:00 PM - 7:00 PM)
    * Tasks:
        * Set up the development environment.
        * Reviewed project requirements and outlined the approach.
        * Created initial project structure.

* **Day 2: Document Preprocessing**
    * Time Spent: 3 hours (5:00 PM - 8:00 PM)
    * Tasks:
        * Developed and tested document preprocessing methods for various file types (PDF, DOCX, CSV, TXT).
        * Implemented text extraction and normalization.
        * Debugged and refined preprocessing functions.

* **Day 3: Retrieval System Development and Integration**
    * Time Spent: 3.5 hours (5:00 PM - 8:30 PM)
    * Tasks:
        * Implemented and tested retrieval system using LlamaIndex and Fireworks LLM.
        * Developed custom search algorithm and integrated it with the existing system.
        * Evaluated retrieval accuracy and made necessary adjustments.

* **Day 4: Generation Model Integration, Testing, and Documentation**
    * Time Spent: 4 hours (5:00 PM - 9:00 PM)
    * Tasks:
        * Integrated the generation model with the retrieval system.
        * Conducted end-to-end testing of the RAG system.
        * Documented code and added comments.
        * Prepared the evaluation report and project documentation.

**Total Time Spent:** 12.5 hours

Total: 12.5 hours

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

