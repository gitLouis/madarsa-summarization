# Madarsa Questionnaire Summarization

## Overview

This project aims to summarize response for questionnaires in Hebrew, English and Arabic for a language school NGO called מדרסה (Madrase). The project utilizes NLTK, BERTopic, Dicta 2.0, Llama3.1 and other natural language processing tools.


## Installation

To set up the project, follow these steps:

1. **Install Ollama**: Download and install Ollama from [Ollama's website](https://ollama.com/download) or use Homebrew:
    ```bash
    brew install ollama
    ```

2. **Pull local LLM**: Download model, eg Hebrew-English LLM Dicta 2.0:
    ```bash
    ollama start
    ollama pull aminadaven/dictalm2.0-instruct:f16
    ```

3. **Set Up Virtual Environment**:
    ```bash
    python3 -m venv venv
    ```

4. **Activate Virtual Environment**:
    ```bash
    . venv/bin/activate
    ```

5. **Upgrade pip**:
    ```bash
    pip install --upgrade pip
    ```

6. **Install Required Packages**:
    ```bash
    pip install -r requirements.txt
    ```
