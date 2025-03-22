# Automated Search Strategy Generator & Resume Analyzer

This repository hosts the final project for the CM3070 module at the University of London. The application uses a fine-tuned T5 model to transform job descriptions into Boolean queries, which are then enhanced with semantic similarity scoring to evaluate candidate resumes. It includes a PyQt5-based interface for generating queries, searching local candidate data, and exporting resumes.

---

## Key Features

- **Boolean Query Generation:**  
  Converts job descriptions into optimized Boolean search statements using a fine-tuned T5 model.

- **Hybrid Resume Analysis:**  
  Combines Boolean logic with semantic similarity measures to rank resumes by relevance.

- **User-Friendly GUI:**  
  A PyQt5 application that lets you:
  - Paste or type a job description.
  - Generate and refine Boolean queries.
  - Search through a local candidate resume dataset (`candidate_resumes.csv`).
  - View and export resumes.

- **Optional Model Training:**  
  Fine-tune the T5 model with your own dataset (`Cleaned_Dataset.csv`) if desired.

---

## Install Dependencies


pip install -r requirements.txt


Ensure you have Python 3.7+ installed. The pinned versions in `requirements.txt` provide a consistent environment.

---

## Usage

### Running the Application

To launch the GUI, run:


python automated_search_strategy_generator.py


- Paste a job description into the text box.
- Click **Generate Boolean Query** to see the suggested query.
- Refine the query if needed.
- Optionally, click **Search Local Candidates** to match the query against your local CSV of candidate resumes.

### Training or Fine-Tuning the Model

To fine-tune the T5 model with a custom dataset:

- Place your data in `Cleaned_Dataset.csv`.
- Run:


python automated_search_strategy_generator.py train


The fine-tuned model will be saved in the `fine-tuned-t5` directory.

---

## File Structure

- **automated_search_strategy_generator.py**  
  Main script for both the GUI and model training routines.

- **fine-tuned-t5/**  
  Directory for the fine-tuned T5 model checkpoints.

- **local_t5_base/**  
  Base T5 model checkpoint directory.

- **candidate_resumes.csv**  
  Sample file containing candidate resume data for local searching.

- **Cleaned_Dataset.csv**  
  Training dataset for fine-tuning the T5 model.

- **favicon.ico**  
  Icon file used by the PyQt5 interface.

---

## Acknowledgements

**Academic Context:**  
Developed for the CM3070 module at the University of London, this project demonstrates the practical application of  NLP techniques for recruitment and resume analysis.

**Libraries & Tools:**
- PyTorch
- Hugging Face Transformers
- PyQt5
- pandas
- python-docx

