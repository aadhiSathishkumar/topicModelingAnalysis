# NMF, BERTopic, and Stacked Models

Topic modeling of Missouri S&T Electrical and Computer Engineering course descriptions (with ABET outcomes) using NMF, BERTopic, and a stacked combination of the two.

## How to run

1. Download / clone the repository.
2. Open the repository in a code editor or IDE (VS Code was used to create the program).
3. Create and activate a virtual environment:

   ```powershell
   # Windows (PowerShell or Command Prompt)
   python -m venv .venv
   .\.venv\Scripts\activate.bat
   ```

   ```bash
   # macOS / Linux
   python -m venv .venv
   source .venv/bin/activate
   ```

   If PowerShell blocks the `.ps1` activator, either run the `.bat` version above or allow scripts for the current session:

   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
   .\.venv\Scripts\Activate.ps1
   ```

4. Install dependencies and the spaCy model:

   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

5. Run a model:

   ```bash
   python nmfModel.py
   python bertopicModel.py
   python stackedModel.py
   ```
