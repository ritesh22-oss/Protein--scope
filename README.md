# ProteinScope

**An Interactive AI-Assisted Protein Structure Visualization and Analysis System**

ProteinScope is a first-year computational-biology working model. It retrieves selected protein prediction records from the AlphaFold Protein Structure Database, obtains sequence metadata from UniProt, visualizes the predicted structure in 3D, and displays basic sequence and confidence analysis.

## Features
- Curated protein examples with UniProt accessions
- Live AlphaFold DB structure retrieval (PDB/mmCIF)
- UniProt sequence and organism metadata
- Interactive 3D structure visualization using py3Dmol / 3Dmol.js
- Amino-acid composition chart
- Per-residue pLDDT confidence plot when PDB B-factor data is available
- FASTA sequence download
- Explicit scientific limitations and interpretation notes

## Requirements
- Windows 10/11, macOS or Linux
- Python 3.10 or newer recommended
- Internet connection
- No dedicated GPU required

## Windows setup
1. Extract the ZIP file.
2. Open Command Prompt in the `ProteinScope_Project` folder.
3. Create a virtual environment:

   ```bash
   py -m venv .venv
   .venv\Scripts\activate
   ```

4. Install packages:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Run the application:

   ```bash
   streamlit run app.py
   ```

6. Streamlit will open the app in your browser (usually `http://localhost:8501`).

## First demonstration
1. Open **Explore proteins**.
2. Select **Human lysozyme**.
3. Click **Load protein structure**.
4. Rotate and zoom the 3D model.
5. Open the sequence and confidence tabs.
6. Repeat with myoglobin and growth hormone.

## Protein examples
| Display name | UniProt accession | Note |
|---|---|---|
| Human lysozyme | P61626 | Single-chain example |
| Human myoglobin | P02144 | Single-chain globular protein |
| Human growth hormone | P01241 | Single-chain hormone |
| Human haemoglobin subunit beta | P68871 | One chain only; not the complete tetramer |
| Human insulin precursor | P01308 | Precursor entry; not the mature two-chain molecule |

The app deliberately labels haemoglobin beta and insulin precursor to avoid presenting a single-chain prediction as the complete mature biological assembly.

## Data sources
- AlphaFold Protein Structure Database: https://alphafold.com/
- AlphaFold API: https://alphafold.ebi.ac.uk/api-doc
- UniProt: https://www.uniprot.org/
- Streamlit: https://docs.streamlit.io/
- Biopython: https://biopython.org/
- py3Dmol: https://github.com/3dmol/3Dmol.js

## Scientific interpretation
- AlphaFold models are predictions, not automatically experimentally verified structures.
- pLDDT is a per-residue local confidence score, conventionally reported on a 0–100 scale.
- High confidence does not prove that a protein is active, correctly assembled in vivo, or suitable as a drug target.
- A single-chain prediction does not represent a complete multimeric assembly.
- The app is an educational demonstrator, not a medical diagnostic or drug-design tool.

## Troubleshooting
- **No model found:** confirm the UniProt accession and try another example.
- **Network error:** check internet access and retry; public services may occasionally be unavailable.
- **3D view is blank:** update the browser, allow WebGL, and rerun the app.
- **Package installation fails:** use Python 3.10/3.11 in a fresh virtual environment and install requirements again.
- **Streamlit command not found:** activate the virtual environment and run `python -m streamlit run app.py`.

## Project structure
```text
ProteinScope_Project/
├── app.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT.docx
├── PRESENTATION.pptx
├── docs/
│   ├── project_report.md
│   ├── viva_questions.md
│   └── demo_script.md
└── tests/
    └── test_sequence_utils.py
```

## Academic integrity and customization
This is a project starter package. Before submission, students should run the app themselves, verify each live example, add their names/registration details/team contributions, capture their own screenshots, and revise the report to reflect the actual tested results and instructor's format.
