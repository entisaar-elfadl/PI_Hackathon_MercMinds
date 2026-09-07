# Setup Instructions

The repository contains two workflows: a Python modelling pipeline and a browser-based evaluation dashboard.

## Requirements

- Python 3.10 or newer for the modelling pipeline.
- Node.js 18 or newer and npm for the dashboard.
- Access to the challenge data in `assets/dataset/`.

## Python Environment

Create and activate a virtual environment, then install the libraries used by the modelling scripts and notebooks:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install numpy pandas scikit-learn jupyter matplotlib seaborn
```

The final modelling pipeline expects `train.csv` and `test.csv` in `assets/dataset/`. Run it from the repository root:

```powershell
python demo\final_submission.py
```

The script writes `final_winning_submission.csv` to the current working directory.

## Dashboard Installation

Install the frontend dependencies from the `src` directory:

```powershell
cd src
npm install
```

Start the local development server:

```powershell
npm run dev
```

Open `http://localhost:3000` in a browser. To verify the production bundle, run:

```powershell
npm run lint
npm run build
```