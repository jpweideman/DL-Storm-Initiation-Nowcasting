
Before training the LSTM model, extract the required data:

```bash
python Data_Extraction.py

```

## Data Folder Structure and Workflow

- `data/raw/`: Place your original raw data files here (e.g., `.h5` radar files). These files are never overwritten.
- `data/processed/`: This is where all processed data files go (e.g., `.npy`, `.json`). All training scripts use files from this folder by default.

**Workflow:**
1. Place your raw `.h5` files in `data/raw/`.
2. Run the data processing scripts in `src/data/` to generate `.npy` and `.json` files in `data/processed/`.
3. Use the training scripts, which by default read from `data/processed/`.

This structure ensures reproducibility and keeps raw and processed data clearly separated.
