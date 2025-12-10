# Remote Sensing Processing Pipeline

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Package Status](https://img.shields.io/badge/status-experimental-orange.svg)](#)
[![Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-required-red.svg)](#)
[![CI](https://github.com/<your-org>/<your-repo>/workflows/CI/badge.svg)](https://github.com/<your-org>/<your-repo>/actions)

---

## 🌍 Overview
This repository provides a complete Sentinel‑2 & MODIS preprocessing workflow:
- Cloud masking
- Spectral indices computation
- Sentinel‑2 → MODIS reprojection
- Metadata extraction
- Date matching & merging

---

## ⚙️ Quick Start
```bash
git clone https://github.com/varunkt91/Clark_Albedo
cd <your-repo>
pip install -r requirements.txt
earthengine authenticate
```
Edit `config.py` and run:
```bash
jupyter notebook Main_code.ipynb
```

---

## 🔁 How it Works (Flow)
```mermaid
flowchart TB
  A[Define ROI & Dates (config.py)] --> B[Load Sentinel-2 & MODIS]
  B --> C[Apply Cloud Masking]
  C --> D[Compute Spectral Indices]
  D --> E[Match Dates & Merge Collections]
  E --> F[Reproject to MODIS Grid]
  F --> G[Extract Metadata to DataFrame]
  G --> H[Export]
  style A fill:#f9f,stroke:#333,stroke-width:2px
  style H fill:#bbf,stroke:#333,stroke-width:2px
```

---

## 📊 Example Outputs
| NDVI Map | Observed vs Predicted |
|---|---|
| ![NDVI](examples/example_ndvi_map.png) | ![Scatter](examples/example_scatter.png) |

_All example assets stored under `/examples`._

---

## 📚 Modules
| Module | Description |
|---|---|
| `cloud_mask.py` | Cloud mask filtering for Sentinel‑2 |
| `Indices.py` | NDVI, EVI, LSWI & spectral metrics |
| `merge_S2_MODIS.py` | Temporal matching & merge |
| `modis_projection.py` | Reprojection tools (S2 → MODIS) |
| `meta_data_to_pandas.py` | Export metadata to pandas DataFrame |
| `Date_format.py` | Formatting utilities |

---

## 🧪 Documentation Auto‑Generation
```bash
pip install pdoc3
pdoc --html --output-dir docs --force .
```
GitHub Pages deployment available in `.github/workflows/docs.yml`.

---

## 🖼️ Export Flowchart for AGU Poster
```bash
npm install -g @mermaid-js/mermaid-cli
mmdc -i flowchart.mmd -o assets/flowchart.svg
mmdc -i flowchart.mmd -o assets/flowchart.png -s 2
```

---

## 📁 Repo Structure
```bash
|-- Main_code.ipynb
|-- cloud_mask.py
|-- Indices.py
|-- merge_S2_MODIS.py
|-- modis_projection.py
|-- meta_data_to_pandas.py
|-- Date_format.py
|-- examples/
|-- assets/
|-- docs/
|-- config.py
|-- requirements.txt
|-- README.md
```

---

## 📝 License
MIT

---

For questions or contributions, open an Issue or Pull Request.

