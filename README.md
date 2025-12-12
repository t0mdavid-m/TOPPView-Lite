# TOPPView-Lite

A lightweight web-based viewer for mass spectrometry data.

## Features

- **Peak Map**: Interactive 2D heatmap with zoom-based resolution
- **Spectrum View**: Click to view individual mass spectra
- **Data Tables**: Browse spectra and peaks with sorting/filtering
- **Identification Support**: Load idXML files for peptide sequence visualization
- **Ion Mobility**: FAIMS/TIMS data support
- **Fast Loading**: Preprocessed parquet caching for instant visualization

## Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/t0mdavid-m/TOPPView-Lite.git
   cd TOPPView-Lite
   ```

2. **Create environment and install dependencies**
   ```bash
   conda create -n toppview-lite python=3.10 -y
   conda activate toppview-lite
   pip install -r requirements.txt
   ```

3. **Launch the app**
   ```bash
   streamlit run app.py
   ```

## Docker Deployment

```bash
# Build and run with docker-compose
docker-compose up -d --build

# Or build manually
docker build -t toppview-lite:latest .
docker run -p 8501:8501 toppview-lite:latest
```

## Windows Installer

Windows MSI installers are automatically built via GitHub Actions on each release. Download from the [Releases](https://github.com/t0mdavid-m/TOPPView-Lite/releases) page.

## Supported File Formats

- **mzML**: Mass spectrometry data (MS1 and MS2)
- **idXML**: Peptide identifications (optional, for sequence visualization)

## Citation

Please cite:

Müller, T. D., Siraj, A., et al. OpenMS WebApps: Building User-Friendly Solutions for MS Analysis. Journal of Proteome Research (2025). [https://doi.org/10.1021/acs.jproteome.4c00872](https://doi.org/10.1021/acs.jproteome.4c00872)

## References

- Pfeuffer, J., Bielow, C., Wein, S. et al. OpenMS 3 enables reproducible analysis of large-scale mass spectrometry data. Nat Methods 21, 365–367 (2024). [https://doi.org/10.1038/s41592-024-02197-7](https://doi.org/10.1038/s41592-024-02197-7)

- Röst HL, Schmitt U, Aebersold R, Malmström L. pyOpenMS: a Python-based interface to the OpenMS mass-spectrometry algorithm library. Proteomics. 2014 Jan;14(1):74-7. [https://doi.org/10.1002/pmic.201300246](https://doi.org/10.1002/pmic.201300246)
