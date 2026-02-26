# B.1 Quellcode und Versionierung

Der vollständige Quellcode ist öffentlich verfügbar unter:
`https://github.com/eybmits/parameterized-hamiltonians-id`

Für die Reproduktion der Ergebnisse dieser Arbeit wurde exakt folgender Stand verwendet:
Commit `e70d8bf` (permalink):
`https://github.com/eybmits/parameterized-hamiltonians-id/commit/e70d8bf`

Die Experimente wurden aus dem Repository-Root ausgeführt.

# B.2 Setup der Laufumgebung

```bash
git clone https://github.com/eybmits/parameterized-hamiltonians-id.git
cd parameterized-hamiltonians-id
git checkout e70d8bf

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Validierung der Umgebung:

```bash
ruff check src tests experiments
ruff format --check src tests experiments
pytest -v
```

# B.3 Reproduction Commands

Die vorliegende Repository-Version verwendet dedizierte Experiment-Entrypoints (kein einzelner globaler Entrypoint mit `experiment-id + config`).
Für jedes Experiment wird ein Run-Command angegeben; die PDF-Abbildungen werden pro Skript in den jeweiligen Output-Ordner geschrieben.

Beispielstruktur:

```bash
python experiments/<experiment_script>.py <args> --out outputs/<experiment-id>/<run-id>
```

Beispiel (Kernexperiment):

```bash
python experiments/exp01_id_vs_fd_core_demo.py \
  --kind periodic --n 12 --outer 30 --inner 20 --fmt pdf \
  --out outputs/exp01_id_vs_fd_core_demo/run01
```

Beispiel (separate Plot-Variante):

```bash
python experiments/exp01_id_vs_fd_core_demo_refined_plots.py \
  --kind periodic --n 12 --outer 30 --inner 20 --fmt pdf \
  --out outputs/exp01_id_vs_fd_core_demo_refined_plots/run01
```

# B.4 Output-Konvention

Alle Artefakte werden unter `outputs/<experiment-id>/<run-id>/` abgelegt, typischerweise als:

- `*.pdf` (Abbildungen)
- `*.csv` (Tabellen/Rohmetriken)
- `*summary*.txt` (kompakte Ergebniszusammenfassungen)

# B.5 Verweis in der Arbeit (Formulierungsvorschlag)

"The source code and scripts used for all experiments are available at
`https://github.com/eybmits/parameterized-hamiltonians-id`.
All reported results were generated from commit `e70d8bf`, ensuring a fixed and reproducible code state."
