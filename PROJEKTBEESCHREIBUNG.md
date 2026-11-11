# Projekt-Briefing: Tm-Vorhersage mit Zwei-Stufen-Modell und Ensemble

Dieses Dokument ist die einzige Quelle der Wahrheit für das Projekt. Wer eine
Aufgabe übernimmt, liest es vorher ganz. Abweichungen vom hier Festgelegten
werden vorher abgestimmt, nicht nachträglich erklärt.

Laufzeit: 2026-11-09 bis 2026-11-27. Vier Bearbeiter (A, B, C, D).

> **Änderung am 2026-11-11.** Das Projekt hat keine Forschungsfrage mehr. Die
> Hypothesen H1–H4, die vier Split-Bedingungen und die Leakage-Analyse sind
> gestrichen. Ziel ist jetzt ein fertiges Vorhersage-Tool. Daten und
> ESM-Embeddings aus den ersten Tagen bleiben unverändert gültig.

---

## 1. Ziel

Ein Lernprojekt mit einem fertigen, nutzbaren Modell am Ende:

**Input:** Aminosäuresequenz eines Proteins.
**Output:** vorhergesagte Schmelztemperatur Tm in °C mit Fehlerangabe, dazu der
vermutete Herkunftsorganismus.

Es ist egal, ob es so etwas schon gibt. Es geht darum, Modelle selbst zu bauen,
zu vergleichen und zu einem Werkzeug zusammenzuführen. Ein Ensemble, das nur
wenig besser ist als das beste Einzelmodell, ist ein normales Ergebnis und kein
Scheitern.

---

## 2. Stand

| Teil | Stand | Datei |
| --- | --- | --- |
| Datensatz | fertig | `data/meltome.parquet`, gebaut von `scripts/build_meltome.py` |
| ESM-2-Embeddings (35M) | fertig | `cache/embeddings.npy`, gebaut von `scripts/embed_esm.py` |
| Split und Folds | offen | `splits/` |
| Stufe 1, Tm-Modelle, Ensemble | offen | |
| `predict.py`, App | offen | |

### Daten in Zahlen

- 33.904 Proteine aus 13 Organismen (Meltome Atlas über den FLIP-Dump).
- Median-Tm pro Organismus reicht von 41 °C (*C. elegans*) bis 85 °C
  (*Thermus thermophilus*).
- 24.113 Zeilen `lysate`, 9.791 Zeilen `cell`. **Alle `cell`-Zeilen sind Mensch,
  und Mensch kommt nur als `cell` vor.** Messart und Organismus sind dadurch
  nicht trennbar (siehe Abschnitt 10).
- Die kleinste Klasse ist *Danio rerio* mit 253 Proteinen, die größte Mensch
  mit 9.791.

---

## 3. Architektur

```
Sequenz ──► Stufe 1: Organismus-Klassifikator (ESM ──► 13 Wahrscheinlichkeiten)
   │                         │
   ├──► comp + Organismus ──► Ridge ────────► Tm₁ ─┐
   ├──► ESM  + Organismus ──► Ridge / MLP ──► Tm₂ ─┼──► Ensemble ──► Tm ± Fehler
   └──► One-hot ──► CNN (optional) ─────────► Tm₃ ─┘
```

### 3.1 Stufe 1: Organismus-Klassifikator

| | |
| --- | --- |
| Input | ESM-Embedding, 480 Zahlen |
| Output | eine Wahrscheinlichkeit pro Organismus |
| Modell | logistische Regression, optional kleines MLP |
| Pflicht | Klassengewichte, weil die Klassen sehr ungleich groß sind |

### 3.2 Stufe 2: Tm-Modelle

| Kürzel | Features | Modell | Status |
| --- | --- | --- | --- |
| `comp` | 20 AA-Frequenzen + Länge + Organismus-Wahrscheinlichkeiten | Ridge | Pflicht |
| `esm` | ESM-Embedding + Organismus-Wahrscheinlichkeiten | Ridge, optional MLP | Pflicht |
| `cnn` | One-hot der Sequenz, sonst nichts | 1D-CNN from scratch | optional, Entscheidung bei Gate 1 |

Das CNN bleibt klein (2–3 Faltungsschichten, 32–64 Kanäle, Sequenzen auf 1000
Reste gekürzt), sonst sprengt es die Rechenzeit auf dem Laptop.

### 3.3 Ensemble

Ein Ridge-Modell lernt, wie stark jedes Tm-Modell zählt, z. B.
`Tm = 0,6 · Tm₂ + 0,3 · Tm₃ + 0,1 · Tm₁ + b`. Die Gewichte werden als Plot
berichtet. Fällt das CNN weg, besteht das Ensemble aus `comp` und `esm`.

### 3.4 Baselines

| Kürzel | Vorhersage |
| --- | --- |
| `mean` | Mittelwert der Trainings-Tm |
| `organism` | Median-Tm des wahren Herkunftsorganismus, Sequenz ignoriert |

Jedes Modell muss sich an beiden messen lassen.

### 3.5 Out-of-fold-Vorhersagen

Stufe 2 und das Ensemble lernen auf Vorhersagen anderer Modelle. Diese
Vorhersagen müssen **out-of-fold** sein: Für die Trainingszeilen in Fold k kommt
die Vorhersage von einem Modell, das ohne Fold k trainiert wurde. Sonst ist
Stufe 1 auf den Trainingsdaten übertrieben sicher, Stufe 2 verlässt sich zu
stark darauf und versagt im Test. Die Folds kommen aus der Split-Datei
(Abschnitt 5.2).

---

## 4. Evaluation

### Test-Split

Ein fester Test-Split: Sequenzen werden mit MMseqs2 bei einer
Identitätsschwelle geclustert (Vorschlag 30 %), 80 % der Cluster gehen ins
Training, 20 % in den Test. Kein Testprotein ist damit einem Trainingsprotein
zu ähnlich, und die Fehlerangabe ist ehrlich. Innerhalb des Trainings gibt es
fünf Folds, ebenfalls nach Clustern aufgeteilt.

### Metriken

| Was | Metriken |
| --- | --- |
| Tm-Modelle und Ensemble | RMSE in °C (Hauptmaß), MAE, Spearman |
| Stufe 1 | Accuracy, Balanced Accuracy, Confusion Matrix |

Die Fehlerangabe in `predict.py` ist der Test-RMSE des finalen Ensembles.

### Vergleiche im Bericht

1. Jedes Einzelmodell gegen das Ensemble.
2. Mit und ohne Organismus-Feature: Bringt Stufe 1 etwas?
3. Alle Modelle gegen die Baselines `mean` und `organism`.
4. Confusion Matrix: Welche Organismen werden verwechselt?
5. Ensemble-Gewichte: Welches Modell trägt wie viel bei?

---

## 5. Schnittstellen

Diese Formate stehen am Ende von Gate 1 fest und ändern sich danach nicht mehr
ohne Absprache mit allen.

### 5.1 Datensatz

Eine Parquet-Datei `data/meltome.parquet`:

| Spalte | Typ | Beschreibung |
| --- | --- | --- |
| `id` | str | UniProt-Accession, eindeutig |
| `sequence` | str | Aminosäuresequenz, Großbuchstaben |
| `tm` | float | Schmelztemperatur in °C |
| `organism` | str | Herkunftsorganismus, normalisiert |
| `condition` | str | `lysate` oder `cell` |

Zeilennummern sind der Indexraum aller anderen Dateien. `content_sha256` in
`data/meltome.manifest.json` muss auf jedem Rechner übereinstimmen.

### 5.2 Split

Eine Datei `splits/main.json`:

```json
{
  "identity": 0.30,
  "seed": 42,
  "train": [0, 4, 7, 11],
  "train_fold": [2, 0, 4, 1],
  "test": [1, 2, 9]
}
```

`train_fold[i]` ist der Fold (0–4) der Zeile `train[i]`. Für das finale
Training auf allen Daten liefert A zusätzlich `splits/final.json` mit `train`
über alle Zeilen, `train_fold` und leerem `test`.

### 5.3 Vorhersagen

Jedes Modell schreibt `predictions/{kürzel}.parquet`:

| Spalte | Typ | Beschreibung |
| --- | --- | --- |
| `row` | int | Zeilennummer in `data/meltome.parquet` |
| `part` | str | `oof` (Training, out-of-fold) oder `test` |
| `tm_pred` | float | vorhergesagte Tm in °C |

Stufe 1 schreibt `predictions/organism.parquet` mit `row`, `part` und einer
Spalte `p_{organismus}` pro Klasse.

### 5.4 Ergebnisse

```python
def evaluate(y_true, y_pred) -> dict:
    """Gibt zurück: {rmse, mae, spearman, n}"""
```

Tm-Ergebnisse landen als eine Zeile pro Modell in `results/results.csv`
(`model, rmse, mae, spearman, n_train, n_test, seed`), Stufe 1 in
`results/classifier.csv`.

### 5.5 Vorhersage-Tool

```
python predict.py --sequence MKTAYIAKQR...
python predict.py --fasta proteins.fasta
```

Ausgabe pro Sequenz: `tm`, `tm_error`, `organism`, `organism_prob`. Die App
ruft dieselbe Funktion auf und rechnet nichts selbst.

---

## 6. Harte Regeln

Diese Regeln existieren, weil ihre Verletzung das Projekt still kaputt macht.

1. **Split und Folds gehören A.** Niemand sonst erzeugt sie. Wer ein Modell
   baut, lädt `splits/*.json`.
2. **Das Testset wird nur zur Endauswertung angefasst.** Jede
   Hyperparameter-Suche läuft über die Folds im Training.
3. **Stufe 2 und Ensemble lernen nur auf Out-of-fold-Vorhersagen.**
4. **Kein Finetuning von ESM-2.** Es frisst die Restzeit auf CPU.
5. **Embeddings werden einmal berechnet und gecacht.** Nicht pro Modell neu.
6. **Nach Gate 2 (2026-11-20) kommt nichts Neues dazu.** Woche 3 ist App,
   Plots und Bericht.
7. **Jedes Ergebnis ist reproduzierbar**: fester Seed, Seed in der
   Ergebniszeile.

---

## 7. Rollen

| Rolle | Verantwortung | Liefert |
| --- | --- | --- |
| A | Daten, `lysate`/`cell`-Entscheidung, MMseqs2-Clustering, Split und Folds | `data/meltome.parquet`, `splits/*.json` |
| B | Stufe 1, CNN | `predictions/organism.parquet`, `predictions/cnn.parquet`, `results/classifier.csv` |
| C | `comp`- und `esm`-Modelle, Ensemble, Evaluation, `predict.py` | `predictions/*.parquet`, `results/results.csv`, `predict.py` |
| D | App, Plots, Bericht | `app.py`, `figures/`, `paper/` |

D unterstützt in Woche 1 und 2 A und C und übernimmt ab Gate 2 eigenständig.

---

## 8. Zeitplan

| Zeitraum | A | B | C | D |
| --- | --- | --- | --- | --- |
| 09.11.–13.11. | Daten (fertig), Split und Folds | Stufe 1 | `comp`, `esm`, Baselines | unterstützt A und C |
| 16.11.–20.11. | finale Folds, Reproduktion | CNN | Ensemble, `predict.py` | App-Gerüst, erste Plots |
| 23.11.–27.11. | Reproduktion prüfen | Reproduktion prüfen | Reproduktion prüfen | App, Plots, Bericht |

### Gates

**Gate 1 — Freitag 13.11.** Split steht. Stufe 1, `comp` und `esm` haben
Test-Zahlen, ebenso beide Baselines. Entscheidung, ob das CNN mitkommt.

**Gate 2 — Freitag 20.11.** Ensemble und `predict.py` laufen auf dem finalen
Modell. Ab hier Feature-Freeze.

**Abgabe — Freitag 27.11.**

---

## 9. Rechenaufwand

Ein normaler Laptop reicht. Referenz: Die ESM-2-35M-Embeddings haben auf CPU mit
12 Threads 3,6 Stunden gebraucht und sind fertig.

| Schritt | Aufwand |
| --- | --- |
| `comp`, `esm`, Stufe 1, Ensemble, Baselines | Sekunden bis Minuten |
| MMseqs2-Clustering | 10–30 Minuten |
| CNN, 32–64 Kanäle | etwa 20 Minuten bis 1,5 Stunden pro Training auf allen Daten |

Wird das CNN zu langsam: Kanäle verkleinern, über Nacht laufen lassen,
Google Colab mit GPU nutzen oder weglassen.

---

## 10. Risiken

| Risiko | Reaktion |
| --- | --- |
| Mensch ist gleichzeitig die einzige `cell`-Messung | Nur `lysate` nehmen (12 Organismen) oder akzeptieren und im Bericht nennen |
| Stufe 1 sagt zu oft die großen Klassen voraus | Klassengewichte, Balanced Accuracy berichten |
| MMseqs2 läuft nicht unter Windows | WSL oder Windows-Build, notfalls CD-HIT |
| CNN zu langsam | siehe Abschnitt 9 |
| Ensemble bringt kaum etwas | ist ein Ergebnis, im Bericht nennen |

### Messrauschen

Meltome-Daten stammen aus Massenspektrometrie und sind verrauscht. Ein Teil des
Fehlers ist Messfehler, keine Modellschwäche. Im Bericht als Grenze nennen.

---

## 11. Was dieses Projekt nicht ist

- Keine Forschungsfrage, kein Beitrag zur Forschung.
- Kein Versuch, State of the Art zu schlagen.
- Keine neue Architektur.
- Kein Protein-Engineering, keine Mutationsvorhersage.
- Kein Scraping. Alle Daten kommen aus einem Download.

Wer eine Aufgabe übernimmt und merkt, dass sie in eine dieser Richtungen läuft,
hält an und fragt nach.

---

## 12. Offene Punkte

- [ ] Nur `lysate` (12 Organismen) oder beide Messarten? Entscheidung A, bis Gate 1.
- [ ] Identitätsschwelle für den Split, Vorschlag 30 %. Entscheidung A, bis Gate 1.
- [ ] Kommt das CNN mit? Entscheidung bei Gate 1.
- [ ] Stufe 1: logistische Regression reicht, oder MLP?
- [ ] App mit Streamlit oder Gradio?
- [ ] Optional: Stufe 1 sagt die Wachstumstemperatur oder die Klasse
      (kälteliebend / normal / hitzeliebend) vorher statt der Organismus-ID.
      Braucht eine kleine Literaturtabelle. Nur falls vor Gate 2 Zeit bleibt.
- [x] ESM-2-Variante: 35M, Embeddings sind berechnet.
