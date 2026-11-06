# Projekt-Briefing: Tm-Vorhersage und Organismus-Leakage

Dieses Dokument ist die einzige Quelle der Wahrheit für das Projekt. Wer eine
Aufgabe übernimmt, liest es vorher ganz. Abweichungen vom hier Festgelegten
werden vorher abgestimmt, nicht nachträglich erklärt.

Laufzeit: 2026-11-09 bis 2026-11-27. Vier Bearbeiter (A, B, C, D).

---

## 1. Forschungsfrage

Wie viel der Vorhersageleistung für Protein-Schmelztemperaturen (Tm) beruht auf
echter Sequenz-Biophysik, und wie viel auf der Herkunft des Organismus?

**Kontext.** Der Meltome Atlas enthält die Proteome von 13 Organismen. Thermophile
Organismen haben durchgängig stabilere Proteome. Ein Modell, das die Abstammungs-
linie einer Sequenz erkennt, sagt Tm respektabel vorher, ohne etwas über Stabilität
zu wissen. Das ist Leakage auf Organismus-Ebene und liegt über der üblichen
Homologie-Leakage.

### Hypothesen

| ID | Hypothese | Wie geprüft |
| --- | --- | --- |
| H1 | Zufallssplit überschätzt die Generalisierung deutlich | Bedingung 1, Abfall über die Identitätsschwellen |
| H2 | Ein großer Teil der Leistung ist durch Organismus-Identität allein erklärbar | Organismus-Median-Baseline vs. Modelle; Bedingung 2 |
| H3 | Innerhalb eines Proteoms bleibt reduziertes, aber messbares Signal | Bedingung 3 |
| H4 | Ein von Grund auf trainiertes Modell nutzt die Herkunfts-Abkürzung stärker als ein vortrainiertes | CNN vs. ESM-2 über Bedingungen 0, 2, 3 |

H3 und H4 sind offen. Wir kennen die Antwort nicht. Das ist der Beitrag.

**Ein negatives Ergebnis ist ein Ergebnis.** Falls kein Modell die
Organismus-Baseline schlägt, ist das die Aussage des Projekts, nicht sein
Scheitern.

---

## 2. Experimentdesign

Vier Evaluationsbedingungen, geschachtelt von permissiv nach streng. Dieselben
Daten, dieselbe Pipeline, dieselben Metriken — nur der Split ändert sich.

| # | Bedingung | Entfernt | Folds / Wiederholungen |
| --- | --- | --- | --- |
| 0 | Zufallssplit | nichts | 5 Seeds |
| 1 | Identitäts-Cluster 90 / 70 / 50 / 30 % | nahe Homologe | 5 Cluster-Zuweisungen pro Schwelle |
| 2 | Leave-one-organism-out | Organismus-Signal | 13 Folds |
| 3 | Innerhalb eines Organismus | Organismus-Signal konstant | pro geeignetem Organismus 5-fach CV |

### Modelle

| Kürzel | Modell | Wissensquelle | Läuft auf Bedingungen |
| --- | --- | --- | --- |
| `comp` | 20 AA-Frequenzen + Länge, Ridge | nur Statistik | 0, 1, 2, 3 |
| `esm` | ESM-2 (8M oder 35M), mean-pooled, eingefroren + Ridge | Vortraining | 0, 1, 2, 3 |
| `cnn` | 1D-CNN auf One-hot, from scratch | nur Trainingsset | 0, 2, 3 |

`cnn` läuft bewusst **nicht** auf Bedingung 1. Die volle Identitätskurve bleibt
`esm` vorbehalten, sonst sprengt es Woche 2.

### Baselines

| Kürzel | Vorhersage | Zweck |
| --- | --- | --- |
| `mean` | Trainingsmittelwert | Untergrenze |
| `organism` | Median-Tm des Herkunftsorganismus, Sequenz ignoriert | prüft H2 |

`organism` ist auf Bedingung 2 nicht definiert und auf Bedingung 3 konstant.
Das ist kein Fehler, sondern der Punkt. Die Ergebnistabelle trägt dort `null`.

### Metriken

Hauptmaß **RMSE in °C**. Zweitmaß **Spearman**.

**Pflicht:** Jede Ergebniszeile trägt zusätzlich `test_std` — die Standard-
abweichung der Tm-Werte im jeweiligen Testset. Ohne diese Spalte ist RMSE
zwischen Bedingungen nicht vergleichbar: Innerhalb eines Proteoms ist die
Tm-Spannweite kleiner, der RMSE sinkt also automatisch, auch wenn das Modell
nichts kann.

---

## 3. Schnittstellen

Diese drei Formate stehen am Ende von Tag 2 fest und ändern sich danach nicht
mehr ohne Absprache mit allen.

### 3.1 Datensatz

Eine Parquet-Datei `data/meltome.parquet`:

| Spalte | Typ | Beschreibung |
| --- | --- | --- |
| `id` | str | UniProt-Accession, eindeutig |
| `sequence` | str | Aminosäuresequenz, Großbuchstaben |
| `tm` | float | Schmelztemperatur in °C |
| `organism` | str | Herkunftsorganismus, normalisiert |
| `condition` | str | `lysate` oder `cell` |

### 3.2 Splits

Eine JSON-Datei pro Split unter `splits/`, Namensschema
`{bedingung}_{parameter}_{wiederholung}.json`, z. B. `cond1_id30_rep3.json`:

```json
{
  "condition": 1,
  "params": {"identity": 0.30},
  "rep": 3,
  "train": [0, 4, 7, 11],
  "test": [1, 2, 9]
}
```

Die Indizes beziehen sich auf Zeilennummern in `data/meltome.parquet`. Die Datei
wird nicht neu generiert, sondern eingecheckt.

### 3.3 Evaluation

```python
def evaluate(train_idx, test_idx, model_name) -> dict:
    """Gibt zurück: {rmse, spearman, mae, n_train, n_test, test_std}"""
```

Ergebnisse landen als eine Zeile pro (Split × Modell) in `results/results.csv`.

---

## 4. Harte Regeln

Diese Regeln existieren, weil ihre Verletzung das Projekt still kaputt macht.

1. **Splits gehören A.** Niemand sonst erzeugt Splits. Wer ein Modell baut, lädt
   Split-Dateien, berechnet sie nicht. Selbst erzeugte Splits führen zu Zahlen,
   die nicht vergleichbar sind, und man merkt es nicht.
2. **Keine Hyperparameter-Wahl auf dem Testset.** Jede Suche läuft in einer
   inneren CV innerhalb des Trainsets.
3. **Kein Finetuning von ESM-2.** Es beantwortet die Forschungsfrage nicht und
   frisst die Restzeit. Vorschläge in diese Richtung werden abgelehnt.
4. **Embeddings werden einmal berechnet und gecacht.** Nicht pro Split neu.
5. **Nach Gate 2 (2026-11-20) kommt nichts Neues dazu.** Woche 3 ist Analyse und
   Schreiben.
6. **Jedes Ergebnis ist reproduzierbar**: fester Seed, Seed in der Ergebniszeile.

---

## 5. Rollen

| Rolle | Verantwortung | Liefert |
| --- | --- | --- |
| A | Daten, Organismus-Labels, MMseqs2-Clustering, alle Splits | `data/meltome.parquet`, `splits/*.json` |
| B | ESM-2-Embeddings, Ridge, CNN-Training | `cache/embeddings.npy`, `train.py` |
| C | Metriken, Baselines, Evaluationsschleife | `evaluate.py`, `results/results.csv` |
| D | Abbildungen, Rauschabschätzung, Interpretation, Paper | `figures/`, `paper/` |

D arbeitet in Woche 1 und 2 bei A und C mit und übernimmt ab Gate 2 eigenständig.

---

## 6. Zeitplan

| Zeitraum | A | B | C | D |
| --- | --- | --- | --- | --- |
| 09.11.–13.11. | Daten, Labels prüfen | Embeddings, Ridge | Harness, Baselines | unterstützt A und C |
| 16.11.–20.11. | MMseqs2, alle Splits | Läufe über alle Bedingungen | Rauschen, Ablation | unterstützt C |
| 23.11.–27.11. | Reproduktion prüfen | Reproduktion prüfen | Reproduktion prüfen | Figuren, Text |

### Gates

**Gate 1 — Freitag 13.11.** Es existiert eine Zahl auf Bedingung 0 für alle drei
Modelle plus beide Baselines. Falls nicht: Umfang halbieren auf Bedingung 0 und
2, Identitätskurve streichen.

**Gate 2 — Freitag 20.11.** Alle Bedingungen sind durchgelaufen. Ab hier
Feature-Freeze.

**Abgabe — Freitag 27.11.**

---

## 7. Risiken

| Risiko | Sichtbar ab | Reaktion |
| --- | --- | --- |
| Organismus-Labels fehlen im FLIP-Dump | Tag 1 | UniProt-Accessions mappen, halber Tag |
| Zu wenige Proteine pro Organismus für Bedingung 3 | Tag 2 | auf die 2–3 größten Proteome beschränken, explizit berichten |
| `lysate` und `cell` gemischt | Tag 2 | auf eine Bedingung beschränken oder als Kovariate führen |
| MMseqs2 nicht lauffähig | Tag 6 | CD-HIT als Ersatz |
| Embeddings zu langsam auf CPU | Tag 3 | 35M → 8M, notfalls Teilmenge von 5000 Sequenzen |
| CNN-Training zu langsam | Tag 8 | Kernel und Kanäle verkleinern; CNN notfalls nur auf Bedingung 0 und 3 |

### Messrauschen

Meltome-Daten stammen aus Massenspektrometrie und sind verrauscht. Ein Teil der
nicht erklärten Varianz ist Messfehler, keine Modellschwäche. Das trifft
Bedingung 3 am härtesten, weil dort die feinste Diskriminierung verlangt wird.

Falls Replikate vorhanden sind: daraus eine Obergrenze schätzen. Falls nicht: als
Limitation benennen, nicht kaschieren. „Wir konnten es nicht vorhersagen" und
„es ist auf dieser Präzision nicht messbar" sind verschiedene Aussagen.

---

## 8. Was dieses Projekt nicht ist

- Kein Versuch, State of the Art auf einem Benchmark zu schlagen.
- Keine neue Architektur.
- Kein Protein-Engineering, keine Mutationsvorhersage.
- Kein Scraping. Alle Daten kommen aus einem Download.

Wer eine Aufgabe übernimmt und merkt, dass sie in eine dieser Richtungen läuft,
hält an und fragt nach.

---

## 9. Offene Punkte

- [ ] ESM-2-Variante: 8M oder 35M? Entscheidung bis Tag 2.
- [ ] Sind Replikate im Meltome-Dump enthalten? Entscheidet, ob eine
      Rauschobergrenze möglich ist.
- [ ] Welche Organismen haben genug Proteine für Bedingung 3?
- [ ] Optional für Woche 3: externe Validierung auf ProThermDB (andere
      Messmethode). Nur falls Zeit nach Gate 2 übrig ist.
