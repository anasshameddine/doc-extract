# Comparaison des 3 approches · jeu eval

## SROIE (100 reçus) · exact match par champ, en %

| Champ | 1 · OCR + règles | 2 · OCR + LLM qwen2.5:3b + vérif | 2 · OCR + LLM qwen2.5:7b + vérif | 3 · Donut (Document AI) |
|---|---|---|---|---|
| company | 45.0 | 58.0 | 54.0 | **89.0** |
| date | 90.0 | 91.0 | **93.0** | 89.0 |
| total | 90.0 | **94.0** | 92.0 | 77.0 |
| *company, sans espaces (diagnostic)* | 80.0 | 67.0 | 64.0 | **89.0** |
| *champs corrects* | 75.0 | 81.0 | 79.7 | **85.0** |
| *reçus 100 % corrects* | 36.0 | 48.0 | 44.0 | **67.0** |

## CORD (100 reçus) · exact match par champ, en %

| Champ | 1 · OCR + règles | 2 · OCR + LLM qwen2.5:3b + vérif | 2 · OCR + LLM qwen2.5:7b + vérif | 3 · Donut (Document AI) |
|---|---|---|---|---|
| total | 84.2 | 80.0 | 89.5 | **99.0** |
| tax | 63.4 | 68.3 | 46.3 | **97.6** |
| *champs corrects* | 77.9 | 76.5 | 76.5 | **98.5** |
| *reçus 100 % corrects* | 69.0 | 69.0 | 67.0 | **92.0** |

## Ressources · temps moyen par reçu, de l'image au JSON

| Approche | Jeu | Matériel | Temps moyen / reçu | Mémoire GPU |
|---|---|---|---|---|
| 1 · OCR + règles | SROIE | CPU (rapidocr) | 2.5 s | aucune |
| 2 · OCR + LLM qwen2.5:3b + vérif | SROIE | GPU (Ollama) | 5.9 s | 2.16 Go |
| 2 · OCR + LLM qwen2.5:7b + vérif | SROIE | GPU (Ollama) | 9.9 s | 2.31 Go |
| 3 · Donut (Document AI) | SROIE | GPU (local) | 567 ms | 1.47 Go |
| 1 · OCR + règles | CORD | CPU (rapidocr) | 1.2 s | aucune |
| 2 · OCR + LLM qwen2.5:3b + vérif | CORD | GPU (Ollama) | 4.2 s | 2.16 Go |
| 2 · OCR + LLM qwen2.5:7b + vérif | CORD | GPU (Ollama) | 6.4 s | 2.31 Go |
| 3 · Donut (Document AI) | CORD | GPU (local) | 704 ms | 1.47 Go |

## À lire avec ce tableau

- Les temps des approches 1 et 2 comptent l'OCR (rapidocr, sur CPU) ; Donut n'a pas d'OCR. Tout est mesuré sur le même PC (RTX 2050, 4 Go).
- Approche 3, jeu CORD : checkpoint public entraîné par les auteurs de Donut sur le train CORD (dans le domaine). Jeu SROIE : notre propre fine-tuning (526 reçus, 75 min sur un GPU gratuit).
- « company, sans espaces » est un diagnostic ajouté après l'évaluation : il ignore espaces et ponctuation, car l'OCR colle les mots (« KEDAIPAPANYEWCHUAN »). La mesure officielle reste l'exact match ; aucun modèle n'a été relancé ni réglé pour ce diagnostic.
- Les 3 approches sont notées avec le même code (normalisation des dates et montants, mêmes métriques) sur les mêmes reçus.
- Réglages faits sur le jeu dev uniquement ; le jeu eval n'est évalué qu'une fois, avec des réglages figés.
