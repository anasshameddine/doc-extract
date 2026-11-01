"""Genere colab/DONUT_COLAB.ipynb a partir de colab/donut_colab.py (une seule source de verite).

    python colab/build_notebook.py
Le script Python est copie tel quel dans une cellule `%%writefile` : le notebook ne peut pas diverger de ce qui est teste.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = (HERE / "donut_colab.py").read_text(encoding="utf-8")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


CELLS = [
    md("""
# DOC-EXTRACT · Étape 4b · Donut (Document AI)

Ce notebook fait trois choses, **dans l'ordre** :
1. entraîne Donut sur SROIE (`sroie_train`, 526 reçus) et suit la perte sur `sroie_dev` ;
2. prédit sur `sroie_dev` avec ton modèle, et sur `cord_dev` avec le checkpoint public des auteurs de Donut ;
3. sauvegarde tout sur ton Google Drive.

**Avant de commencer :** menu *Exécution → Modifier le type d'exécution → GPU T4*. Garde cet onglet ouvert et le PC branché.
Lance les cellules une par une, de haut en bas. La cellule 5 est un **test rapide** : lance-la avant l'entraînement complet.
"""),
    md("## 1. Vérifier le GPU"),
    code("!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    md("## 2. Brancher Google Drive et décompresser les données\nLe zip doit être dans `Mon Drive/DOC-EXTRACT/donut_data.zip`."),
    code("""
from google.colab import drive
drive.mount('/content/drive')

DRIVE = '/content/drive/MyDrive/DOC-EXTRACT'
!mkdir -p /content/donut_data
!unzip -q -o "{DRIVE}/donut_data.zip" -d /content/donut_data
!ls /content/donut_data
!wc -l /content/donut_data/*/metadata.jsonl
"""),
    md("## 3. Installer les bibliothèques (version fixée de `transformers`)"),
    code("!pip -q install \"transformers==4.46.3\" sentencepiece"),
    md("## 4. Écrire le code d'entraînement (fichier `donut_colab.py`, copié tel quel depuis le projet)"),
    code("%%writefile donut_colab.py\n" + SCRIPT),
    md("## 4 bis. Charger la liste des reçus (à relancer aussi après une coupure de session)"),
    code("""
import importlib, donut_colab as dc
importlib.reload(dc)
DATA = '/content/donut_data'
train_rows = dc.read_set(f'{DATA}/sroie_train')
dev_rows = dc.read_set(f'{DATA}/sroie_dev')
print(len(train_rows), 'reçus train |', len(dev_rows), 'reçus dev')
"""),
    md("""
## 5. TEST RAPIDE (2 à 4 minutes) : lance-le avant l'entraînement complet
Il entraîne sur 16 reçus pendant 1 époque pour vérifier qu'il n'y a pas d'erreur de mémoire.
Regarde **`s/recu`** et **`peak_gpu_gb`** : si `peak_gpu_gb` dépasse ~14, dis-le-moi avant de continuer.
Si tu as une erreur *CUDA out of memory*, passe `checkpointing=True` (déjà le cas) et dis-le-moi.
"""),
    code("""
import torch

model, processor = dc.load_base()
info = dc.train(model, processor, train_rows, dev_rows, '/content/smoke', epochs=1, limit=16)
per_receipt = info['history'][0]['seconds'] / 16
print('peak_gpu_gb =', info['peak_gpu_gb'], '| s/reçu =', round(per_receipt, 2))
print('Estimation de l entraînement complet (15 époques) :', round(per_receipt * len(train_rows) * 15 / 60), 'min')
del model; torch.cuda.empty_cache()
"""),
    md("""
## 6. ENTRAÎNEMENT COMPLET (environ 1 h 30 à 2 h)
- 15 époques sur les 526 reçus. Sauvegarde dans ton Drive : `best/` quand la perte dev baisse, `last/` toutes les 5 époques.
- Si la session se coupe : relance les cellules 2, 3, 4 et 4 bis, puis **la cellule 7 (reprise)** au lieu de celle-ci.
- Tu peux suivre la perte : elle doit baisser d'une époque à l'autre.
"""),
    code("""
import torch

OUT = f'{DRIVE}/donut_sroie'
model, processor = dc.load_base()
info = dc.train(model, processor, train_rows, dev_rows, OUT, epochs=15)
print('terminé en', info['train_minutes'], 'min | meilleure perte dev =', round(info['best_dev_loss'], 4))
"""),
    md("## 7. (Seulement si la session s'est coupée) Reprendre depuis la dernière époque sauvegardée\nIl repart des poids `last/` ; l'optimiseur redémarre à zéro, ce qui est acceptable."),
    code("""
# import torch
# OUT = f'{DRIVE}/donut_sroie'
# model, processor = dc.load_base(source=f'{OUT}/last')
# info = dc.train(model, processor, train_rows, dev_rows, OUT, epochs=8)   # nombre d'époques restantes
"""),
    md("""
## 8. Prédictions sur le jeu **dev**
- SROIE : ton modèle (le `best/` sauvegardé).
- CORD : le checkpoint public `donut-base-finetuned-cord-v2` (entraîné par ses auteurs sur le train CORD ; le dev est son split *validation*, jamais vu à l'entraînement).
"""),
    code("""
import json, torch, donut_colab as dc

PRED = f'{DRIVE}/donut_preds'
gpu = torch.cuda.get_device_name(0)

# --- SROIE : ton modèle fine-tuné
model, processor = dc.load_base(source=f'{DRIVE}/donut_sroie/best')
res = dc.predict(model, processor, dc.read_set(f'{DATA}/sroie_dev'))
dc.save_predictions(res, f'{PRED}/predictions_sroie_dev.json', meta={'model': 'donut-base fine-tuné SROIE', 'gpu': gpu, 'set': 'dev'})
print('SROIE dev :', len(res), 'reçus | médiane', sorted(v['ms'] for v in res.values())[len(res)//2], 'ms')
print(json.dumps(list(res.items())[0], ensure_ascii=False)[:400])
del model; torch.cuda.empty_cache()

# --- CORD : checkpoint public
from transformers import DonutProcessor, VisionEncoderDecoderModel
cord_proc = DonutProcessor.from_pretrained(dc.CORD_MODEL)
cord_model = VisionEncoderDecoderModel.from_pretrained(dc.CORD_MODEL)
res = dc.predict(cord_model, cord_proc, dc.read_set(f'{DATA}/cord_dev'), task_prompt='<s_cord-v2>', max_length=768)
dc.save_predictions(res, f'{PRED}/predictions_cord_dev.json', meta={'model': dc.CORD_MODEL, 'gpu': gpu, 'set': 'dev'})
print('CORD dev :', len(res), 'reçus | médiane', sorted(v['ms'] for v in res.values())[len(res)//2], 'ms')
print(json.dumps(list(res.items())[0], ensure_ascii=False)[:600])
"""),
    md("""
## 9. Récupérer les résultats
Les fichiers sont dans `Mon Drive/DOC-EXTRACT/donut_preds/` : `predictions_sroie_dev.json` et `predictions_cord_dev.json`.
Télécharge-les sur ton PC (clic droit → Télécharger), et envoie-moi aussi `donut_sroie/train_info.json`.

---
### À NE LANCER QU'À L'ÉTAPE 5 (score final, jeu eval)
Ne lance **pas** la cellule suivante maintenant : le jeu eval ne sert qu'une seule fois, avec des réglages figés.
"""),
    code("""
# ÉTAPE 5 UNIQUEMENT : décommente pour produire les prédictions du jeu eval.
# model, processor = dc.load_base(source=f'{DRIVE}/donut_sroie/best')
# res = dc.predict(model, processor, dc.read_set(f'{DATA}/sroie_eval'))
# dc.save_predictions(res, f'{PRED}/predictions_sroie_eval.json', meta={'model': 'donut-base fine-tuné SROIE', 'gpu': gpu, 'set': 'eval'})
# res = dc.predict(cord_model, cord_proc, dc.read_set(f'{DATA}/cord_eval'), task_prompt='<s_cord-v2>', max_length=768)
# dc.save_predictions(res, f'{PRED}/predictions_cord_eval.json', meta={'model': dc.CORD_MODEL, 'gpu': gpu, 'set': 'eval'})
"""),
]


def build() -> dict:
    return {"cells": CELLS, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}}}


if __name__ == "__main__":
    out = HERE / "DONUT_COLAB.ipynb"
    out.write_text(json.dumps(build(), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"ecrit : {out}")
