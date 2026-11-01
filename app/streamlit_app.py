"""Demo : depose un recu, choisis la methode, compare. Lancement :  streamlit run app/streamlit_app.py"""
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from docextract import demo

HERE = Path(__file__).parent
ALL = "all"
CHOICES = [ALL, *demo.METHODS]
CHOICE_LABELS = {ALL: "Les trois côte à côte", **demo.METHODS}
DATASET_LABELS = {"sroie": "Malaisien (SROIE)", "cord": "Indonésien (CORD)"}

st.set_page_config(page_title="Lire un reçu", page_icon="🧾", layout="wide")
st.markdown(f"<style>{(HERE / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

tab_try, tab_bench = st.tabs(["Essayer", "Résultats mesurés"])

with tab_try:
    st.title("Lire un reçu")
    st.markdown('<p class="lede">Déposez la photo d\'un ticket de caisse. Trois méthodes en sortent le fournisseur, '
                "la date, le total et la taxe, et chacune imprime son propre ticket pour que vous voyiez laquelle "
                "s'en sort le mieux.</p>", unsafe_allow_html=True)

    left, right = st.columns([5, 6], gap="large")
    with left:
        upload = st.file_uploader("Image du reçu", type=["jpg", "jpeg", "png", "webp", "bmp"],
                                  label_visibility="collapsed")
        image = None
        if upload is not None:
            try:
                image = Image.open(upload)
                image.load()
            except (UnidentifiedImageError, OSError):
                st.error("Cette image est illisible. Envoyez un JPEG ou un PNG.")
            else:
                st.markdown(demo.photo_html(image, upload.name), unsafe_allow_html=True)

    with right:
        dataset = st.radio("Type de reçu", list(DATASET_LABELS), format_func=DATASET_LABELS.get, horizontal=True,
                           help="Donut a un modèle par corpus : SROIE lit fournisseur, date et total ; CORD lit total et taxe.")
        choice = st.radio("Méthode", CHOICES, format_func=CHOICE_LABELS.get)
        llm_model = st.selectbox("Modèle du LLM", ["qwen2.5:3b", "qwen2.5:7b"],
                                 help="Ollama doit être lancé. Le premier appel de chaque méthode charge son modèle en mémoire.")
        go = st.button("Lire le reçu", type="primary", disabled=image is None)
        if image is None:
            st.markdown('<p class="hint">Déposez une image (JPEG ou PNG) pour pouvoir lancer la lecture.</p>',
                        unsafe_allow_html=True)

    key = (getattr(upload, "name", None), getattr(upload, "size", None))      # les resultats suivent l'image
    if go and image is not None:
        methods = list(demo.METHODS) if choice == ALL else [choice]
        doc_id = upload.name.rsplit(".", 1)[0]
        with st.spinner("Impression des tickets…"):
            st.session_state["run"] = {"key": key, "fresh": True,
                                       "outcomes": demo.run_methods(methods, image, dataset, llm_model, doc_id)}
    run = st.session_state.get("run")
    if run and image is not None and run["key"] == key:
        tickets = "".join(demo.ticket_html(o, fresh=run["fresh"]) for o in run["outcomes"])
        run["fresh"] = False                                   # l'animation ne rejoue pas a chaque clic ailleurs
        st.markdown(f'<div class="tickets">{tickets}</div>', unsafe_allow_html=True)
        with st.expander("JSON brut"):
            st.json({o.method: o.result.model_dump(mode="json") for o in run["outcomes"] if o.result})

with tab_bench:
    text = demo.benchmark_markdown("eval")
    if text:
        st.markdown(text)
    else:
        st.info("Le tableau n'existe pas encore : python scripts/10_benchmark.py --set eval --final")
