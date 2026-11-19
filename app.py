import io
import re
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import py3Dmol
from Bio.PDB import PDBParser, MMCIFParser
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from streamlit.components.v1 import html as st_html

APP_TITLE = "ProteinScope"
AF_API = "https://alphafold.ebi.ac.uk/api/prediction/{}"
UNIPROT_API = "https://rest.uniprot.org/uniprotkb/{}.json"

# Curated examples. The two multi-part/precursor teaching examples include
# explicit interpretation notes so users do not mistake one predicted chain
# for the complete mature or multimeric biological assembly.
PROTEINS = {
    "Human lysozyme": {
        "accession": "P61626",
        "role": "Enzyme involved in innate antibacterial defence.",
        "note": "A compact single-chain example suitable for introductory structure exploration.",
        "group": "Core examples",
    },
    "Human myoglobin": {
        "accession": "P02144",
        "role": "Oxygen-binding protein found mainly in muscle.",
        "note": "A single-chain globular protein and a useful example of an alpha-helical fold.",
        "group": "Core examples",
    },
    "Human growth hormone": {
        "accession": "P01241",
        "role": "Hormone involved in growth and metabolism.",
        "note": "A single-chain protein; its structure is useful for discussing helices and protein function.",
        "group": "Core examples",
    },
    "Human haemoglobin subunit beta": {
        "accession": "P68871",
        "role": "One of the globin chains in adult haemoglobin.",
        "note": "This AlphaFold entry represents an individual chain, not the complete four-subunit haemoglobin assembly.",
        "group": "Additional examples",
    },
    "Human insulin precursor": {
        "accession": "P01308",
        "role": "Precursor of the hormone insulin, which regulates blood glucose.",
        "note": "The UniProt entry is a precursor sequence. Do not confuse its predicted precursor model with mature, processed two-chain insulin.",
        "group": "Additional examples",
    },
}

st.set_page_config(page_title=APP_TITLE, page_icon="🧬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
:root {
    --txt:#e6edf7; --muted:#8fa0ba; --line:rgba(255,255,255,.09);
    --cyan:#22d3ee; --violet:#8b5cf6; --pink:#f472b6;
    --grad:linear-gradient(120deg,#22d3ee 0%,#8b5cf6 55%,#f472b6 100%);
}
html, body, .stApp, button, input, textarea, select {font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;}
.stApp {
    color: var(--txt);
    background:
        radial-gradient(900px 520px at 6% -6%, rgba(34,211,238,.15), transparent 60%),
        radial-gradient(820px 520px at 96% 0%, rgba(139,92,246,.20), transparent 60%),
        radial-gradient(760px 620px at 50% 112%, rgba(244,114,182,.09), transparent 60%),
        #070b16;
    background-attachment: fixed;
}
.stApp p, .stApp li, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {color: var(--txt);}
header[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 2.5rem; padding-bottom: 2.5rem; max-width: 1400px; animation: fadeUp .6s ease both;}
@keyframes fadeUp {from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:none}}
@keyframes shimmer {to{background-position:200% center}}
@keyframes flow {from{background-position-x:0} to{background-position-x:-360px}}
@keyframes pulse {0%,100%{box-shadow:0 0 0 0 rgba(52,211,153,.6)} 50%{box-shadow:0 0 0 7px rgba(52,211,153,0)}}

/* ---------- hero ---------- */
.hero {
    position: relative; overflow: hidden; padding: 2.1rem 2.3rem 1.9rem; border-radius: 24px;
    border: 1px solid var(--line);
    background: linear-gradient(135deg, rgba(34,211,238,.10), rgba(139,92,246,.11) 55%, rgba(244,114,182,.08));
    box-shadow: 0 24px 70px -26px rgba(0,0,0,.7), inset 0 1px 0 rgba(255,255,255,.07);
}
.hero::before {
    content: ""; position: absolute; top: 0; right: 0; bottom: 0; width: 55%;
    background: url("data:image/svg+xml,%3Csvg%20xmlns=%22http://www.w3.org/2000/svg%22%20width=%22360%22%20height=%22220%22%20viewBox=%220%200%20360%20220%22%3E%3Cpolyline%20points=%220%2C110.0%206%2C123.7%2012%2C136.8%2018%2C148.8%2024%2C159.0%2030%2C167.2%2036%2C172.8%2042%2C175.6%2048%2C175.6%2054%2C172.8%2060%2C167.2%2066%2C159.0%2072%2C148.8%2078%2C136.8%2084%2C123.7%2090%2C110.0%2096%2C96.3%20102%2C83.2%20108%2C71.2%20114%2C61.0%20120%2C52.8%20126%2C47.2%20132%2C44.4%20138%2C44.4%20144%2C47.2%20150%2C52.8%20156%2C61.0%20162%2C71.2%20168%2C83.2%20174%2C96.3%20180%2C110.0%20186%2C123.7%20192%2C136.8%20198%2C148.8%20204%2C159.0%20210%2C167.2%20216%2C172.8%20222%2C175.6%20228%2C175.6%20234%2C172.8%20240%2C167.2%20246%2C159.0%20252%2C148.8%20258%2C136.8%20264%2C123.7%20270%2C110.0%20276%2C96.3%20282%2C83.2%20288%2C71.2%20294%2C61.0%20300%2C52.8%20306%2C47.2%20312%2C44.4%20318%2C44.4%20324%2C47.2%20330%2C52.8%20336%2C61.0%20342%2C71.2%20348%2C83.2%20354%2C96.3%20360%2C110.0%22%20fill=%22none%22%20stroke=%22%2322d3ee%22%20stroke-width=%223%22%20stroke-linecap=%22round%22/%3E%3Cpolyline%20points=%220%2C110.0%206%2C96.3%2012%2C83.2%2018%2C71.2%2024%2C61.0%2030%2C52.8%2036%2C47.2%2042%2C44.4%2048%2C44.4%2054%2C47.2%2060%2C52.8%2066%2C61.0%2072%2C71.2%2078%2C83.2%2084%2C96.3%2090%2C110.0%2096%2C123.7%20102%2C136.8%20108%2C148.8%20114%2C159.0%20120%2C167.2%20126%2C172.8%20132%2C175.6%20138%2C175.6%20144%2C172.8%20150%2C167.2%20156%2C159.0%20162%2C148.8%20168%2C136.8%20174%2C123.7%20180%2C110.0%20186%2C96.3%20192%2C83.2%20198%2C71.2%20204%2C61.0%20210%2C52.8%20216%2C47.2%20222%2C44.4%20228%2C44.4%20234%2C47.2%20240%2C52.8%20246%2C61.0%20252%2C71.2%20258%2C83.2%20264%2C96.3%20270%2C110.0%20276%2C123.7%20282%2C136.8%20288%2C148.8%20294%2C159.0%20300%2C167.2%20306%2C172.8%20312%2C175.6%20318%2C175.6%20324%2C172.8%20330%2C167.2%20336%2C159.0%20342%2C148.8%20348%2C136.8%20354%2C123.7%20360%2C110.0%22%20fill=%22none%22%20stroke=%22%238b5cf6%22%20stroke-width=%223%22%20stroke-linecap=%22round%22/%3E%3Cline%20x1=%2212%22%20y1=%22136.8%22%20x2=%2212%22%20y2=%2283.2%22%20stroke=%22%23f472b6%22%20stroke-opacity=%22.35%22%20stroke-width=%222%22/%3E%3Ccircle%20cx=%2212%22%20cy=%22136.8%22%20r=%223.2%22%20fill=%22%2322d3ee%22/%3E%3Ccircle%20cx=%2212%22%20cy=%2283.2%22%20r=%223.2%22%20fill=%22%238b5cf6%22/%3E%3Cline%20x1=%2224%22%20y1=%22159.0%22%20x2=%2224%22%20y2=%2261.0%22%20stroke=%22%23f472b6%22%20stroke-opacity=%22.35%22%20stroke-width=%222%22/%3E%3Ccircle%20cx=%2224%22%20cy=%22159.0%22%20r=%223.2%22%20fill=%22%2322d3ee%22/%3E%3Ccircle%20cx=%2224%22%20cy=%2261.0%22%20r=%223.2%22%20fill=%22%238b5cf6%22/%3E%3Cline%20x1=%2236%22%20y1=%22172.8%22%20x2=%2236%22%20y2=%2247.2%22%20stroke=%22%23f472b6%22%20stroke-opacity=%22.35%22%20stroke-width=%222%22/%3E%3Ccircle%20cx=%2236%22%20cy=%22172.8%22%20r=%223.2%22%20fill=%22%2322d3ee%22/%3E%3Ccircle%20cx=%2236%22%20cy=%2247.2%22%20r=%223.2%22%20fill=%22%238b5cf6%22/%3E%3Cline%20x1=%2248%22%20y1=%22175.6%22%20x2=%2248%22%20y2=%2244.4%22%20stroke=%22%23f472b6%22%20stroke-opacity=%22.35%22%20stroke-width=%222%22/%3E%3Ccircle%20cx=%2248%22%20cy=%22175.6%22%20r=%223.2%22%20fill=%22%2322d3ee%22/%3E%3Ccircle%20cx=%2248%22%20cy=%2244.4%22%20r=%223.2%22%20fill=%22%238b5cf6%22/%3E%3Cline%20x1=%2260%22%20y1=%22167.2%22%20x2=%2260%22%20y2=%2252.8%22%20stroke=%22%23f472b6%22%20stroke-opacity=%22.35%22%20stroke-width=%222%22/%3E%3Ccircle%20cx=%2260%22%20cy=%22167.2%22%20r=%223.2%22%20fill=%22%2322d3ee%22/%3E%3Ccircle%20cx=%2260%22%20cy=%2252.8%22%20r=%223.2%22%20fill=%22%238b5cf6%22/%3E%3Cline%20x1=%2272%22%20y1=%22148.8%22%20x2=%2272%22%20y2=%2271.2%22%20stroke=%22%23f472b6%22%20stroke-opacity=%22.35%22%20stroke-width=%222%22/%3E%3Ccircle%20cx=%2272%22%20cy=%22148.8%22%20r=%223.2%22%20fill=%22%2322d3ee%22/%3E%3Ccircle%20cx=%2272%22%20cy=%2271.2%22%20r=%223.2%22%20fill=%22%238b5cf6%22/%3E%3Cline%20x1=%2284%22%20y1=%22123.7%22%20x2=%2284%22%20y2=%2296.3%22%20stroke=%22%23f472b6%22%20stroke-opacity=%22.35%22%20stroke-width=%222%22/%3E%3Ccircle%20cx=%2284%22%20cy=%22123.7%22%20r=%223.2%22%20fill=%22%2322d3ee%22/%3E%3Ccircle%20cx=%2284%22%20cy=%2296.3%22%20r=%223.2%22%20fill=%22%238b5cf6%22/%3E%3Cline%20x1=%2296%22%20y1=%2296.3%22%20x2=%2296%22%20y2=%22123.7%22%20stroke=%22%23f472b6%22%20stroke-opacity=%22.35%22%20stroke-width=%222%22/%3E%3Ccircle%20cx=%2296%22%20cy=%2296.3%22%20r=%223.2%22%20fill=%22%2322d3ee%22/%3E%3Ccircle%20cx=%2296%22%20cy=%22123.7%22%20r=%223.2%22%20fill=%22%238b5cf6%22/%3E%3Cline%20x1=%22108%22%20y1=%2271.2%22%20x2=%22108%22%20y2=%22148.8%22%20stroke=%22%23f472b6%22%20stroke-opacity=%22.35%22%20stroke-width=%222%22/%3E%3Ccircle%20cx=%22108%22%20cy=%2271.2%22%20r=%223.2%22%20fill=%22%2322d3ee%22/%3E%3Ccircle%20cx=%22108%22%20cy=%22148.8%22%20r=%223.2%22%20fill=%22%238b5cf6%22/%3E%3Cline%20x1=%22120%22%20y1=%2252.8%22%20x2=%22120%22%20y2=%22167.2%22%20stroke=%22%23f472b6%22%20stroke-opacity=%22.35%22%20stroke-width=%222%22/%3E%3Ccircle%20cx=%22120%22%20cy=%2252.8%22%20r=%223.2%22%20fill=%22%2322d3ee%22/%3E%3Ccircle%20cx=%22120%22%20cy=%22167.2%22%20r=%223.2%22%20fill=%22%238b5cf6%22/%3E%3Cline%20x1=%22132%22%20y1=%2244.4%22%20x2=%22132%22%20y2=%22175.6%22%20stroke=%22%23f472b6%22%20stroke-opacity=%22.35%22%20stroke-width=%222%22/%3E%3Ccircle%20cx=%22132%22%20cy=%2244.4%22%20r=%223.2%22%20fill=%22%2322d3ee%22/%3E%3Ccircle%20cx=%22132%22%20cy=%22175.6%22%20r=%223.2%22%20fill=%22%238b5cf6%22/%3E%3Cline%20x1=%22144%22%20y1=%2247.2%22%20x2=%22144%22%20y2=%22172.8%22%20stroke=%22%23f472b6%22%20stroke-opacity=%22.35%22%20stroke-width=%222%22/%3E%3Ccircle%20cx=%22144%22%20cy=%2247.2%22%20r=%223.2%22%20fill=%22%2322d3ee%22/%3E%3Ccircle%20cx=%22144%22%20cy=%22172.8%22%20r=%223.2%22%20fill=%22%238b5cf6%22/%3E%3Cline%20x1=%22156%22%20y1=%2261.0%22%20x2=%22156%22%20y2=%22159.0%22%20stroke=%22%23f472b6%22%20stroke-opacity=%22.35%22%20stroke-width=%222%22/%3E%3Ccircle%20cx=%22156%22%20cy=%2261.0%22%20r=%223.2%22%20fill=%22%2322d3ee%22/%3E%3Ccircle%20cx=%22156%22%20cy=%22159.0%22%20r=%223.2%22%20fill=%22%238b5cf6%22/%3E%3Cline%20x1=%22168%22%20y1=%2283.2%22%20x2=%22168%22%20y2=%22136.8%22%20stroke=%22%23f472b6%22%20stroke-opacity=%22.35%22%20stroke-width=%222%22/%3E%3Ccircle%20cx=%22168%22%20cy=%2283.2%22%20r=%223.2%22%20fill=%22%2322d3ee%22/%3E%3Ccircle%20cx=%22168%22%20cy=%22136.8%22%20r=%223.2%22%20fill=%22%238b5cf6%22/%3E%3Cline%20x1=%22180%22%20y1=%22110.0%22%20x2=%22180%22%20y2=%22110.0%22%20stroke=%22%23f472b6%22%20stroke-opacity=%22.35%22%20stroke-width=%222%22/%3E%3Ccircle%20cx=%22180%22%20cy=%22110.0%22%20r=%223.2%22%20fill=%22%2322d3ee%22/%3E%3Ccircle%20cx=%22180%22%20cy=%22110.0%22%20r=%223.2%22%20fill=%22%238b5cf6%22/%3E%3Cline%20x1=%22192%22%20y1=%22136.8%22%20x2=%22192%22%20y2=%2283.2%22%20stroke=%22%23f472b6%22%20stroke-opacity=%22.35%22%20stroke-width=%222%22/%3E%3Ccircle%20cx=%22192%22%20cy=%22136.8%22%20r=%223.2%22%20fill=%22%2322d3ee%22/%3E%3Ccircle%20cx=%22192%22%20cy=%2283.2%22%20r=%223.2%22%20fill=%22%238b5cf6%22/%3E%3Cline%20x1=%22204%22%20y1=%22159.0%22%20x2=%22204%22%20y2=%2261.0%22%20stroke=%22%23f472b6%22%20stroke-opacity=%22.35%22%20stroke-width=%222%22/%3E%3Ccircle%20cx=%22204%22%20cy=%22159.0%22%20r=%223.2%22%20fill=%22%2322d3ee%22/%3E%3Ccircle%20cx=%22204%22%20cy=%2261.0%22%20r=%223.2%22%20fill=%22%238b5cf6%22/%3E%3Cline%20x1=%22216%22%20y1=%22172.8%22%20x2=%22216%22%20y2=%2247.2%22%20stroke=%22%23f472b6%22%20stroke-opacity=%22.35%22%20stroke-width=%222%22/%3E%3Ccircle%20cx=%22216%22%20cy=%22172.8%22%20r=%223.2%22%20fill=%22%2322d3ee%22/%3E%3Ccircle%20cx=%22216%22%20cy=%2247.2%22%20r=%223.2%22%20fill=%22%238b5cf6%22/%3E%3Cline%20x1=%22228%22%20y1=%22175.6%22%20x2=%22228%22%20y2=%2244.4%22%20stroke=%22%23f472b6%22%20stroke-opacity=%22.35%22%20stroke-width=%222%22/%3E%3Ccircle%20cx=%22228%22%20cy=%22175.6%22%20r=%223.2%22%20fill=%22%2322d3ee%22/%3E%3Ccircle%20cx=%22228%22%20cy=%2244.4%22%20r=%223.2%22%20fill=%22%238b5cf6%22/%3E%3Cline%20x1=%22240%22%20y1=%22167.2%22%20x2=%22240%22%20y2=%2252.8%22%20stroke=%22%23f472b6%22%20stroke-opacity=%22.35%22%20stroke-width=%222%22/%3E%3Ccircle%20cx=%22240%22%20cy=%22167.2%22%20r=%223.2%22%20fill=%22%2322d3ee%22/%3E%3Ccircle%20cx=%22240%22%20cy=%2252.8%22%20r=%223.2%22%20fill=%22%238b5cf6%22/%3E%3Cline%20x1=%22252%22%20y1=%22148.8%22%20x2=%22252%22%20y2=%2271.2%22%20stroke=%22%23f472b6%22%20stroke-opacity=%22.35%22%20stroke-width=%222%22/%3E%3Ccircle%20cx=%22252%22%20cy=%22148.8%22%20r=%223.2%22%20fill=%22%2322d3ee%22/%3E%3Ccircle%20cx=%22252%22%20cy=%2271.2%22%20r=%223.2%22%20fill=%22%238b5cf6%22/%3E%3Cline%20x1=%22264%22%20y1=%22123.7%22%20x2=%22264%22%20y2=%2296.3%22%20stroke=%22%23f472b6%22%20stroke-opacity=%22.35%22%20stroke-width=%222%22/%3E%3Ccircle%20cx=%22264%22%20cy=%22123.7%22%20r=%223.2%22%20fill=%22%2322d3ee%22/%3E%3Ccircle%20cx=%22264%22%20cy=%2296.3%22%20r=%223.2%22%20fill=%22%238b5cf6%22/%3E%3Cline%20x1=%22276%22%20y1=%2296.3%22%20x2=%22276%22%20y2=%22123.7%22%20stroke=%22%23f472b6%22%20stroke-opacity=%22.35%22%20stroke-width=%222%22/%3E%3Ccircle%20cx=%22276%22%20cy=%2296.3%22%20r=%223.2%22%20fill=%22%2322d3ee%22/%3E%3Ccircle%20cx=%22276%22%20cy=%22123.7%22%20r=%223.2%22%20fill=%22%238b5cf6%22/%3E%3Cline%20x1=%22288%22%20y1=%2271.2%22%20x2=%22288%22%20y2=%22148.8%22%20stroke=%22%23f472b6%22%20stroke-opacity=%22.35%22%20stroke-width=%222%22/%3E%3Ccircle%20cx=%22288%22%20cy=%2271.2%22%20r=%223.2%22%20fill=%22%2322d3ee%22/%3E%3Ccircle%20cx=%22288%22%20cy=%22148.8%22%20r=%223.2%22%20fill=%22%238b5cf6%22/%3E%3Cline%20x1=%22300%22%20y1=%2252.8%22%20x2=%22300%22%20y2=%22167.2%22%20stroke=%22%23f472b6%22%20stroke-opacity=%22.35%22%20stroke-width=%222%22/%3E%3Ccircle%20cx=%22300%22%20cy=%2252.8%22%20r=%223.2%22%20fill=%22%2322d3ee%22/%3E%3Ccircle%20cx=%22300%22%20cy=%22167.2%22%20r=%223.2%22%20fill=%22%238b5cf6%22/%3E%3Cline%20x1=%22312%22%20y1=%2244.4%22%20x2=%22312%22%20y2=%22175.6%22%20stroke=%22%23f472b6%22%20stroke-opacity=%22.35%22%20stroke-width=%222%22/%3E%3Ccircle%20cx=%22312%22%20cy=%2244.4%22%20r=%223.2%22%20fill=%22%2322d3ee%22/%3E%3Ccircle%20cx=%22312%22%20cy=%22175.6%22%20r=%223.2%22%20fill=%22%238b5cf6%22/%3E%3Cline%20x1=%22324%22%20y1=%2247.2%22%20x2=%22324%22%20y2=%22172.8%22%20stroke=%22%23f472b6%22%20stroke-opacity=%22.35%22%20stroke-width=%222%22/%3E%3Ccircle%20cx=%22324%22%20cy=%2247.2%22%20r=%223.2%22%20fill=%22%2322d3ee%22/%3E%3Ccircle%20cx=%22324%22%20cy=%22172.8%22%20r=%223.2%22%20fill=%22%238b5cf6%22/%3E%3Cline%20x1=%22336%22%20y1=%2261.0%22%20x2=%22336%22%20y2=%22159.0%22%20stroke=%22%23f472b6%22%20stroke-opacity=%22.35%22%20stroke-width=%222%22/%3E%3Ccircle%20cx=%22336%22%20cy=%2261.0%22%20r=%223.2%22%20fill=%22%2322d3ee%22/%3E%3Ccircle%20cx=%22336%22%20cy=%22159.0%22%20r=%223.2%22%20fill=%22%238b5cf6%22/%3E%3Cline%20x1=%22348%22%20y1=%2283.2%22%20x2=%22348%22%20y2=%22136.8%22%20stroke=%22%23f472b6%22%20stroke-opacity=%22.35%22%20stroke-width=%222%22/%3E%3Ccircle%20cx=%22348%22%20cy=%2283.2%22%20r=%223.2%22%20fill=%22%2322d3ee%22/%3E%3Ccircle%20cx=%22348%22%20cy=%22136.8%22%20r=%223.2%22%20fill=%22%238b5cf6%22/%3E%3C/svg%3E") repeat-x right center / 360px 220px;
    opacity: .5; animation: flow 22s linear infinite;
    -webkit-mask-image: linear-gradient(to left, #000 25%, transparent 95%);
    mask-image: linear-gradient(to left, #000 25%, transparent 95%);
}
.hero > * {position: relative; z-index: 1;}
.main-title {
    display: block; font-family: 'Space Grotesk','Inter',sans-serif;
    font-size: 3.1rem; font-weight: 700; letter-spacing: -0.04em;
    line-height: 1.3; padding-top: 0.12em; padding-bottom: 0.08em; margin: 0; overflow: visible;
}
.main-title .grad {
    background: var(--grad); background-size: 200% auto;
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
    animation: shimmer 7s linear infinite;
}
.subtitle {color:#a9b8d0; font-size:1.08rem; margin-top:.35rem; max-width:640px; line-height:1.55}
.chips {display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1.2rem}
.chip {
    display:inline-flex; align-items:center; gap:.45rem; padding:.3rem .8rem; border-radius:999px;
    font-size:.78rem; font-weight:600; color:#cfe3ff;
    border:1px solid rgba(255,255,255,.15); background:rgba(255,255,255,.055); backdrop-filter: blur(6px);
}
.chip .dot {width:8px; height:8px; border-radius:50%; background:#34d399; animation:pulse 2s infinite}

.metric-card {padding:.85rem 1rem; border:1px solid var(--line); border-radius:14px}
.small-note {font-size:.88rem; color:var(--muted)}

/* ---------- workflow steps ---------- */
.flow {display:flex; flex-wrap:wrap; align-items:center; gap:.55rem; margin-top:.6rem}
.flow .step {
    display:inline-flex; align-items:center; gap:.5rem; padding:.45rem .85rem; border-radius:12px;
    font-size:.85rem; font-weight:600; border:1px solid var(--line);
    background:linear-gradient(160deg, rgba(255,255,255,.07), rgba(255,255,255,.02));
    transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease;
}
.flow .step:hover {transform:translateY(-2px); border-color:rgba(139,92,246,.55); box-shadow:0 12px 28px -14px rgba(139,92,246,.7)}
.flow .step b {
    display:inline-flex; align-items:center; justify-content:center; width:22px; height:22px; border-radius:50%;
    font-size:.72rem; color:#071018; background:var(--grad);
}
.flow .arrow {color:var(--muted); font-weight:700}

/* ---------- headings & dividers ---------- */
h2, h3 {font-family:'Space Grotesk','Inter',sans-serif; letter-spacing:-0.02em; font-weight:700}
h4 {font-family:'Space Grotesk','Inter',sans-serif; letter-spacing:-0.01em}
hr {border:none !important; height:1px !important; background:linear-gradient(90deg, transparent, rgba(139,92,246,.55), rgba(34,211,238,.45), transparent) !important; margin:1.4rem 0 !important}
div[data-testid="stCaptionContainer"], .stApp small {color: var(--muted);}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(185deg, #0c1428 0%, #090e1c 100%);
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] h1 {font-family:'Space Grotesk','Inter',sans-serif; font-weight:700; letter-spacing:-0.03em; color:#fff}
section[data-testid="stSidebar"] div[role="radiogroup"] {gap:.4rem}
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    width:100%; padding:.62rem .85rem; border-radius:12px; border:1px solid transparent;
    transition: all .2s ease; cursor:pointer;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {background:rgba(255,255,255,.05); border-color:var(--line); transform:translateX(3px)}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(120deg, rgba(34,211,238,.17), rgba(139,92,246,.22));
    border-color: rgba(139,92,246,.5); box-shadow: 0 0 26px -10px rgba(139,92,246,.8);
}

/* ---------- widgets ---------- */
div[data-testid="stWidgetLabel"] p {
    color:#a9b8d0; font-weight:600; font-size:.78rem; letter-spacing:.06em; text-transform:uppercase;
}
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, div[data-baseweb="base-input"] {
    background: rgba(255,255,255,.045) !important; border:1px solid var(--line) !important;
    border-radius:12px !important; transition: border-color .2s ease, box-shadow .2s ease;
}
div[data-baseweb="select"] > div:hover, div[data-baseweb="input"] > div:hover {border-color: rgba(34,211,238,.5) !important}
div[data-baseweb="select"] > div:focus-within, div[data-baseweb="input"] > div:focus-within {
    border-color: var(--cyan) !important; box-shadow: 0 0 0 3px rgba(34,211,238,.18);
}
div[data-baseweb="popover"] ul {background:#0e1527 !important; border:1px solid var(--line); border-radius:12px}

button[data-testid="stBaseButton-primary"], button[kind="primary"] {
    background: var(--grad) !important; background-size:160% auto !important; color:#06101a !important;
    border:none !important; border-radius:14px !important; font-weight:700 !important; letter-spacing:.01em;
    padding:.7rem 1rem !important; box-shadow: 0 14px 34px -14px rgba(139,92,246,.9);
    transition: transform .2s ease, box-shadow .2s ease, background-position .4s ease !important;
}
button[data-testid="stBaseButton-primary"]:hover:not(:disabled), button[kind="primary"]:hover:not(:disabled) {
    transform: translateY(-2px); background-position: 100% center !important; box-shadow: 0 20px 44px -14px rgba(34,211,238,.75);
}
button[data-testid="stBaseButton-primary"]:disabled, button[kind="primary"]:disabled {opacity:.4; box-shadow:none; filter:saturate(.6)}
button[data-testid="stBaseButton-secondary"], .stDownloadButton button {
    background: rgba(255,255,255,.05) !important; border:1px solid var(--line) !important; border-radius:12px !important;
    color: var(--txt) !important; font-weight:600 !important; transition: all .2s ease !important;
}
button[data-testid="stBaseButton-secondary"]:hover, .stDownloadButton button:hover {
    border-color: var(--cyan) !important; box-shadow: 0 10px 26px -14px rgba(34,211,238,.9); transform: translateY(-2px);
}

/* ---------- metrics ---------- */
div[data-testid="stMetric"] {
    position:relative; overflow:hidden; padding:16px 18px; border-radius:16px; border:1px solid var(--line);
    background: linear-gradient(160deg, rgba(255,255,255,.065), rgba(255,255,255,.02));
    transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
}
div[data-testid="stMetric"]::before {content:""; position:absolute; left:0; right:0; top:0; height:3px; background:var(--grad)}
div[data-testid="stMetric"]:hover {transform:translateY(-3px); border-color:rgba(139,92,246,.5); box-shadow:0 20px 44px -20px rgba(139,92,246,.7)}
div[data-testid="stMetricLabel"] p {color:var(--muted); font-weight:600}
div[data-testid="stMetricValue"] {font-family:'Space Grotesk','Inter',sans-serif; font-weight:700; letter-spacing:-0.02em}

/* ---------- tabs ---------- */
div[data-baseweb="tab-list"] {
    gap:.35rem; padding:.35rem; border-radius:14px; width:fit-content; max-width:100%;
    background:rgba(255,255,255,.035); border:1px solid var(--line);
}
button[data-baseweb="tab"] {border-radius:10px; padding:.5rem 1.05rem; background:transparent; transition:all .2s ease}
button[data-baseweb="tab"] p {color:var(--muted); font-weight:600}
button[data-baseweb="tab"]:hover p {color:#fff}
button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(120deg, rgba(34,211,238,.2), rgba(139,92,246,.3)); box-shadow: 0 8px 22px -12px rgba(139,92,246,.9);
}
button[data-baseweb="tab"][aria-selected="true"] p {color:#fff}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {display:none}

/* ---------- alerts, expanders, code, tables, viewer ---------- */
div[data-testid="stAlert"] {border-radius:14px; border:1px solid var(--line); backdrop-filter: blur(6px)}
div[data-testid="stExpander"] details {border:1px solid var(--line) !important; border-radius:14px !important; background:rgba(255,255,255,.03)}
div[data-testid="stExpander"] summary:hover {color:var(--cyan)}
div[data-testid="stCode"], .stCodeBlock {border-radius:14px; border:1px solid var(--line); overflow:hidden}
div[data-testid="stDataFrame"] {border-radius:14px; overflow:hidden; border:1px solid var(--line)}
iframe {border-radius:20px; border:1px solid rgba(139,92,246,.35) !important; box-shadow:0 30px 80px -30px rgba(34,211,238,.35), 0 0 0 1px rgba(255,255,255,.03)}
div[data-testid="stImage"] img, div[data-testid="stPyplot"] img {border-radius:14px}
.stMarkdown a {color:var(--cyan); text-decoration:none; border-bottom:1px dashed rgba(34,211,238,.5)}
.stMarkdown a:hover {color:#fff; border-bottom-color:#fff}
.stMarkdown ul {list-style:none; padding-left:.2rem}
.stMarkdown ul > li {position:relative; padding-left:1.3rem; margin:.3rem 0}
.stMarkdown ul > li::before {content:""; position:absolute; left:0; top:.62em; width:7px; height:7px; border-radius:50%; background:var(--grad)}
.stMarkdown ol {list-style:none; counter-reset:step; padding-left:0}
.stMarkdown ol > li {
    counter-increment:step; position:relative; margin:.55rem 0; padding:.75rem 1rem .75rem 3.3rem;
    border-radius:14px; border:1px solid var(--line); background:linear-gradient(160deg, rgba(255,255,255,.05), rgba(255,255,255,.015));
    transition: transform .2s ease, border-color .2s ease;
}
.stMarkdown ol > li:hover {transform:translateX(4px); border-color:rgba(34,211,238,.45)}
.stMarkdown ol > li::before {
    content:counter(step); position:absolute; left:.85rem; top:50%; transform:translateY(-50%);
    width:30px; height:30px; border-radius:50%; display:flex; align-items:center; justify-content:center;
    font:700 .85rem 'Space Grotesk',sans-serif; color:#071018; background:var(--grad);
}
::-webkit-scrollbar {width:10px; height:10px}
::-webkit-scrollbar-thumb {background:rgba(139,92,246,.4); border-radius:10px}
::-webkit-scrollbar-track {background:transparent}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_alpha_fold(accession):
    """Fetch AlphaFold prediction metadata and its coordinate file."""
    response = requests.get(AF_API.format(accession), timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not payload:
        raise ValueError("No AlphaFold prediction was returned for this UniProt accession.")
    item = payload[0] if isinstance(payload, list) else payload

    pdb_url = item.get("pdbUrl") or item.get("pdb_url")
    cif_url = item.get("cifUrl") or item.get("cif_url")
    sequence = item.get("sequence") or item.get("uniprotSequence") or ""
    if not pdb_url and not cif_url:
        raise ValueError("The API response did not contain a PDB or mmCIF download URL.")

    # Prefer PDB because its B-factor field is straightforward to inspect;
    # use mmCIF when a PDB URL is not supplied.
    structure_url = pdb_url or cif_url
    file_response = requests.get(structure_url, timeout=60)
    file_response.raise_for_status()
    return item, file_response.text, ("pdb" if pdb_url else "cif"), sequence


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_uniprot(accession):
    response = requests.get(UNIPROT_API.format(accession), timeout=30)
    response.raise_for_status()
    data = response.json()
    sequence = data.get("sequence", {}).get("value", "")
    protein_name = (
        data.get("proteinDescription", {})
        .get("recommendedName", {})
        .get("fullName", {})
        .get("value", accession)
    )
    organism = data.get("organism", {}).get("scientificName", "Not available")
    return sequence, protein_name, organism


def safe_sequence(sequence):
    """Keep standard amino-acid letters for sequence-level calculations."""
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", (sequence or "").upper())


def extract_confidence(structure_text, file_format):
    """Read per-residue AlphaFold confidence from coordinate B-factor values."""
    parser = PDBParser(QUIET=True) if file_format == "pdb" else MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure("model", io.StringIO(structure_text))
        values = []
        for residue in structure.get_residues():
            if "CA" in residue:
                values.append(float(residue["CA"].get_bfactor()))
        if not values:
            return [], None
        return values, sum(values) / len(values)
    except Exception:
        # Keep the app usable if a provider changes a coordinate-file detail.
        return [], None


def make_viewer(structure_text, file_format, colour_mode):
    view = py3Dmol.view(width="100%", height=520)
    view.addModel(structure_text, file_format)
    if colour_mode == "Confidence (pLDDT)":
        view.setStyle({"cartoon": {
            "colorscheme": {
                "prop": "b", "gradient": "roygb", "min": 50, "max": 100
            }
        }})
    elif colour_mode == "Chain":
        view.setStyle({"cartoon": {"color": "chain"}})
    else:
        view.setStyle({"cartoon": {"color": "spectrum"}})
    view.addStyle({"hetflag": True}, {"stick": {"radius": 0.18}})
    view.zoomTo()
    view.setBackgroundColor("#0a1020")
    return view._make_html()


def sequence_properties(sequence):
    """Calculate numeric physicochemical descriptors from the supplied sequence."""
    cleaned = safe_sequence(sequence)
    if not cleaned:
        raise ValueError("No standard amino-acid residues were found in this sequence.")
    analysis = ProteinAnalysis(cleaned)
    counts = analysis.count_amino_acids()
    length = len(cleaned)
    # Ikai's aliphatic index: relative volume contribution of Ala, Val, Ile and Leu.
    aliphatic_index = 100 * (
        counts["A"] / length
        + 2.9 * counts["V"] / length
        + 3.9 * (counts["I"] + counts["L"]) / length
    )
    return {
        "Molecular weight (Da)": analysis.molecular_weight(),
        "Theoretical pI": analysis.isoelectric_point(),
        "Instability index": analysis.instability_index(),
        "Aliphatic index": aliphatic_index,
        "GRAVY": analysis.gravy(),
    }


def format_sequence_property(name, value):
    if name == "Molecular weight (Da)":
        return f"{value:,.1f} Da"
    if name == "GRAVY":
        return f"{value:.3f}"
    return f"{value:.2f}"


@st.cache_data(show_spinner=False, ttl=3600)
def comparative_property_data():
    """Fetch each curated UniProt sequence and calculate properties live."""
    rows = []
    errors = []
    for display_name, info in PROTEINS.items():
        accession = info["accession"]
        try:
            sequence, protein_name, organism = fetch_uniprot(accession)
            sequence = safe_sequence(sequence)
            if not sequence:
                raise ValueError("UniProt returned no usable protein sequence.")
            props = sequence_properties(sequence)
            rows.append({
                "Protein": display_name,
                "UniProt accession": accession,
                "Sequence length (aa)": len(sequence),
                "Molecular weight (kDa)": props["Molecular weight (Da)"] / 1000,
                "Theoretical pI": props["Theoretical pI"],
                "Instability index": props["Instability index"],
                "Aliphatic index": props["Aliphatic index"],
                "GRAVY": props["GRAVY"],
            })
        except Exception as exc:
            errors.append(f"{display_name} ({accession}): {exc}")
    return pd.DataFrame(rows), errors


def confidence_interpretation(score):
    if score is None:
        return "Confidence information could not be extracted from this coordinate file."
    if score >= 90:
        return "The mean score is in the very-high confidence band. This supports confidence in local structural geometry, but it is not experimental validation."
    if score >= 70:
        return "The mean score is in the confident band overall. Some regions may still be less certain."
    if score >= 50:
        return "The mean score suggests low-confidence regions may be present. Interpret flexible or disordered parts cautiously."
    return "The mean score is very low. The predicted coordinates should be interpreted with substantial caution."


st.sidebar.title("🧬 ProteinScope")
st.sidebar.caption("Explore protein sequences, predicted structures and confidence.")
page = st.sidebar.radio(
    "Navigate",
    ["Explore proteins", "Comparative dashboard", "How it works", "About & limitations"],
)
st.sidebar.divider()
st.sidebar.caption("Educational project · Live data from AlphaFold DB and UniProt")

st.markdown(
    '<div class="hero">'
    '<div class="main-title">🧬 <span class="grad">ProteinScope</span></div>'
    '<div class="subtitle">An interactive AI-assisted protein structure visualization and analysis system</div>'
    '<div class="chips">'
    '<span class="chip"><span class="dot"></span>Live data</span>'
    '<span class="chip">AlphaFold DB</span>'
    '<span class="chip">UniProt</span>'
    '<span class="chip">Biopython</span>'
    '<span class="chip">3Dmol.js</span>'
    '</div></div>',
    unsafe_allow_html=True,
)
st.divider()

if page == "Explore proteins":
    col_a, col_b = st.columns([1, 2])
    with col_a:
        protein_options = list(PROTEINS.keys()) + ["Custom UniProt accession"]
        choice = st.selectbox("Choose a demonstration protein", protein_options)
        if choice == "Custom UniProt accession":
            accession = st.text_input(
                "Enter UniProt accession",
                placeholder="e.g. P61626",
                help="ProteinScope will retrieve the AlphaFold model and calculate properties from the retrieved sequence when available.",
            ).strip().upper()
            st.write("**Biological role:** Custom UniProt entry")
            st.info("Custom accessions require a corresponding AlphaFold DB prediction and an accessible sequence.")
        else:
            protein_info = PROTEINS[choice]
            accession = protein_info["accession"]
            st.write(f"**UniProt accession:** `{accession}`")
            st.write(f"**Biological role:** {protein_info['role']}")
            st.info(protein_info["note"])
        colour_mode = st.selectbox(
            "3D structure colouring",
            ["Confidence (pLDDT)", "Rainbow by residue", "Chain"],
        )
        load = st.button(
            "Load protein structure", type="primary", use_container_width=True,
            disabled=not bool(re.fullmatch(r"[A-Z0-9]{6,10}", accession or "")),
        )
    with col_b:
        st.markdown("#### What this application demonstrates")
        st.write(
            "A protein is represented by an amino-acid sequence. AlphaFold DB provides "
            "predicted three-dimensional models for many UniProt entries. ProteinScope "
            "retrieves a selected model and presents its sequence, structure and available "
            "confidence information."
        )
        st.caption("An internet connection is required to retrieve live AlphaFold DB and UniProt records.")
        st.markdown("**Workflow**")
        _steps = ["Select protein", "Retrieve public prediction", "Inspect sequence",
                  "Explore 3D structure", "Interpret confidence"]
        st.markdown(
            '<div class="flow">'
            + '<span class="arrow">→</span>'.join(
                f'<span class="step"><b>{i}</b>{s}</span>' for i, s in enumerate(_steps, 1)
            )
            + '</div>',
            unsafe_allow_html=True,
        )

    if "loaded_accession" not in st.session_state:
        st.session_state.loaded_accession = None
    if "loaded_colour_mode" not in st.session_state:
        st.session_state.loaded_colour_mode = "Confidence (pLDDT)"

    if load:
        st.session_state.loaded_accession = accession
        st.session_state.loaded_colour_mode = colour_mode

    # Show results only for the protein currently selected. Selecting a new
    # protein requires the user to press Load, preventing stale results.
    active_accession = (
        st.session_state.loaded_accession
        if st.session_state.loaded_accession == accession
        else None
    )
    if active_accession:
        active_choice = next(
            (name for name, info in PROTEINS.items() if info["accession"] == active_accession),
            choice,
        )
        active_info = PROTEINS[active_choice]
        active_colour = colour_mode
        st.session_state.loaded_colour_mode = colour_mode
        try:
            with st.spinner("Retrieving prediction metadata and structure coordinates..."):
                metadata, structure_text, fmt, api_sequence = fetch_alpha_fold(active_accession)
                uniprot_warning = None
                try:
                    uniprot_sequence, protein_name, organism = fetch_uniprot(active_accession)
                except requests.RequestException as exc:
                    uniprot_sequence = ""
                    protein_name = metadata.get("uniprotDescription") or active_accession
                    organism = metadata.get("organismScientificName") or "Not available"
                    uniprot_warning = str(exc)

                sequence = safe_sequence(api_sequence or uniprot_sequence)
                if not protein_name or protein_name == active_accession:
                    protein_name = metadata.get("uniprotDescription") or active_accession
                if not organism or organism == "Not available":
                    organism = metadata.get("organismScientificName") or "Not available"

                confidence_values, mean_confidence = extract_confidence(structure_text, fmt)

            st.success("Protein structure and available sequence data loaded successfully.")
            if uniprot_warning:
                st.warning(
                    "UniProt could not be reached, so ProteinScope is using the sequence and "
                    "basic metadata supplied by AlphaFold DB where available. Some UniProt "
                    f"annotations may be unavailable. Details: {uniprot_warning}"
                )

            st.subheader(protein_name)
            st.caption(f"{organism} · UniProt {active_accession} · Structure format: {fmt.upper()}")

            m1, m2, m3 = st.columns(3)
            m1.metric("Sequence length", f"{len(sequence):,} aa" if sequence else "Not available")
            m2.metric("Mean pLDDT", f"{mean_confidence:.1f}/100" if mean_confidence is not None else "Not available")
            m3.metric("Confidence residues", f"{len(confidence_values):,}" if confidence_values else "Not available")

            with st.expander("Download original structure coordinates"):
                st.download_button(
                    "Download structure file",
                    data=structure_text,
                    file_name=f"AF-{active_accession}-F1.{fmt}",
                    mime="chemical/x-pdb" if fmt == "pdb" else "chemical/x-mmcif",
                    use_container_width=True,
                )

            st.markdown("#### Interactive 3D predicted structure")
            st.caption("Drag to rotate; scroll to zoom. Confidence colouring uses the coordinate-file B-factor field when available.")
            viewer_html = make_viewer(structure_text, fmt, active_colour)
            st_html(viewer_html, height=550, scrolling=False)

            tab1, tab2, tab3 = st.tabs(
                ["Amino-acid sequence", "Confidence analysis", "Structure notes"]
            )
            with tab1:
                if sequence:
                    st.markdown("**Amino-acid sequence (one-letter code)**")
                    st.code(
                        "\n".join(sequence[i:i + 10] for i in range(0, len(sequence), 10)),
                        language="text",
                    )
                    fasta = f">{active_accession}|{protein_name}\n" + "\n".join(
                        sequence[i:i + 60] for i in range(0, len(sequence), 60)
                    )
                    st.download_button(
                        "Download FASTA sequence",
                        data=fasta,
                        file_name=f"{active_accession}.fasta",
                        mime="text/plain",
                    )

                    st.markdown("**Physicochemical properties**")
                    try:
                        props = sequence_properties(sequence)
                        prop_cols = st.columns(len(props))
                        for col, (name, value) in zip(prop_cols, props.items()):
                            col.metric(name, format_sequence_property(name, value))
                        st.caption(
                            "Calculated dynamically from the loaded amino-acid sequence using Biopython. "
                            "These are theoretical sequence descriptors, not experimental measurements."
                        )
                    except Exception as exc:
                        st.warning(f"Sequence property calculation unavailable: {exc}")

                    try:
                        analysis = ProteinAnalysis(sequence)
                        aa_counts = analysis.count_amino_acids()
                        aa_df = pd.DataFrame(
                            [{"Amino acid": aa, "Count": int(count)}
                             for aa, count in aa_counts.items()]
                        )
                        st.markdown("**Amino-acid composition**")
                        chart_col, data_col = st.columns([1.35, 1])
                        with chart_col:
                            st.bar_chart(aa_df.set_index("Amino acid"), color="#8b5cf6")
                        with data_col:
                            st.dataframe(aa_df, hide_index=True, use_container_width=True)
                        st.download_button(
                            "Download composition as CSV",
                            data=aa_df.to_csv(index=False),
                            file_name=f"{active_accession}_amino_acid_composition.csv",
                            mime="text/csv",
                        )
                    except Exception as exc:
                        st.warning(f"Composition analysis unavailable: {exc}")
                else:
                    st.warning("The sequence was not available from the selected records.")

            with tab2:
                if confidence_values:
                    df = pd.DataFrame({
                        "Residue position": range(1, len(confidence_values) + 1),
                        "pLDDT": confidence_values,
                    })
                    fig, ax = plt.subplots(figsize=(10, 3.4))
                    fig.patch.set_alpha(0)
                    ax.set_facecolor("none")
                    # Soft confidence bands (AlphaFold palette) behind the curve.
                    for lo, hi, band_colour in [(0, 50, "#FF7D45"), (50, 70, "#FFDB13"),
                                                (70, 90, "#65CBF3"), (90, 105, "#0053D6")]:
                        ax.axhspan(lo, hi, color=band_colour, alpha=0.10, linewidth=0)
                    ax.fill_between(df["Residue position"], df["pLDDT"], 0, color="#22d3ee", alpha=0.12)
                    ax.plot(df["Residue position"], df["pLDDT"], linewidth=1.8, color="#22d3ee")
                    ax.set_ylim(0, 105)
                    ax.set_xlim(1, max(2, len(df)))
                    ax.set_xlabel("Residue position", color="#a9b8d0")
                    ax.set_ylabel("pLDDT", color="#a9b8d0")
                    ax.set_title("Per-residue prediction confidence", color="#e6edf7",
                                 fontsize=12, fontweight="bold", loc="left")
                    ax.tick_params(colors="#8fa0ba")
                    ax.grid(alpha=0.12, color="#ffffff")
                    for side in ("top", "right"):
                        ax.spines[side].set_visible(False)
                    for side in ("left", "bottom"):
                        ax.spines[side].set_color("#334155")
                    fig.tight_layout()
                    st.pyplot(fig, clear_figure=True, transparent=True)

                    high = sum(v >= 90 for v in confidence_values)
                    medium = sum(70 <= v < 90 for v in confidence_values)
                    low = sum(v < 70 for v in confidence_values)
                    total = len(confidence_values)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("High confidence (≥90)", high, f"{high / total * 100:.1f}%")
                    c2.metric("Medium (70–<90)", medium, f"{medium / total * 100:.1f}%")
                    c3.metric("Low (<70)", low, f"{low / total * 100:.1f}%")
                    st.info(confidence_interpretation(mean_confidence))
                    st.download_button(
                        "Download confidence values as CSV",
                        data=df.to_csv(index=False),
                        file_name=f"{active_accession}_confidence.csv",
                        mime="text/csv",
                    )
                    st.caption(
                        "These descriptive pLDDT bands are commonly used for interpretation. "
                        "Confidence is not the same as experimental validation."
                    )
                else:
                    st.info(
                        "Per-residue confidence could not be extracted from this coordinate "
                        "file. The 3D model can still be explored."
                    )

            with tab3:
                st.write(
                    "**What the model represents:** an AI-generated structural prediction "
                    "retrieved from the AlphaFold Protein Structure Database."
                )
                st.write(
                    "**What it does not prove:** the structure is not automatically "
                    "experimentally verified, and a monomer prediction does not establish "
                    "a complete multimeric biological assembly."
                )
                st.write("**Interpretation reminder:** protein function depends on more than a predicted 3D shape; biological context, processing, binding partners and experimental evidence matter.")
                model_id = (
                    metadata.get("modelEntityId")
                    or metadata.get("entryId")
                    or f"AF-{active_accession}-F1"
                )
                st.code(model_id)
                st.markdown(f"[Open AlphaFold entry](https://alphafold.com/entry/{model_id})")

        except requests.HTTPError as exc:
            st.error(f"Data service returned an HTTP error: {exc}. Check the accession or try again later.")
        except requests.RequestException as exc:
            st.error(f"Network error while contacting public databases: {exc}")
        except Exception as exc:
            st.error(f"Could not load this protein: {exc}")
            st.caption("Try another example or check the project troubleshooting notes.")

elif page == "Comparative dashboard":
    st.subheader("Comparative dashboard")
    st.write(
        "Compare AlphaFold prediction confidence and sequence length across the five "
        "curated proteins, alongside their dynamically calculated physicochemical properties."
    )

    # Preserve the original structure/confidence comparison dataset collected and
    # verified during the project. Mean pLDDT values are from the user's saved results.
    confidence_df = pd.DataFrame([
        {"Protein": "Human lysozyme", "UniProt accession": "P61626", "Sequence length (aa)": 148, "Mean pLDDT": 94.1, "High (≥90)": 130, "Medium (70–<90)": 3, "Low (<70)": 15, "Confidence residues": 148, "Format": "PDB"},
        {"Protein": "Human myoglobin", "UniProt accession": "P02144", "Sequence length (aa)": 154, "Mean pLDDT": 97.2, "High (≥90)": 150, "Medium (70–<90)": 3, "Low (<70)": 1, "Confidence residues": 154, "Format": "PDB"},
        {"Protein": "Human growth hormone", "UniProt accession": "P01241", "Sequence length (aa)": 217, "Mean pLDDT": 79.3, "High (≥90)": 90, "Medium (70–<90)": 73, "Low (<70)": 54, "Confidence residues": 217, "Format": "PDB"},
        {"Protein": "Human haemoglobin subunit beta", "UniProt accession": "P68871", "Sequence length (aa)": 147, "Mean pLDDT": 97.2, "High (≥90)": 143, "Medium (70–<90)": 3, "Low (<70)": 1, "Confidence residues": 147, "Format": "PDB"},
        {"Protein": "Human insulin precursor", "UniProt accession": "P01308", "Sequence length (aa)": 110, "Mean pLDDT": 52.9, "High (≥90)": 0, "Medium (70–<90)": 14, "Low (<70)": 96, "Confidence residues": 110, "Format": "PDB"},
    ])
    st.markdown("### Structure and confidence comparison")
    st.dataframe(confidence_df, hide_index=True, use_container_width=True)

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("**Mean pLDDT comparison**")
        st.bar_chart(confidence_df.set_index("Protein")[["Mean pLDDT"]], color="#22d3ee")
    with chart_right:
        st.markdown("**Sequence length comparison**")
        st.bar_chart(confidence_df.set_index("Protein")[["Sequence length (aa)"]], color="#f472b6")

    st.markdown("### How to read the confidence bands")
    st.markdown("""
    - **High (≥90):** very high local prediction confidence.
    - **Medium (70–<90):** confident prediction, but less certain than the high band.
    - **Low (<70):** low local confidence; interpret the corresponding model regions cautiously.

    The per-residue counts describe how many Cα confidence values fall into each band.
    The overall mean pLDDT is a summary and can hide local differences. pLDDT is not
    experimental validation, a measure of protein stability, or proof of biological function.
    """)

    st.markdown("### Automatic observations")
    st.markdown("""
    - Human myoglobin and human haemoglobin subunit beta share the highest displayed mean pLDDT (97.2/100).
    - Human lysozyme also has a high mean pLDDT (94.1/100).
    - Human growth hormone has an intermediate mean pLDDT (79.3/100) and a more mixed per-residue confidence distribution.
    - Human insulin precursor has the lowest mean pLDDT (52.9/100), with most residues in the low-confidence band.
    - Human growth hormone has the longest sequence (217 aa), while human insulin precursor has the shortest (110 aa).
    """)

    st.markdown("### Scientific notes for this dataset")
    st.markdown("""
    - **Insulin:** P01308 is the precursor sequence/model, not mature processed two-chain insulin.
    - **Haemoglobin:** P68871 represents the individual beta subunit, not the complete haemoglobin tetramer.
    - **Confidence interpretation:** pLDDT describes local model confidence; high values do not establish experimental accuracy, stability, or biological activity.
    - **Mean-score caution:** two proteins can have similar means but different residue-level confidence distributions. Inspect the per-residue plot before interpreting a model.
    """)

    st.divider()
    st.subheader("Comparative physicochemical properties")
    st.write(
        "The following sequence-derived properties are recalculated from the current UniProt "
        "sequences using the same methods used in Explore proteins."
    )
    with st.spinner("Retrieving UniProt sequences and calculating physicochemical properties..."):
        comparison_df, comparison_errors = comparative_property_data()

    if not comparison_df.empty:
        display_df = comparison_df.copy()
        numeric_cols = [
            "Molecular weight (kDa)", "Theoretical pI", "Instability index",
            "Aliphatic index", "GRAVY",
        ]
        display_df[numeric_cols] = display_df[numeric_cols].round(3)
        st.dataframe(display_df, hide_index=True, use_container_width=True)
        st.download_button(
            "Download physicochemical comparison as CSV",
            data=display_df.to_csv(index=False),
            file_name="ProteinScope_physicochemical_comparison.csv",
            mime="text/csv",
        )

        chart_specs = [
            ("Molecular weight (kDa)", "Molecular weight comparison", "Molecular weight (kDa)"),
            ("Instability index", "Instability index comparison", "Instability index"),
            ("GRAVY", "Hydropathicity comparison (GRAVY)", "GRAVY"),
        ]
        chart_colours = {
            "Molecular weight (kDa)": "#22d3ee",
            "Instability index": "#8b5cf6",
            "GRAVY": "#f472b6",
        }
        for column, title, y_label in chart_specs:
            chart_data = comparison_df.set_index("Protein")[[column]]
            st.markdown(f"**{title}**")
            st.bar_chart(chart_data, y=column, color=chart_colours[column])

        st.markdown("**How to interpret these descriptors**")
        st.markdown(
            "- **Molecular weight:** estimated mass calculated from the amino-acid sequence.\n"
            "- **Instability index:** a sequence-based heuristic; values below 40 are conventionally classified as stable by this index, not experimentally proven stable.\n"
            "- **GRAVY:** the average hydropathicity of the sequence; negative values indicate an overall hydrophilic tendency and positive values an overall hydrophobic tendency.\n"
            "- **Theoretical pI:** estimated pH at which the protein has zero net charge.\n"
            "- **Aliphatic index:** reflects the relative content of alanine, valine, isoleucine and leucine; it is sometimes discussed in relation to thermostability, but is not a direct thermal measurement."
        )
        st.warning(
            "Biological context matters: somatotropin and insulin entries are precursor sequences, "
            "and haemoglobin beta is one subunit rather than the complete haemoglobin complex. "
            "Sequence-derived values should not be treated as measured properties of mature proteins."
        )
    else:
        st.error("No comparison data could be retrieved. Check your internet connection and try again.")
    if comparison_errors:
        with st.expander("Entries that could not be analysed"):
            for error in comparison_errors:
                st.write(f"- {error}")
    st.caption("All values are recalculated from the retrieved sequence. UniProt responses are cached for up to one hour.")

elif page == "How it works":
    st.subheader("How ProteinScope works")
    st.markdown("""
    1. **Select a protein:** choose a curated UniProt accession.
    2. **Retrieve prediction metadata:** ProteinScope requests the prediction record from AlphaFold DB.
    3. **Retrieve sequence information:** UniProt is queried; if it is temporarily unavailable, AlphaFold DB metadata is used as a fallback where possible.
    4. **Visualize coordinates:** the PDB/mmCIF coordinate file is rendered as an interactive 3D model with 3Dmol.js through py3Dmol.
    5. **Analyze the sequence:** Biopython calculates physicochemical properties and amino-acid composition directly from the retrieved sequence.
    6. **Compare proteins:** the comparative dashboard retrieves the curated UniProt sequences and recalculates their physicochemical descriptors.
    7. **Analyze confidence:** for coordinate files where the B-factor field is available, per-residue pLDDT values are summarized and plotted.
    8. **Interpret carefully:** confidence scores describe model confidence, not proof of biological activity or experimental accuracy.
    """)
    st.info("The project uses an existing published AI prediction service. It does not train a new protein-folding model.")
    st.markdown("**Technology stack:** Python · Streamlit · AlphaFold DB API · UniProt REST API · Biopython · py3Dmol · Matplotlib · Pandas")

else:
    st.subheader("About the project")
    st.write(
        "ProteinScope is an educational computational-biology demonstrator designed "
        "to connect amino-acid sequences, protein structure and AI-assisted prediction."
    )
    st.markdown("""
    **Scope**
    - Retrieve selected protein prediction records.
    - Display amino-acid sequence and basic composition.
    - Calculate sequence-derived physicochemical properties dynamically.
    - Compare five curated proteins in a dashboard.
    - Accept a custom UniProt accession when a corresponding AlphaFold model is available.
    - Render predicted 3D coordinates.
    - Summarize per-residue confidence where available.
    - Export FASTA, CSV and coordinate files for learning.

    **Limitations**
    - Internet access is required for live data retrieval.
    - The available prediction may represent one chain or one isoform.
    - Predicted structures require experimental context and validation.
    - pLDDT is a local confidence metric; it is not a probability that the protein is biologically correct.
    - Sequence-derived properties are theoretical calculations based on the sequence retrieved at runtime; database sequence versions and precursor processing can affect comparisons.
    - The application is educational and is not a medical or drug-design decision tool.
    """)
    st.markdown(
        "**Course alignment:** Unit III (protein structures, prediction methods, "
        "3D visualization, Python structure analysis and basics of drug design), "
        "supported by Unit II (biomolecules and sequence data) and Unit IV (AI/ML foundations)."
    )
    st.caption(
        "Prepared as a first-year moderate-scope project. Confirm final formatting "
        "and assessment requirements with your course instructor."
    )

st.divider()
st.caption("ProteinScope · Educational use only · Predictions are not experimental proof.")
