# AI-Powered Medical Equipment Supplier Risk Intelligence & Procurement Copilot

M.Tech capstone prototype. A hospital procurement team selects an RFQ, compares
supplier quotations, sees a predicted disruption risk with feature-level
explanations, reads the buyer policy and contract clauses that apply, receives an
evidence-backed recommendation — and then makes the decision themselves, with
every step written to an audit trail.

**The product principle, stated once and enforced throughout: the AI does not
select suppliers. It helps a procurement team make explainable, evidence-backed
decisions. Final award authority stays with the human approver.**

All data is synthetic. Every metric in this repository is a prototype result on
synthetic data and is labelled as such wherever it is displayed.

---

## Quick start

### In VS Code

1. **File → Open Folder…** and pick `procurement-copilot` (the folder containing
   `run.py`). Opening the *parent* folder is the usual mistake — the paths below
   assume `run.py` is at the top level of the workspace.
2. Install the **Python** extension if VS Code prompts you (it is in
   `.vscode/extensions.json`).
3. **⇧⌘P → Python: Select Interpreter** and pick Python 3.10 or newer.
4. Install the two required packages in the VS Code terminal (**⌃`**):
   ```bash
   pip install numpy pypdf
   ```
5. Press **F5** and choose **“Run the app”**, or run `python run.py` in the
   terminal.
6. Open <http://127.0.0.1:8000>. Stop it with **⌃C**.

`.vscode/launch.json` also gives you *Rebuild everything*, *Run tests* and
*Export offline build* on the F5 menu; **⇧⌘B** runs the build task.

### Or from any terminal

```bash
cd procurement-copilot
pip install numpy pypdf
python run.py
```

**Requirements: Python 3.10+, plus `numpy` and `pypdf`.** That is the whole
list. `pip install -r requirements.txt` additionally installs FastAPI (adding
OpenAPI docs at `/docs`), scikit-learn, XGBoost and SHAP — the application
detects each at import time and uses it if present, so every one of them is
optional and the system behaves identically without them.

First run takes about 40 seconds: it ingests the data, indexes the buyer
documents, trains and compares the three risk models, scores all 120 quotations,
and then serves. Subsequent runs start immediately.

```bash
python scripts/build_all.py      # rebuild every artefact from scratch
python tests/test_all.py         # 102 tests, standard library only
python tests/test_huggingface.py # 21 tests against a local mock of the HF router
node tests/render_check.js       # renders all 10 dashboard views headlessly
node tests/css_check.js          # stylesheet regression checks
python scripts/export_static.py  # single-file offline build (demo insurance)
```

---

## The demo journey

1. **Overview** — active RFQs, supplier count, high-risk suppliers, pending approvals, risk distribution.
2. **RFQ-102 supplier comparison** — 8 quotations with price, delivery, quality, risk, risk probability, confidence and decision score.
3. **Get Recommendation** — the recommended supplier with reasons, concerns, a score gauge and cited buyer documentation.
4. **Click any row** — decision-score composition, Shapley risk drivers, confidence breakdown, policy flags with clause citations.
5. **Ask the Copilot** — "Why is this supplier recommended?", "Why is supplier C high risk?", "What does the policy say about suspended certifications?"
6. **Approve or reject** — with an approver and a reason.
7. **Audit Trail** — every recommendation, review and decision, with scores and timestamps.

RFQ-102 (Infusion Pump) and RFQ-101 (Patient Monitor) are both hardened as
repeatable golden paths. Everything is seeded, so the numbers are identical on
every run.

RFQ-102 is a good demo because it makes the central point on its own: the
second-highest-scoring quotation is **blocked** — the supplier is under a
regulatory consent decree — and the cheapest quotation does not win.

---

## What the system computes

Four quantities are kept separate, all the way to the screen, because conflating
them is the most common way a system like this misleads its user.

| Quantity | What it answers | How it is produced |
|---|---|---|
| **Decision score** (0–100) | Which offer is best for this RFQ? | Deterministic weighted sum, fully auditable |
| **Risk probability / level** | What is likely to happen with this supplier? | Learned classifier, cross-validated |
| **Model confidence** (0–100) | How much should I trust that prediction? | Evidence strength — *not* 100 − risk |
| **Policy eligibility** | What is permitted? | Hard rules over regulatory evidence |

A supplier can score well commercially, carry high predicted risk, be predicted
with high confidence, and still be blocked on compliance. The dashboard shows
all four.

### Decision score

```
0.25 × price + 0.20 × delivery + 0.20 × historical performance
+ 0.15 × quality/reliability + 0.15 × regulatory compliance
+ 0.05 × financial/service resilience
```

Weights come from `data/scoring_spec.json` and are read at request time — edit
that file and reload to reconfigure the engine without touching code. The
engine is industry-agnostic; retargeting it means swapping the sub-score
functions in `features.py`.

Price is scored **relative to the competing quotations on the same RFQ**, which
is why scoring always runs per-RFQ across the whole quote set. A bid below 75%
of the peer median stops earning additional points and raises a
`SUSPICIOUS_LOW_BID` flag — on medical equipment an implausibly cheap quote is a
risk signal, not a bargain.

### Risk model

Target: a material supply-disruption or late-delivery event attributed to the
supplier. Logistic Regression (declared baseline), Random Forest and XGBoost are
compared under 5-fold stratified cross-validation repeated 10 times (50 fits
each), and the model with the best mean ROC-AUC is selected — the rule is fixed
before results are seen and XGBoost is not privileged.

**On this dataset Random Forest wins.** See `docs/ML_METHODOLOGY.md` for the
figures and the honest reading of them.

Two things worth knowing:

- `Synthetic Disruption Probability` is the latent generator parameter behind the
  training label. Using it as a feature would be target leakage, so it is
  excluded from the feature set.
- The risk shown for each of the 50 suppliers is its **cross-validated
  out-of-fold** probability — no supplier is ever scored by a model that saw its
  own label. In-sample probabilities would look far more decisive and would be
  meaningless.

### Explainability

Feature attributions are Shapley values, via the `shap` package when installed
and otherwise via permutation sampling (Štrumbelj & Kononenko 2014) — the same
value function, estimated a different way. Each attribution is expressed in
percentage points of predicted risk.

### Confidence

Five components, weighted: data completeness (25%), probability certainty (25%),
model agreement across the three candidates (20%), consistency with the
deterministic evidence view (20%), RFQ data completeness (10%). A supplier can be
confidently high-risk or unconfidently low-risk, and the dashboard says which.

### RAG

The knowledge base holds exactly three document types — buyer RFQs, the buyer
procurement policy, and the buyer master contract. Supplier contracts are out of
scope and cannot enter the index; there is no `supplier_contracts` table and the
document discovery function refuses any other type. Supplier and quotation facts
stay structured in SQLite and are never answered from retrieved text.

Retrieval is hybrid BM25 + TF-IDF cosine with reciprocal rank fusion, over
clause-level chunks that carry `document_type`, `document_id`, `rfq_id`, `page`
and `section`. `docs/ARCHITECTURE.md` explains why that beats a neural embedding
index on a 17-document regulatory corpus, and how to swap one in.

---

## Layout

```
run.py                        one entry point: build if needed, then serve
backend/
  api_fastapi.py              FastAPI adapter (primary, /docs)
  serve_stdlib.py             standard-library adapter (zero-dependency fallback)
  app/
    config.py                 paths, weights, bands, vocabularies, env
    db.py                     SQLite schema (no supplier_contracts)
    ingest.py                 CSV -> SQLite
    features.py               ML features + deterministic sub-scores
    scoring.py                decision-score engine, price rules, eligibility
    recommender.py            combines score + risk + policy + evidence
    copilot.py                intent routing, grounded answers, LLM-optional
    audit.py                  approvals + append-only audit trail
    routes.py                 framework-agnostic handlers (one route table)
    ml/                       numpy_models, backends, evaluate, explain,
                              confidence, train
    rag/                      parse, index, retrieve
frontend/                     index.html, styles.css, app.js (no build step)
data/                         4 CSVs + scoring_spec.json
rag_documents/                policy, master contract, RFQ-101..115 PDFs
models/                       generated: db, model artefacts, RAG index
scripts/                      build_all.py, export_static.py
tests/                        test_all.py (86), render_check.js
docs/                         ARCHITECTURE, ML_METHODOLOGY, DEMO_SCRIPT,
                              API, LIMITATIONS, dashboard_design.png
```

Both server adapters register the same handlers from `routes.py`, so the HTTP
surface cannot drift between them.

---

## API

`GET /api/dashboard/summary` · `GET /api/rfqs` · `GET /api/rfqs/{id}` ·
`GET /api/rfqs/{id}/suppliers` · `GET /api/rfqs/{id}/suppliers/{sid}` ·
`POST /api/rfqs/{id}/recommendation` · `GET /api/suppliers` ·
`GET /api/suppliers/{id}` · `GET /api/suppliers/{id}/risk` · `GET /api/risk` ·
`GET /api/model` · `GET /api/model/evaluation` · `GET /api/knowledge-base` ·
`GET /api/documents/search` · `POST /api/copilot/query` · `POST /api/approvals` ·
`GET /api/approvals` · `POST /api/reviews` · `GET /api/audit` ·
`GET /api/analytics` · `GET /api/settings` · `GET /api/health`

Full detail in `docs/API.md`.

---

## Enabling Hugging Face (one key, both AI components)

```bash
cp .env.example .env
# set HF_API_KEY=hf_...   (read token from https://huggingface.co/settings/tokens)
python scripts/build_all.py     # re-embeds the knowledge base, caches the vectors
python run.py
```

That single key switches on **both** AI slots — no other variable is required:

| Slot | Off (default) | With `HF_API_KEY` |
|---|---|---|
| **Generation** | deterministic grounded answers | `meta-llama/Llama-3.3-70B-Instruct` via the HF router |
| **Retrieval** | BM25 + TF-IDF | BM25 + TF-IDF **+** `all-MiniLM-L6-v2` embeddings |

The provider is inferred from whichever key is present, so `LLM_PROVIDER` is
only needed to override that. Override models with `HF_CHAT_MODEL` /
`HF_EMBED_MODEL`; force the lexical retriever with `USE_DENSE_RETRIEVAL=0`.

**Nothing about this can break the demo.** Three independent guarantees:

1. The deterministic answer is composed *before* the LLM is called, and is only
   replaced if the call returns usable text. Any failure — network, quota, cold
   model — falls back to it and says so in the `mode` field.
2. Chunk vectors are embedded once at build time and cached in
   `models/rag_index.json`, so starting the server makes no API call. Only the
   query is embedded live, and if that fails the dense signal is simply dropped
   and lexical ranking stands.
3. The LLM only ever receives the grounded frame — verified database facts plus
   retrieved citations. It rephrases; it cannot introduce a supplier fact or
   invent a clause.

`tests/test_huggingface.py` verifies all of this against a local mock of the HF
router, so it runs with no key and no network.

Keys are read from the environment only and are never written to disk.

### A note on fusion weighting

The dense ranking is deliberately down-weighted (`DENSE_FUSION_WEIGHT`, default
0.5) against 1.0 for each lexical ranking. On a regulatory corpus the decisive
terms are exact surface forms — "510(k)", "consent decree" — so embeddings
should broaden recall without overriding an exact clause match. Equal-weight
fusion was measurably worse: it pushed the correct clause out of the top 3 on a
known-clause probe, which is what `test_lexical_precision_survives_an_uninformative_dense_signal`
now guards against.

---

## Honest limitations

Read `docs/LIMITATIONS.md` before presenting. The short version: 50 suppliers
with 18 positive labels is a small dataset, the confidence intervals on every
metric are wide, the data is synthetic so absolute metric values do not transfer
to real suppliers, and the label is a single historical binary event rather than
a rate over time. The methodology is the contribution; the numbers illustrate it.
