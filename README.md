# Q-GraphRAG

Quality-aware graph retrieval-augmented generation for multi-hop question answering.

The pipeline constructs an evidence-grounded knowledge graph with edge-quality features,
trains a query-conditioned GNN with a shared quality encoder, selects a connected reasoning
subgraph with PCST, and integrates graph and text evidence for answer generation.

## Installation

Python 3.10–3.12 and PyTorch with the appropriate CUDA build are required for GPU training.

~~~bash
pip install -e ".[data]"
~~~

Serve Qwen3-14B-AWQ through an OpenAI-compatible chat endpoint. The default endpoint is
http://127.0.0.1:8000/v1. Model paths and the endpoint can be set through environment variables:

~~~bash
export QGRAPHRAG_LLM_BASE_URL=http://127.0.0.1:8000/v1
export QGRAPHRAG_LLM_MODEL=Qwen3-14B-AWQ
export QGRAPHRAG_EMBEDDING_MODEL=/path/to/all-mpnet-base-v2
export QGRAPHRAG_RERANKER_MODEL=/path/to/ms-marco-MiniLM-L-6-v2
~~~

For an authenticated endpoint, set QGRAPHRAG_LLM_API_KEY. Local model paths are optional;
otherwise the embedding and reranking models are loaded by their Hugging Face identifiers.

## Data

Download and prepare a benchmark:

~~~bash
python scripts/prepare_data.py --dataset hotpotqa --download --output data/hotpotqa
python scripts/prepare_data.py --dataset 2wikimultihopqa --download --output data/2wikimultihopqa
python scripts/prepare_data.py --dataset musique --download --output data/musique
~~~

Data sources: [HotpotQA](https://hotpotqa.github.io/),
[2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop), and
[MuSiQue](https://github.com/StonyBrookNLP/musique).
If a hosting service is unavailable, download the official files and replace --download
with --source-dir /path/to/files. The command also accepts --train-file and --dev-file.
For 2Wiki, id_aliases.json is detected beside train.json, or can be supplied with
--answer-aliases; its aliases and demonyms are retained for answer evaluation.

Preparation selects 4,000 training questions and 1,000 evaluation questions from the
official training and development splits, respectively, with seed 1024. Training questions
are divided into four groups of 1,000. Each group has its own corpus under
train_0/raw/ through train_3/raw/; evaluation has a separate corpus and test.json under
eval/raw/. Corpora retain all candidate passages, including distractors. split.json records
question IDs, graph assignments, and source/corpus hashes.

Use --train-size, --test-size, and --num-train-graphs for other sizes, or --split-file to
replay a saved split. A size of zero selects the entire source split. When changing the
graph count, update datasets.train_names in the dataset configuration. Use a new output
directory for a different split. --corpus can supply an external document collection from
which each group's candidate passages are resolved.
Raw data, model weights, and generated artifacts are excluded from the repository.

Custom corpora use a JSON list of objects with id, title, and text. QA files use a JSON
list with id, question, answer, and supporting_facts (a list of document IDs or
[document title, sentence index] pairs). Optional fields are answer_aliases and
question_entities; question_entities must contain mentions from the question.
Training labels come from entities in the supporting documents. Inference receives
only the question and optional question-side mentions.

## Run

The following commands use configs/hotpotqa.yaml. Replace the configuration path for
2WikiMultihopQA or MuSiQue. The dataset configurations specify four training graphs
and one evaluation graph. A single model and optimizer are shared across the training
graphs; each batch uses one graph. Prediction uses only the evaluation graph.

~~~bash
# Construct and encode the four training graphs and the evaluation graph.
qgraphrag --config configs/hotpotqa.yaml build-index
qgraphrag --config configs/hotpotqa.yaml prepare

# Graph-completion pretraining with unit propagation coefficients.
qgraphrag --config configs/hotpotqa.yaml --set paths.output=outputs/hotpotqa/pretrain pretrain

# Joint GNN and quality-encoder training with retrieval supervision.
qgraphrag --config configs/hotpotqa.yaml --set paths.output=outputs/hotpotqa/finetune finetune \
  --pretrained outputs/hotpotqa/pretrain/last.pt

# Retrieval, sufficiency assessment, optional text supplementation, and QA.
qgraphrag --config configs/hotpotqa.yaml --set paths.output=outputs/hotpotqa/predict predict \
  --checkpoint outputs/hotpotqa/finetune/last.pt --queries data/hotpotqa/eval/raw/test.json
~~~

For a single question, use ask --checkpoint /path/to/last.pt --question "Your question".
Training supports --resume with the same training graph collection. predict resumes
completed queries in the same output directory and writes final-document Recall@K, answer EM, and token F1 when labels are present.

A compatible GFM-RAG DistMult/sum export can initialize the backbone:

~~~bash
qgraphrag import-gfm --source /path/to/pretrained --output artifacts/gfm.pt
qgraphrag --config configs/hotpotqa.yaml finetune --pretrained artifacts/gfm.pt
~~~

The export must include its config.json. Quality-encoder training is required before QA.
Graphs, checkpoints, and cached propagation coefficients carry quality-feature versions;
quality-aware inference requires a matching jointly fine-tuned checkpoint.

## Configuration

The defaults in config.py are also listed in configs/default.yaml; dataset configurations
specify graph names and paths.
All settings can be changed with --set section.key=value. Training writes the effective
configuration alongside checkpoints. Defaults use Qwen3-14B-AWQ, all-mpnet-base-v2,
and ms-marco-MiniLM-L6-v2; the final evidence budget is five documents.

Text supplementation allows up to two rounds (retrieval.max_supplementation_rounds).
Each round merges and deduplicates documents, then reassesses sufficiency. The loop stops
when evidence is sufficient, the round limit is reached, or no new documents are added.
The PCST subgraph stays fixed throughout refinement.

For graph-builder comparisons, override llm.model and its endpoint only for build-index.
Use a separate paths.index and paths.output for each builder, and repeat pretraining and
fine-tuning. Keep the query-side and answering LLM settings unchanged.

## Tests

~~~bash
pip install -e ".[dev]"
pytest -q
~~~

## Acknowledgments

The GNN backbone is adapted from [GFM-RAG](https://github.com/RManLuo/gfm-rag) and
ULTRA. See NOTICE for attribution. This project is released under the Apache-2.0 license.
