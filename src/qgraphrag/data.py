"""Download public benchmarks and build a shared document corpus with reproducible splits."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict
from pathlib import Path

from .io import file_sha256, fingerprint, read_json, write_json
from .schema import Document, load_corpus, nonempty

DATASETS = {
    "hotpotqa": ("hotpot_train_v1.1.json", "hotpot_dev_distractor_v1.json"),
    "2wikimultihopqa": ("train.json", "dev.json"),
    "musique": ("musique_ans_v1.0_train.jsonl", "musique_ans_v1.0_dev.jsonl"),
}


def _download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Q-GraphRAG/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as source, temporary.open("wb") as target:
            if "text/html" in source.headers.get("Content-Type", ""):
                raise ValueError(f"Download returned an HTML page: {url}")
            shutil.copyfileobj(source, target, length=1024 * 1024)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _extract_files(archive_path, names, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for name in names:
            matches = [
                p
                for p in archive.infolist()
                if Path(p.filename).name == name
                and not p.is_dir()
                and "__MACOSX" not in Path(p.filename).parts
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {name} in {archive_path}")
            target = directory / name
            with archive.open(matches[0]) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def download_sources(dataset, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = DATASETS[dataset]
    paths = [directory / name for name in names]
    if all(p.exists() for p in paths):
        return paths
    if dataset == "hotpotqa":
        try:
            for name, target in zip(names, paths):
                if not target.exists():
                    _download("http://curtis.ml.cmu.edu/datasets/hotpot/" + name, target)
        except (OSError, urllib.error.URLError, ValueError):
            try:
                from datasets import load_dataset
            except ImportError as error:
                raise RuntimeError("Install qgraphrag[data] or provide --source-dir") from error
            for split, target in zip(("train", "validation"), paths):
                if not target.exists():
                    rows = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
                    write_json(target, list(rows))
    else:
        archive_path = directory / (dataset + ".zip")
        if not archive_path.exists():
            if dataset == "2wikimultihopqa":
                _download(
                    "https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1",
                    archive_path,
                )
            else:
                try:
                    import gdown
                except ImportError as error:
                    raise RuntimeError("Install qgraphrag[data] or provide --source-dir") from error
                temporary = archive_path.with_suffix(".part")
                result = gdown.download(
                    id="1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h", output=str(temporary), quiet=False
                )
                if result is None:
                    raise RuntimeError(
                        "MuSiQue download failed; use --source-dir with the official data"
                    )
                temporary.replace(archive_path)
        extraction_names = list(names)
        if dataset == "2wikimultihopqa":
            with zipfile.ZipFile(archive_path) as archive:
                if any(Path(name).name == "id_aliases.json" for name in archive.namelist()):
                    extraction_names.append("id_aliases.json")
        _extract_files(archive_path, extraction_names, directory)
    return paths


def read_records(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8-sig") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    else:
        rows = read_json(path)
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected a nonempty list of examples in {path}")
    return rows


def question_id(row):
    key = row.get("id", row.get("_id"))
    if key is None or not str(key):
        raise ValueError("Missing question ID")
    return str(key)


def select_records(rows, count, seed, selected_ids=None):
    by_id = {question_id(row): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate question IDs in source split")
    if selected_ids is not None:
        if not selected_ids or len(set(selected_ids)) != len(selected_ids):
            raise ValueError("Split manifest must contain nonempty, unique IDs")
        missing = set(selected_ids) - by_id.keys()
        if missing:
            raise ValueError(f"Split IDs absent from source: {sorted(missing)[:3]}")
        return [by_id[key] for key in selected_ids]
    if count < 0 or count > len(rows):
        raise ValueError(f"Requested {count} questions from a split with {len(rows)} examples")
    candidates = sorted(by_id)
    ids = candidates if count == 0 else random.Random(seed).sample(candidates, count)
    return [by_id[key] for key in ids]


def _context(row, dataset):
    if dataset == "musique":
        paragraphs = row.get("paragraphs")
        if not isinstance(paragraphs, list) or not paragraphs:
            raise ValueError("MuSiQue requires paragraphs")
        entries, support = [], []
        for paragraph in paragraphs:
            title = nonempty(paragraph.get("title"), "paragraph title")
            text = nonempty(paragraph.get("paragraph_text"), "paragraph text")
            entries.append((str(paragraph["idx"]), title, text))
            if paragraph.get("is_supporting") is True:
                support.append(str(paragraph["idx"]))
        if row.get("answerable") is False:
            raise ValueError("Use MuSiQue-Ans examples with answerable questions")
    else:
        context = row.get("context")
        if isinstance(context, dict):
            context = list(zip(context["title"], context["sentences"], strict=True))
        if not isinstance(context, list) or not context:
            raise ValueError("Expected context paragraphs")
        entries = []
        for title, sentences in context:
            title = nonempty(title, "paragraph title")
            text = (
                " ".join(s.strip() for s in sentences) if isinstance(sentences, list) else sentences
            )
            entries.append((title, title, nonempty(text, "paragraph text")))
        facts = row.get("supporting_facts")
        if isinstance(facts, dict):
            facts = list(zip(facts["title"], facts["sent_id"], strict=True))
        if not isinstance(facts, list) or not facts:
            raise ValueError("Expected supporting facts")
        support = list(
            dict.fromkeys(
                str(item[0]) if isinstance(item, (list, tuple)) else str(item) for item in facts
            )
        )
    if not support or len({key for key, _, _ in entries}) != len(entries):
        raise ValueError("Missing support or ambiguous context identifiers")
    if not set(support) <= {key for key, _, _ in entries}:
        raise ValueError("Supporting document is absent from the source context")
    return entries, support


def load_answer_aliases(path):
    if not path:
        return {}
    aliases = {}
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = nonempty(row.get("Q_id"), "answer entity ID")
            values = row.get("aliases", []) + row.get("demonyms", [])
            if key in aliases or any(not isinstance(value, str) for value in values):
                raise ValueError("Invalid or duplicate answer-alias entry")
            aliases[key] = sorted(set(values))
    return aliases


def prepare_data(
    dataset,
    train_file,
    dev_file,
    output,
    train_size=2000,
    test_size=1000,
    seed=1024,
    split_file=None,
    corpus_file=None,
    alias_file=None,
):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    answer_aliases = load_answer_aliases(alias_file)
    specification = read_json(split_file) if split_file else None
    if specification and specification.get("dataset", dataset) != dataset:
        raise ValueError("Split manifest dataset mismatch")
    source_hashes = {"train": file_sha256(train_file), "dev": file_sha256(dev_file)}
    if corpus_file:
        source_hashes["corpus"] = file_sha256(corpus_file)
    if alias_file:
        source_hashes["aliases"] = file_sha256(alias_file)
    if specification and specification.get("source_sha256", source_hashes) != source_hashes:
        raise ValueError("Source files differ from the split manifest")
    train = select_records(
        read_records(train_file),
        train_size,
        seed,
        specification["train_ids"] if specification else None,
    )
    test = select_records(
        read_records(dev_file),
        test_size,
        seed + 1,
        specification["test_ids"] if specification else None,
    )
    if {question_id(row) for row in train} & {question_id(row) for row in test}:
        raise ValueError("Training and evaluation question IDs overlap")
    documents = load_corpus(corpus_file) if corpus_file else []
    by_content = {(d.title, d.text): d.id for d in documents}
    by_title = {}
    for doc in documents:
        by_title.setdefault(doc.title, []).append(doc.id)
    converted = []
    for rows in (train, test):
        result = []
        for row in rows:
            entries, support = _context(row, dataset)
            context_ids = {}
            for key, title, text in entries:
                identity = (title, text)
                doc_id = by_content.get(identity)
                if doc_id is None and corpus_file:
                    options = by_title.get(title, [])
                    if len(options) != 1:
                        raise ValueError(
                            f"Context document missing/ambiguous in supplied corpus: {title}"
                        )
                    doc_id = options[0]
                if doc_id is None:
                    doc_id = fingerprint([title, text])[:24]
                    documents.append(Document(doc_id, title, text))
                    by_content[identity] = doc_id
                context_ids[key] = doc_id
            answer = row.get("answer")
            if not isinstance(answer, (str, int, float)):
                raise ValueError(f"Missing reference answer: {question_id(row)}")
            aliases = row.get("answer_aliases", [])
            if not isinstance(aliases, list) or any(not isinstance(a, str) for a in aliases):
                raise ValueError("answer_aliases must be a string list")
            aliases = sorted(set(aliases + answer_aliases.get(row.get("answer_id"), [])))
            result.append(
                {
                    "id": question_id(row),
                    "question": nonempty(row.get("question"), "question"),
                    "answer": str(answer),
                    "answer_aliases": aliases,
                    "supporting_facts": list(dict.fromkeys(context_ids[key] for key in support)),
                }
            )
        converted.append(result)
    documents.sort(key=lambda doc: doc.id)
    corpus = [asdict(doc) for doc in documents]
    manifest = {
        "dataset": dataset,
        "seed": seed,
        "train_ids": [question_id(row) for row in train],
        "test_ids": [question_id(row) for row in test],
        "source_sha256": source_hashes,
        "corpus_mode": "provided" if corpus_file else "all_selected_contexts",
        "corpus_sha256": fingerprint(corpus),
        "documents": len(documents),
    }
    if (
        specification
        and specification.get("corpus_sha256", manifest["corpus_sha256"])
        != manifest["corpus_sha256"]
    ):
        raise ValueError("Prepared corpus differs from the split manifest")
    destination = Path(output)
    raw = destination / "raw"
    existing = destination / "split.json"
    if existing.exists() and read_json(existing) != manifest:
        raise ValueError("Output already contains a different split; choose a new output directory")
    write_json(raw / "dataset_corpus.json", corpus)
    write_json(raw / "train.json", converted[0])
    write_json(raw / "test.json", converted[1])
    write_json(existing, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare multi-hop QA data")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--output", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--download", action="store_true")
    source.add_argument("--source-dir", help="Directory containing the official train/dev files")
    source.add_argument(
        "--train-file", help="Official training JSON/JSONL file; also requires --dev-file"
    )
    parser.add_argument("--dev-file")
    parser.add_argument(
        "--train-size", type=int, default=2000, help="0 selects the entire training split"
    )
    parser.add_argument(
        "--test-size", type=int, default=1000, help="0 selects the entire development split"
    )
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--split-file", help="Existing split.json with exact train_ids/test_ids")
    parser.add_argument("--answer-aliases", help="Official 2Wiki id_aliases.json file")
    parser.add_argument(
        "--corpus", help="Use an existing shared corpus instead of collecting contexts"
    )
    args = parser.parse_args(argv)
    if args.download:
        train, dev = download_sources(args.dataset, Path(args.output) / "downloads")
    elif args.source_dir:
        train, dev = [Path(args.source_dir) / name for name in DATASETS[args.dataset]]
    else:
        if not args.dev_file:
            parser.error("--train-file requires --dev-file")
        train, dev = Path(args.train_file), Path(args.dev_file)
    alias_file = args.answer_aliases
    if args.dataset == "2wikimultihopqa" and not alias_file:
        candidate = train.parent / "id_aliases.json"
        alias_file = candidate if candidate.exists() else None
    manifest = prepare_data(
        args.dataset,
        train,
        dev,
        args.output,
        args.train_size,
        args.test_size,
        args.seed,
        args.split_file,
        args.corpus,
        alias_file,
    )
    print(
        json.dumps(
            {
                "train": len(manifest["train_ids"]),
                "test": len(manifest["test_ids"]),
                "documents": manifest["documents"],
                "split": str(Path(args.output) / "split.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
