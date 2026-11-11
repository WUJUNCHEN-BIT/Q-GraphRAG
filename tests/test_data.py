import copy
import zipfile

import pytest

from qgraphrag.data import _context, _extract_files, prepare_data
from qgraphrag.io import read_json, write_json
from qgraphrag.schema import load_corpus


def records(dataset, prefix):
    if dataset == "musique":
        return [
            {
                "id": prefix + "-1",
                "question": "Where was Ada born?",
                "answer": "Riverford",
                "answer_aliases": ["River Ford"],
                "answerable": True,
                "paragraphs": [
                    {
                        "idx": 0,
                        "title": "Ada",
                        "paragraph_text": "Ada was born in Riverford.",
                        "is_supporting": True,
                    },
                    {
                        "idx": 1,
                        "title": "Distractor",
                        "paragraph_text": prefix + " context.",
                        "is_supporting": False,
                    },
                ],
            }
        ]
    return [
        {
            "_id": prefix + "-1",
            "question": "Where was Ada born?",
            "answer": "Riverford",
            "context": [
                ["Ada", ["Ada was born in Riverford."]],
                ["Distractor", [prefix + " context."]],
            ],
            "supporting_facts": [["Ada", 0]],
        }
    ]


@pytest.mark.parametrize("dataset", ["hotpotqa", "2wikimultihopqa", "musique"])
def test_data_conversion_retains_distractors_and_replays_exact_splits(dataset, tmp_path):
    train, dev = tmp_path / "train.json", tmp_path / "dev.json"
    write_json(train, records(dataset, "train"))
    write_json(dev, records(dataset, "test"))
    first = tmp_path / "first"
    manifest = prepare_data(dataset, train, dev, first, train_size=0, test_size=0)
    docs = load_corpus(first / "raw/dataset_corpus.json")
    assert len(docs) == 3
    assert sum(d.title == "Distractor" for d in docs) == 2
    qa = read_json(first / "raw/train.json")[0]
    assert len(qa["supporting_facts"]) == 1
    assert next(d.title for d in docs if d.id == qa["supporting_facts"][0]) == "Ada"
    second = tmp_path / "second"
    replay = prepare_data(dataset, train, dev, second, seed=999, split_file=first / "split.json")
    assert replay["train_ids"] == manifest["train_ids"]
    assert replay["corpus_sha256"] == manifest["corpus_sha256"]
    assert read_json(second / "raw/train.json") == [qa]
    if dataset == "musique":
        assert qa["answer_aliases"] == ["River Ford"]
    changed = records(dataset, "test")
    changed[0]["answer"] = "Changed"
    write_json(dev, changed)
    with pytest.raises(ValueError, match="Source files differ"):
        prepare_data(dataset, train, dev, tmp_path / "changed", split_file=first / "split.json")


def test_hotpot_huggingface_mirror_has_same_context_semantics():
    original = records("hotpotqa", "train")[0]
    mirrored = copy.deepcopy(original)
    mirrored["context"] = {
        "title": [item[0] for item in original["context"]],
        "sentences": [item[1] for item in original["context"]],
    }
    mirrored["supporting_facts"] = {"title": ["Ada"], "sent_id": [0]}
    assert _context(mirrored, "hotpotqa") == _context(original, "hotpotqa")


def test_fixed_corpus_is_preserved_and_split_cannot_be_overwritten(tmp_path):
    train, dev = tmp_path / "train.json", tmp_path / "dev.json"
    write_json(train, records("hotpotqa", "train"))
    write_json(dev, records("hotpotqa", "test"))
    corpus = tmp_path / "corpus.json"
    write_json(
        corpus,
        [
            {"id": "ada", "title": "Ada", "text": "Ada was born in Riverford."},
            {"id": "noise", "title": "Distractor", "text": "Full fixed article."},
            {"id": "other", "title": "Unrelated", "text": "Background article."},
        ],
    )
    out = tmp_path / "out"
    prepare_data("hotpotqa", train, dev, out, 0, 0, corpus_file=corpus)
    assert read_json(out / "raw/dataset_corpus.json") == read_json(corpus)
    assert read_json(out / "raw/train.json")[0]["supporting_facts"] == ["ada"]
    with pytest.raises(ValueError, match="different split"):
        prepare_data("hotpotqa", train, dev, out, 0, 0, seed=13, corpus_file=corpus)


def test_dataset_zip_extracts_only_expected_files(tmp_path):
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("data/train.json", "[]")
        output.writestr("../../unexpected.txt", "ignored")
    _extract_files(archive, ["train.json"], tmp_path / "files")
    assert (tmp_path / "files/train.json").read_text() == "[]"
    assert sorted(p.name for p in (tmp_path / "files").iterdir()) == ["train.json"]


def test_ambiguous_or_unanswerable_context_fails():
    row = records("musique", "train")[0]
    row["answerable"] = False
    with pytest.raises(ValueError, match="answerable"):
        _context(row, "musique")


def test_twowiki_entity_aliases_and_demonyms_are_available_to_evaluation(tmp_path):
    train, dev = tmp_path / "train.json", tmp_path / "dev.json"
    for file, prefix in [(train, "train"), (dev, "test")]:
        rows = records("2wikimultihopqa", prefix)
        rows[0]["answer_id"] = "Q123"
        write_json(file, rows)
    aliases = tmp_path / "id_aliases.json"
    aliases.write_text('{"Q_id":"Q123","aliases":["River Ford"],"demonyms":["Riverfordian"]}\n')
    out = tmp_path / "out"
    manifest = prepare_data("2wikimultihopqa", train, dev, out, 0, 0, alias_file=aliases)
    assert read_json(out / "raw/train.json")[0]["answer_aliases"] == ["River Ford", "Riverfordian"]
    assert "aliases" in manifest["source_sha256"]
