"""Versioned graph records, including all retained source-instance provenance."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .io import fingerprint, read_json, write_json

SCHEMA_VERSION = 2
QUALITY_FEATURE_VERSION = 2


def normalize_label(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def normalized_triple(triple: Triple) -> Triple:
    return Triple(
        normalize_label(triple.head),
        normalize_label(triple.relation),
        normalize_label(triple.tail),
        tuple(sorted({normalize_label(q) for q in triple.qualifiers})),
    )


def triple_key(triple: Triple) -> tuple:
    value = normalized_triple(triple)
    return value.head, value.relation, value.tail, value.qualifiers


_SUPPORT_SCORES = {"full": 1.0, "partial": 0.5, "unsupported": 0.0, "contradicted": 0.0}
_RELATION_SCORES = {"complete": 1.0, "generalized": 0.5, "incorrect": 0.0}


def validate_assessment(record: dict, *, retained: bool = True) -> None:
    if record.get("support_status") not in _SUPPORT_SCORES:
        raise ValueError("Missing or invalid support_status")
    if record.get("relation_retention") not in _RELATION_SCORES:
        raise ValueError("Missing or invalid relation_retention")
    if type(record.get("entity_binding_verified")) is not bool:
        raise ValueError("Missing entity_binding_verified assessment")
    if type(record.get("unique_correction")) is not bool:
        raise ValueError("Missing unique_correction assessment")
    equivalent = record.get("equivalent")
    if (
        not isinstance(equivalent, list)
        or len(equivalent) != 3
        or any(type(x) is not bool for x in equivalent)
    ):
        raise ValueError("equivalent must contain three boolean semantic deltas")
    qualifiers = record.get("qualifier_assessments")
    if not isinstance(qualifiers, list):
        raise ValueError("Missing qualifier_assessments; use [] only for an assessed empty set")
    seen = set()
    for item in qualifiers:
        if not isinstance(item, dict):
            raise ValueError("Each qualifier assessment must be an object")
        text = normalize_label(nonempty(item.get("text"), "qualifier.text"))
        if text in seen:
            raise ValueError("Duplicate qualifier assessment")
        seen.add(text)
        if type(item.get("preserved")) is not bool:
            raise ValueError("Missing qualifier preserved flag")
    issues = record.get("residual_issues")
    if not isinstance(issues, list) or any(not isinstance(x, str) or not x.strip() for x in issues):
        raise ValueError("Missing or invalid residual_issues")
    if record["support_status"] == "partial" and not issues:
        raise ValueError("Partial support must record the remaining ambiguity")
    if retained:
        if (
            record["support_status"] in ("unsupported", "contradicted")
            or not record["entity_binding_verified"]
            or record["relation_retention"] == "incorrect"
        ):
            raise ValueError("Retained source has unsupported facts or unverified binding/relation")
        if record.get("decision") not in ("KEEP", "REVISE"):
            raise ValueError("Dropped source entered graph")
        if record["decision"] == "KEEP" and not all(equivalent):
            raise ValueError("KEEP source contains a semantic correction")
        if record["decision"] == "REVISE" and record["unique_correction"] is not True:
            raise ValueError("REVISE requires a unique explicit correction")


def assessment_values(record: dict, *, retained: bool = True) -> tuple[float, float, float]:
    validate_assessment(record, retained=retained)
    qualifiers = record["qualifier_assessments"]
    q = sum(item["preserved"] for item in qualifiers) / len(qualifiers) if qualifiers else 0.0
    fidelity = (
        int(record["entity_binding_verified"])
        + _RELATION_SCORES[record["relation_retention"]]
        + bool(qualifiers) * q
    ) / (2 + bool(qualifiers))
    return _SUPPORT_SCORES[record["support_status"]], sum(record["equivalent"]) / 3, fidelity


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    title: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Triple:
    head: str
    relation: str
    tail: str
    qualifiers: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict) -> Triple:
        qualifiers = value.get("qualifiers", [])
        if not isinstance(qualifiers, list) or any(
            not isinstance(x, str) or not x.strip() for x in qualifiers
        ):
            raise ValueError("qualifiers must be a list of nonempty strings")
        return cls(
            nonempty(value.get("head"), "head"),
            nonempty(value.get("relation"), "relation"),
            nonempty(value.get("tail"), "tail"),
            tuple(sorted(set(x.strip() for x in qualifiers), key=normalize_label)),
        )

    def as_dict(self) -> dict:
        result = asdict(self)
        result["qualifiers"] = list(self.qualifiers)
        return result


@dataclass
class Provenance:
    id: str
    document_id: str
    chunk_id: str
    chunk_start: int
    chunk_end: int
    evidence: str
    evidence_start: int
    evidence_end: int
    initial: dict
    final: dict
    decision: str
    reflection: dict = field(default_factory=dict)
    support: float | None = None
    equivalent: list[bool] | None = None
    assessment_fingerprint: str | None = None

    def assessment_digest(self) -> str:
        return fingerprint(
            {
                "quality_feature_version": QUALITY_FEATURE_VERSION,
                "id": self.id,
                "document_id": self.document_id,
                "chunk_id": self.chunk_id,
                "chunk_start": self.chunk_start,
                "chunk_end": self.chunk_end,
                "evidence": self.evidence,
                "evidence_start": self.evidence_start,
                "evidence_end": self.evidence_end,
                "initial": self.initial,
                "final": self.final,
                "decision": self.decision,
                "reflection": self.reflection,
            }
        )

    def validate(self) -> None:
        validate_assessment(self.reflection)
        if self.assessment_fingerprint != self.assessment_digest():
            raise ValueError(f"Reflection assessment binding mismatch: {self.id}")
        if self.decision != self.reflection.get("decision") or (
            Triple.from_dict(self.final) != Triple.from_dict(self.reflection.get("final", {}))
        ):
            raise ValueError("Reflection is not bound to the retained final triple")
        if (
            self.support != _SUPPORT_SCORES[self.reflection["support_status"]]
            or self.equivalent != self.reflection["equivalent"]
        ):
            raise ValueError("Source values disagree with the saved reflection")
        Triple.from_dict(self.initial)

    def values(self) -> tuple[float, float, float]:
        self.validate()
        return assessment_values(self.reflection)

    def evidence_key(self) -> tuple:
        return (
            self.document_id,
            self.evidence_start,
            self.evidence_end,
            triple_key(Triple.from_dict(self.final)),
        )


@dataclass
class Edge:
    id: str
    head: str
    relation: str
    tail: str
    qualifiers: list[str]
    sources: list[Provenance]
    features: list[float] = field(default_factory=list)

    def triple(self) -> dict:
        return {
            "head": self.head,
            "relation": self.relation,
            "tail": self.tail,
            "qualifiers": self.qualifiers,
        }


@dataclass
class GraphIndex:
    documents: list[Document]
    entities: list[str]
    edges: list[Edge]
    document_entities: dict[str, list[str]]
    aliases: dict[str, str] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    quality_feature_version: int = QUALITY_FEATURE_VERSION

    def content(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "quality_feature_version": self.quality_feature_version,
            "documents": [asdict(x) for x in self.documents],
            "entities": self.entities,
            "edges": [asdict(x) for x in self.edges],
            "document_entities": self.document_entities,
            "aliases": self.aliases,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.content())

    def validate(self) -> None:
        if self.quality_feature_version != QUALITY_FEATURE_VERSION:
            raise ValueError("Quality feature version mismatch; rebuild the index")
        entities = set(self.entities)
        documents = {d.id: d for d in self.documents}
        if len(entities) != len(self.entities) or len(documents) != len(self.documents):
            raise ValueError("Duplicate entity or document IDs")
        if len({e.id for e in self.edges}) != len(self.edges):
            raise ValueError("Duplicate edge IDs")
        edge_keys = set()
        for edge in self.edges:
            key = triple_key(Triple.from_dict(edge.triple()))
            if key in edge_keys:
                raise ValueError("Duplicate final edge")
            edge_keys.add(key)
            if edge.head not in entities or edge.tail not in entities or not edge.sources:
                raise ValueError(f"Invalid edge endpoints/provenance: {edge.id}")
            if len(edge.features) != 3 or any(not 0 <= x <= 1 for x in edge.features):
                raise ValueError(f"Invalid [F,A,I]: {edge.id}")
            values, seen = [], set()
            for source in edge.sources:
                if source.document_id not in documents:
                    raise ValueError("Unknown source document")
                text = documents[source.document_id].text
                if (
                    not 0
                    <= source.chunk_start
                    <= source.evidence_start
                    < source.evidence_end
                    <= source.chunk_end
                    <= len(text)
                ):
                    raise ValueError(f"Invalid evidence offsets: {source.id}")
                if text[source.evidence_start : source.evidence_end] != source.evidence:
                    raise ValueError(f"Evidence is not locatable: {source.id}")
                values.append(source.values())
                if triple_key(Triple.from_dict(source.final)) != key:
                    raise ValueError("Final edge differs from its assessed representation")
                evidence_key = source.evidence_key()
                if evidence_key in seen:
                    raise ValueError("Duplicate source evidence")
                seen.add(evidence_key)
            expected = [
                max(v[0] for v in values),
                sum(v[1] for v in values) / len(values),
                sum(v[2] for v in values) / len(values),
            ]
            if any(not math.isclose(a, b, abs_tol=1e-8) for a, b in zip(edge.features, expected)):
                raise ValueError(f"Features disagree with source assessments: {edge.id}")
        if set(self.document_entities) != set(documents):
            raise ValueError("Entity-document index does not cover the corpus")
        for values in self.document_entities.values():
            if not set(values) <= entities:
                raise ValueError("Document references an unknown entity")

    def save(self, directory: str | Path) -> None:
        self.validate()
        directory = Path(directory)
        write_json(
            directory / "graph.json",
            {**self.content(), "metadata": self.metadata, "fingerprint": self.fingerprint},
        )
        entity_documents: dict[str, list[str]] = {e: [] for e in self.entities}
        for document, entities in self.document_entities.items():
            for entity in entities:
                entity_documents[entity].append(document)
        write_json(directory / "entity_documents.json", entity_documents)
        write_json(
            directory / "triple_sources.json",
            {e.id: [asdict(p) for p in e.sources] for e in self.edges},
        )

    @classmethod
    def load(cls, directory: str | Path) -> GraphIndex:
        data = read_json(Path(directory) / "graph.json")
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported graph schema; rebuild the index from source documents")
        if data.get("quality_feature_version") != QUALITY_FEATURE_VERSION:
            raise ValueError("Quality feature version mismatch; rebuild the index")
        edges = [
            Edge(**{**e, "sources": [Provenance(**p) for p in e["sources"]]}) for e in data["edges"]
        ]
        graph = cls(
            [Document(**d) for d in data["documents"]],
            data["entities"],
            edges,
            data["document_entities"],
            data.get("aliases", {}),
            data.get("metadata", {}),
            data["quality_feature_version"],
        )
        graph.validate()
        if graph.fingerprint != data["fingerprint"]:
            raise ValueError("Graph content fingerprint mismatch")
        return graph


def load_corpus(path: str | Path) -> list[Document]:
    data = read_json(path)
    if isinstance(data, dict):
        records = [Document(str(k), str(k), nonempty(v, "document text")) for k, v in data.items()]
    elif isinstance(data, list):
        records = [
            Document(
                str(x.get("id", x.get("title", ""))),
                nonempty(x.get("title", x.get("id")), "document title"),
                nonempty(x.get("text"), "document text"),
            )
            for x in data
        ]
    else:
        raise ValueError("Corpus must be a title-to-text mapping or document list")
    if (
        not records
        or any(not d.id for d in records)
        or len({d.id for d in records}) != len(records)
    ):
        raise ValueError("Corpus must contain unique, nonempty document IDs")
    # Do not strip source text: saved character offsets refer to the original corpus.
    if isinstance(data, dict):
        records = [Document(d.id, d.title, data[d.id]) for d in records]
    else:
        records = [Document(d.id, d.title, x["text"]) for d, x in zip(records, data)]
    return records


def split_chunks(document: Document, max_chars: int, overlap: int) -> list[Chunk]:
    if max_chars <= 0 or not 0 <= overlap < max_chars:
        raise ValueError("Invalid chunk size/overlap")
    chunks, start = [], 0
    while start < len(document.text):
        end = min(start + max_chars, len(document.text))
        if end < len(document.text):
            breaks = list(re.finditer(r"[.!?\n]\s+", document.text[start + max_chars // 2 : end]))
            if breaks:
                candidate = start + max_chars // 2 + breaks[-1].end()
                if candidate > start + overlap:
                    end = candidate
        chunks.append(
            Chunk(
                fingerprint([document.id, start, end])[:24],
                document.id,
                document.title,
                document.text[start:end],
                start,
                end,
            )
        )
        if end == len(document.text):
            break
        start = max(start + 1, end - overlap)
    return chunks
