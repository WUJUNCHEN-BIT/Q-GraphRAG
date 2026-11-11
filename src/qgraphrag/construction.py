"""Stage 1: reflect, persist evidence assessments, merge, and aggregate quality features."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from .io import fingerprint, read_json, write_json
from .llm import JsonBackend
from .schema import (
    QUALITY_FEATURE_VERSION,
    Chunk,
    Edge,
    GraphIndex,
    Provenance,
    Triple,
    load_corpus,
    nonempty,
    normalize_label,
    normalized_triple,
    split_chunks,
    triple_key,
    validate_assessment,
)


def locate_evidence(
    chunk: Chunk, evidence: str, start=None, end=None
) -> tuple[int, int, str] | None:
    """Return original-document offsets, including when only whitespace/Unicode differ."""
    if not isinstance(evidence, str) or not evidence.strip():
        return None
    if (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and 0 <= start < end <= len(chunk.text)
        and normalize_label(chunk.text[start:end]) == normalize_label(evidence)
    ):
        return chunk.start + start, chunk.start + end, chunk.text[start:end]
    at = chunk.text.find(evidence)
    if at >= 0:
        return chunk.start + at, chunk.start + at + len(evidence), evidence
    # Every normalized character retains a link to its original character position.
    chars, positions = [], []
    for index, char in enumerate(chunk.text):
        for normalized in normalize_label(char) if not char.isspace() else " ":
            if normalized == " " and (not chars or chars[-1] == " "):
                continue
            chars.append(normalized)
            positions.append(index)
    needle = normalize_label(evidence)
    at = "".join(chars).find(needle)
    if at < 0:
        return None
    left, right = positions[at], positions[at + len(needle) - 1] + 1
    return chunk.start + left, chunk.start + right, chunk.text[left:right]


class Canonicalizer:
    def __init__(self, aliases: dict | None = None):
        aliases = aliases or {}
        self.entity_aliases = {
            normalize_label(k): normalize_label(v) for k, v in aliases.get("entities", {}).items()
        }
        self.relation_aliases = {
            normalize_label(k): normalize_label(v) for k, v in aliases.get("relations", {}).items()
        }
        for mapping in (self.entity_aliases, self.relation_aliases):
            for key in mapping:
                self.resolve(key, mapping)

    @staticmethod
    def resolve(value: str, mapping: dict) -> str:
        visited = set()
        while value in mapping and mapping[value] != value:
            if value in visited:
                raise ValueError("Cycle in canonical alias map")
            visited.add(value)
            value = mapping[value]
        return value

    def triple(self, triple: Triple) -> Triple:
        return Triple(
            self.resolve(normalize_label(triple.head), self.entity_aliases),
            self.resolve(normalize_label(triple.relation), self.relation_aliases),
            self.resolve(normalize_label(triple.tail), self.entity_aliases),
            tuple(sorted(set(normalize_label(q) for q in triple.qualifiers))),
        )


def _validate_extraction(result: dict) -> None:
    if not isinstance(result.get("triples"), list):
        raise ValueError("extract.triples must be a list")
    seen = set()
    for candidate in result["triples"]:
        key = nonempty(candidate.get("id"), "candidate.id")
        if key in seen:
            raise ValueError("Duplicate candidate ID")
        seen.add(key)
        Triple.from_dict(candidate)


def _validate_reflection(
    result: dict, candidates: list[dict], canonicalizer: Canonicalizer
) -> None:
    decisions = result.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != len(candidates):
        raise ValueError("Reflection must decide every candidate exactly once")
    if {d.get("id") for d in decisions} != {c["id"] for c in candidates}:
        raise ValueError("Reflection candidate IDs mismatch")
    for decision in decisions:
        if decision.get("decision") not in ("KEEP", "REVISE", "DROP"):
            raise ValueError("Unknown reflection decision")
        if decision["decision"] == "DROP":
            nonempty(decision.get("reason"), "DROP reason")
            continue
        final = Triple.from_dict(decision["final"])
        if normalized_triple(final) != canonicalizer.triple(final):
            raise ValueError("Reflection final must already use the assessed canonical aliases")
        nonempty(decision.get("evidence"), "reflection.evidence")
        validate_assessment(decision, retained=False)


def reflect_chunk(
    chunk: Chunk, backend: JsonBackend, canonicalizer: Canonicalizer | None = None
) -> tuple[list[Provenance], list[dict]]:
    canonicalizer = canonicalizer or Canonicalizer()
    payload = {"document_id": chunk.document_id, "title": chunk.title, "text": chunk.text}
    extracted = backend.complete("extract", payload, _validate_extraction)
    candidates = extracted["triples"]
    if not candidates:
        return [], []
    result = backend.complete(
        "reflect",
        {
            "chunk": payload,
            "candidates": candidates,
            "canonical_aliases": {
                "entities": canonicalizer.entity_aliases,
                "relations": canonicalizer.relation_aliases,
            },
            "quality_feature_version": QUALITY_FEATURE_VERSION,
        },
        lambda value: _validate_reflection(value, candidates, canonicalizer),
    )
    by_id = {c["id"]: c for c in candidates}
    retained, dropped = [], []
    for decision in result["decisions"]:
        initial = Triple.from_dict(by_id[decision["id"]])
        source_id = fingerprint([chunk.id, decision["id"]])[:24]
        reason = None
        if decision["decision"] == "DROP":
            reason = decision["reason"]
        elif decision["decision"] == "REVISE" and not decision["unique_correction"]:
            reason = "No unique explicit correction"
        elif (
            not decision["entity_binding_verified"] or decision["relation_retention"] == "incorrect"
        ):
            reason = "Unestablished entity binding or core relation"
        elif decision["support_status"] in ("unsupported", "contradicted"):
            reason = "Unsupported final proposition"
        elif decision["decision"] == "KEEP" and not all(decision["equivalent"]):
            reason = "KEEP contains a semantic correction"
        location = (
            None
            if reason
            else locate_evidence(
                chunk, decision["evidence"], decision.get("start"), decision.get("end")
            )
        )
        if reason or location is None:
            dropped.append(
                {
                    "source_id": source_id,
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.id,
                    "initial": initial.as_dict(),
                    "reason": reason or "Evidence span not found in source",
                    "reflection": decision,
                }
            )
            continue
        start, end, evidence = location
        final = Triple.from_dict(decision["final"])
        support = {"full": 1.0, "partial": 0.5}[decision["support_status"]]
        source = Provenance(
            source_id,
            chunk.document_id,
            chunk.id,
            chunk.start,
            chunk.end,
            evidence,
            start,
            end,
            initial.as_dict(),
            final.as_dict(),
            decision["decision"],
            decision,
            support,
            list(decision["equivalent"]),
        )
        source.assessment_fingerprint = source.assessment_digest()
        source.validate()
        retained.append(source)
    return retained, dropped


def merge_sources(
    sources: list[Provenance], canonicalizer: Canonicalizer, duplicates: list[dict] | None = None
) -> list[Edge]:
    groups: dict[tuple, Edge] = {}
    seen: dict[tuple, Provenance] = {}
    ordered = sorted(
        sources,
        key=lambda p: (
            p.document_id,
            p.evidence_start,
            p.evidence_end,
            p.chunk_start,
            p.chunk_end,
            p.chunk_id,
            p.id,
            fingerprint(asdict(p)),
        ),
    )
    for source in ordered:
        source.validate()
        final = normalized_triple(Triple.from_dict(source.final))
        if canonicalizer.triple(final) != final:
            raise ValueError("Canonicalization would change a previously assessed final triple")
        key = triple_key(final)
        duplicate = source.evidence_key()
        if duplicate in seen:
            selected = seen[duplicate]
            if duplicates is not None:
                duplicates.append(
                    {
                        "selected": asdict(selected),
                        "duplicate": asdict(source),
                        "assessment_disagrees": selected.reflection != source.reflection,
                    }
                )
            continue
        seen[duplicate] = source
        if key not in groups:
            groups[key] = Edge(
                fingerprint(list(key))[:24],
                final.head,
                final.relation,
                final.tail,
                list(final.qualifiers),
                [],
            )
        groups[key].sources.append(source)
    return sorted(groups.values(), key=lambda edge: edge.id)


def assess_edge(edge: Edge) -> Edge:
    if not edge.sources:
        raise ValueError("Cannot aggregate quality without retained sources")
    key = triple_key(Triple.from_dict(edge.triple()))
    seen, values = set(), []
    for source in edge.sources:
        if triple_key(Triple.from_dict(source.final)) != key:
            raise ValueError("Final edge differs from its assessed representation")
        evidence_key = source.evidence_key()
        if evidence_key in seen:
            raise ValueError("Source evidence must be deduplicated before aggregation")
        seen.add(evidence_key)
        values.append(source.values())
    edge.features = [
        max(v[0] for v in values),
        sum(v[1] for v in values) / len(values),
        sum(v[2] for v in values) / len(values),
    ]
    return edge


def build_index(config: dict, backend: JsonBackend) -> GraphIndex:
    documents = load_corpus(config["paths"]["corpus"])
    settings = config["construction"]
    chunks = [
        chunk
        for doc in documents
        for chunk in split_chunks(doc, settings["chunk_chars"], settings["overlap_chars"])
    ]
    aliases = read_json(settings["aliases"]) if settings["aliases"] else {}
    canonicalizer = Canonicalizer(aliases)
    retained, dropped = [], []
    with ThreadPoolExecutor(max_workers=settings["workers"]) as pool:
        results = pool.map(lambda c: reflect_chunk(c, backend, canonicalizer), chunks)
        for sources, rejected in tqdm(results, total=len(chunks), desc="Extract/reflect"):
            retained.extend(sources)
            dropped.extend(rejected)
    write_json(
        Path(config["paths"]["index"]) / "reflection_records.json",
        {
            "quality_feature_version": QUALITY_FEATURE_VERSION,
            "sources": [asdict(p) for p in retained],
            "dropped": dropped,
        },
    )
    duplicates = []
    edges = merge_sources(retained, canonicalizer, duplicates)
    assessed = [assess_edge(edge) for edge in edges]
    doc_entities = {d.id: set() for d in documents}
    for edge in assessed:
        for source in edge.sources:
            doc_entities[source.document_id].update((edge.head, edge.tail))
    entities = sorted({v for e in assessed for v in (e.head, e.tail)})
    graph = GraphIndex(
        documents,
        entities,
        assessed,
        {key: sorted(value) for key, value in doc_entities.items()},
        {
            k: canonicalizer.resolve(v, canonicalizer.entity_aliases)
            for k, v in canonicalizer.entity_aliases.items()
        },
        {
            "construction": settings,
            "builder_model": config["llm"]["model"],
            "corpus_fingerprint": fingerprint([asdict(d) for d in documents]),
        },
    )
    graph.save(config["paths"]["index"])
    write_json(
        Path(config["paths"]["index"]) / "construction_audit.json",
        {
            "documents": len(documents),
            "chunks": len(chunks),
            "entities": len(entities),
            "edges": len(assessed),
            "dropped": dropped,
            "duplicates": duplicates,
            "quality_feature_version": QUALITY_FEATURE_VERSION,
        },
    )
    return graph
