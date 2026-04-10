#!/usr/bin/env python3
"""Brain-integrated Graphify CLI.

Usage:
    graphify_cli.py --repos ~/Desktop/karen ~/Desktop/brain \
                    --output-dir /path/to/brain/raw/graphify/ \
                    --incremental \
                    [--semantic] \
                    [--anthropic-key $KEY]
"""
import argparse
import json
import sys
from pathlib import Path

from graphify.extract import extract
from graphify.build import build_from_json
from graphify.build import build
from graphify.analyze import god_nodes, surprising_connections, suggest_questions
from graphify.report import generate as generate_report
from graphify.cluster import cluster, score_all
from graphify.detect import detect, detect_incremental
from graphify.cache import check_semantic_cache
from graphify.export import to_json


def _file_hub_nodes(G):
    """Yield (node_id, data) for file-level hub nodes (label == source filename)."""
    for node_id, data in G.nodes(data=True):
        source_file = data.get("source_file", "")
        if not source_file:
            continue
        label = data.get("label", "")
        if label == Path(source_file).name:
            yield node_id, data


def _edge_relation(G, u, v):
    """Get edge relation safely, handling missing edges."""
    try:
        return G.edges[u, v].get("relation", "")
    except KeyError:
        return ""


def main():
    parser = argparse.ArgumentParser(description="Brain-integrated Graphify CLI")
    parser.add_argument("--repos", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument("--anthropic-key", type=str, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set API key for potential future semantic inference
    if args.anthropic_key:
        import os
        os.environ["ANTHROPIC_API_KEY"] = args.anthropic_key

    for repo_path in args.repos:
        repo_name = repo_path.name
        repo_out = args.output_dir / repo_name
        repo_out.mkdir(parents=True, exist_ok=True)

        # 1. Detect files
        try:
            if args.incremental:
                detection = detect_incremental(repo_path)
            else:
                detection = detect(repo_path)
        except Exception as exc:
            print(f"[graphify] {repo_name}: detect failed: {exc}", file=sys.stderr)
            continue

        # For incremental mode, use only new/modified files; for full mode, use all files
        files_key = "new_files" if args.incremental and "new_files" in detection else "files"
        code_files = detection[files_key].get("code", [])
        if not code_files:
            print(f"[graphify] {repo_name}: no code files found, skipping")
            continue

        # 2. AST extraction (deterministic, cached per-file SHA256)
        paths = [Path(f) for f in code_files]
        try:
            ast_extraction = extract(paths)
        except Exception as exc:
            print(f"[graphify] {repo_name}: extract failed: {exc}", file=sys.stderr)
            continue

        # 3. Collect extractions to merge
        extractions = [ast_extraction]

        # 4. Optional: semantic extraction (v1: cached results only)
        if args.semantic:
            doc_files = detection[files_key].get("document", [])
            if doc_files:
                cached_nodes, cached_edges, cached_hyper, uncached = \
                    check_semantic_cache(doc_files, root=repo_path)
                if uncached:
                    print(
                        f"[graphify] {repo_name}: {len(uncached)} uncached doc files "
                        f"skipped (semantic inference deferred to v2)",
                        file=sys.stderr,
                    )
                if cached_nodes:
                    semantic_extraction = {
                        "nodes": cached_nodes,
                        "edges": cached_edges,
                        "input_tokens": 0,
                        "output_tokens": 0,
                    }
                    if cached_hyper:
                        semantic_extraction["hyperedges"] = cached_hyper
                    extractions.append(semantic_extraction)

        # 5. Build graph
        if len(extractions) == 1:
            G = build_from_json(extractions[0])
        else:
            G = build(extractions)

        if G.number_of_nodes() == 0:
            print(f"[graphify] {repo_name}: empty graph after extraction, skipping")
            continue

        # 6. Community detection
        communities = cluster(G)
        cohesion = score_all(G, communities)

        # Placeholder community labels
        community_labels = {cid: f"Community {cid}" for cid in communities}

        # 7. Analysis
        gods = god_nodes(G)
        surprises = surprising_connections(G, communities)
        questions = suggest_questions(G, communities, community_labels)

        # 8. Report -> markdown
        report_md = generate_report(
            G, communities, cohesion, community_labels,
            gods, surprises, detection,
            {"input": 0, "output": 0},
            root=str(repo_path),
            suggested_questions=questions,
        )

        report_path = repo_out / f"{repo_name}-architecture.md"
        report_path.write_text(report_md, encoding="utf-8")

        # 9. Graph JSON
        graph_json_path = repo_out / f"{repo_name}-graph.json"
        to_json(G, communities, str(graph_json_path))

        # 10. File-level summaries for LanceDB embedding
        summaries_dir = repo_out / "file-summaries"
        summaries_dir.mkdir(exist_ok=True)

        node_community = {}
        for cid, nodes in communities.items():
            for n in nodes:
                node_community[n] = cid

        for node_id, node_data in _file_hub_nodes(G):
            source_file = node_data.get("source_file", "")
            label = node_data.get("label", node_id)
            community = node_community.get(node_id, -1)

            # Note: G is undirected, so neighbors include both in/out edges.
            # This means imports/calls lists may include reverse relationships.
            # Acceptable for v1 file summaries — directional accuracy is a v2 concern.
            neighbors = list(G.neighbors(node_id))
            imports = [
                G.nodes[n].get("label", n) for n in neighbors
                if _edge_relation(G, node_id, n) in ("imports", "imports_from")
            ]
            contains = [
                G.nodes[n].get("label", n) for n in neighbors
                if _edge_relation(G, node_id, n) == "contains"
            ]
            calls_out = [
                G.nodes[n].get("label", n) for n in neighbors
                if _edge_relation(G, node_id, n) == "calls"
            ]

            imports_md = "\n".join(f"- `{i}`" for i in imports) or "- (none)"
            contains_md = "\n".join(f"- `{c}`" for c in contains) or "- (none)"
            calls_md = "\n".join(f"- `{c}`" for c in calls_out) or "- (none)"

            summary = (
                f"---\n"
                f'title: "{label}"\n'
                f'source_file: "{source_file}"\n'
                f'repo: "{repo_name}"\n'
                f"community: {community}\n"
                f"file_type: code\n"
                f"author: ai\n"
                f"tags: [code-architecture, {repo_name}]\n"
                f"---\n"
                f"\n"
                f"# {label}\n"
                f"\n"
                f"**Repository:** {repo_name}\n"
                f"**File:** `{source_file}`\n"
                f"**Community:** {community}\n"
                f"\n"
                f"## Imports\n"
                f"{imports_md}\n"
                f"\n"
                f"## Contains\n"
                f"{contains_md}\n"
                f"\n"
                f"## Calls\n"
                f"{calls_md}\n"
            )

            try:
                rel_path = str(Path(source_file).relative_to(repo_path))
            except ValueError:
                rel_path = Path(source_file).name
            slug = f"{repo_name}_{rel_path.replace('/', '_').replace('.', '_')}"
            (summaries_dir / f"{slug}.md").write_text(summary, encoding="utf-8")

        node_count = G.number_of_nodes()
        edge_count = G.number_of_edges()
        community_count = len(communities)
        print(
            f"[graphify] {repo_name}: {node_count} nodes, {edge_count} edges, "
            f"{community_count} communities -> {repo_out}"
        )

    print("[graphify] Done.")


if __name__ == "__main__":
    main()
