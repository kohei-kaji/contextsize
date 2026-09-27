#!/usr/bin/env python3
"""Create the Stanza word-level coreference files used by stage 2.

Each non-empty input line is treated as one document.  The output deliberately
uses a compact eight-column layout: word text is column 2, UPOS is column 4,
and pipe-separated coreference chain IDs are column 8.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def coref_ids(word: object) -> str:
    chains = getattr(word, "coref_chains", None) or []
    ids = sorted({str(chain.chain.index) for chain in chains}, key=int)
    return "|".join(ids) if ids else "_"


def write_document(doc: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as output:
        for sentence in doc.sentences:
            for word in sentence.words:
                row = [
                    str(word.id),
                    word.text,
                    word.lemma or "_",
                    word.upos or "_",
                    word.xpos or "_",
                    str(word.head),
                    word.deprel or "_",
                    coref_ids(word),
                ]
                output.write("\t".join(row) + "\n")


def build_corpus(source: Path, output_dir: Path, corpus: str) -> int:
    # Import lazily so --help and unit tests do not initialize Stanza or Torch.
    import stanza

    nlp = stanza.Pipeline(
        "en", processors="tokenize,pos,lemma,depparse,coref", use_gpu=False
    )
    count = 0
    with source.open(encoding="utf-8") as input_file:
        for document_number, line in enumerate(input_file, start=1):
            text = line.strip()
            if not text:
                continue
            destination = output_dir / f"{corpus}_stanza_word_doc{document_number}.conllu"
            write_document(nlp(text), destination)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Text file with one document per line")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--corpus", required=True, help="Filename prefix, e.g. ns")
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use Stanza's default GPU detection (CPU is the reproducible default)",
    )
    args = parser.parse_args()

    if args.use_gpu:
        import stanza

        nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse,coref")
        count = 0
        with args.source.open(encoding="utf-8") as input_file:
            for document_number, line in enumerate(input_file, start=1):
                text = line.strip()
                if not text:
                    continue
                write_document(
                    nlp(text),
                    args.output_dir
                    / f"{args.corpus}_stanza_word_doc{document_number}.conllu",
                )
                count += 1
    else:
        count = build_corpus(args.source, args.output_dir, args.corpus)
    print(f"Wrote {count} Stanza-word documents to {args.output_dir}")


if __name__ == "__main__":
    main()
