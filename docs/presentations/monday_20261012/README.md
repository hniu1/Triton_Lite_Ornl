# Editable Monday presentation

## Current deck: graph integrated into slide 8

Use **TRITON_Methods_Integrated_Halo_Graph.pptx**. Slide **8** now shows the
complete direct U-Net, halo, and optional graph branch in one editable diagram.
Purple arrows show how tile embeddings and message passing modify the bottleneck
conditioning. Its notes explain these connections in detail. Slides 12 and 22
retain the separate graph overview and message-passing walkthrough.

`architecture/Editable_Integrated_Halo_Graph.pptx` is the standalone editable
version. SVG, PNG, and PDF are `architecture/integrated_halo_graph_architecture.*`.
Regeneration script: `integrate_graph_slide8.py`.

## Latest deck: halo plus graph (8 October 2026)

Use **TRITON_Methods_with_Halo_and_Graph.pptx** for the latest 22-slide deck.
The editable halo architecture remains on slide **8**. Slide **12** now shows
the graph integration; slide **22** explains one message-passing layer in detail.
Both graph slides have detailed notes, implementation references, and evidence
limits. Other slides are preserved from the previous deck.

`architecture/Editable_Graph_Architecture.pptx` contains both graph diagrams as
large editable standalone slides with the same notes. PNG, SVG, and PDF versions
are `architecture/graph_architecture.*` and `architecture/message_passing_detail.*`.
Notes are also available in `architecture/GRAPH_SPEAKER_NOTES.md`.
`add_graph_architecture.py` regenerates these outputs using the shared native
shape renderer and a separate Matplotlib process for preview images.

## Updated architecture slide (8 October 2026)

**TRITON_Methods_with_Editable_Architecture.pptx** is the previous halo-only update.
Slide **8** contains the detailed direct-plus-halo architecture as **67 native
editable shapes**, with a detailed walkthrough in its PowerPoint Notes section.
All other slides and their notes are preserved from the original deck.

- `architecture/Editable_Architecture.pptx` provides the same editable diagram
  as a large standalone slide with the same notes.
- `architecture/DETAILED_SPEAKER_NOTES.md` provides the notes as plain text.
- `preview_with_editable_architecture.html` previews the updated slide and notes.
- `insert_editable_architecture.py` creates the updated copy from the original
  deck and `architecture/editable_diagram.json`; it does not overwrite the original.

The architecture JSON is exported by `draw_architecture.py`. The original
deck and its initial standalone preview remain available below.

- **TRITON_Methods_and_Experiment_History.pptx** — 21 slides, including 4 appendix slides; native editable text and diagrams, with speaker notes/source paths. Open in PowerPoint or LibreOffice Impress. The 26-entry history catalog includes proposed work and evaluation protocols; it is not a count of completed experiments.
- **preview.html** — standalone local browser preview with notes; print to PDF if wanted. The preview is a schematic rendering from the same layout data, not a PowerPoint renderer.
- **SLIDE_TEXT_AND_NOTES.md** — all slide wording and presenter notes.
- **CHATGPT_SOURCE_BRIEF.md** and **EXPERIMENT_HISTORY.md** — upload these two first if asking ChatGPT to redesign the deck. The brief includes a paste-ready prompt and evidence rules.
- **CHATGPT_SOURCE_BUNDLE.zip** — the brief, history, notes and full source documents with their paths and SHA256 manifest. Extract before uploading if your client does not accept ZIP archives.
- **contact_sheet.png** — compact visual review of all slides.
- **build_deck.py** — regeneration script; requires Python3.11+, python-pptx and Pillow. Initial temporary command: `PYTHONPATH=/tmp/triton-slides-deps /ccs/home/haoranniu/miniconda3/bin/python docs/presentations/monday_20261012/build_deck.py`.

Before Monday: update the dated rebuild snapshot on slide15 and replace slide16 with corrected-data evaluation. Old forecasting percentages and metric charts are intentionally omitted as evidence. The files do not launch training or modify rebuild jobs.

The source documents contain historical numeric claims; CHATGPT_SOURCE_BRIEF.md explicitly supersedes them. Details of the implemented halo are checked against stage2/model.py, rather than relying on the larger-input-halo shorthand in older plans.
