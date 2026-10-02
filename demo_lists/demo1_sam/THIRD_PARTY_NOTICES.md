# Third-party sources

The bundled SAM and SAM2 upstream source files are unmodified copies from Meta's official repositories, pinned to the following revisions:

- facebookresearch/segment-anything: dca509fe793f601edb92606367a655c15ac00fdf
- facebookresearch/sam2: 2b90b9f5ceec907a1c18123530e92e794ad901a4

Copyright notices remain in the source files. Each upstream project uses the Apache License, Version 2.0; the complete license text is included at:

- sam_shared_decoder/official_cpu_validation/official_sam/LICENSE
- sam_shared_decoder/official_cpu_validation/sam2_extension/LICENSE

The duplicate source snapshots contain the same unchanged bytes as their corresponding runnable upstream modules. The local comment-only `__init__.py` files are package scaffolding, not upstream module modifications. The validation scripts, execution baselines, factor candidate and benchmark are reproduction code, separate from the upstream sources.

Immutable public source URLs, revisions and SHA256 digests are recorded in sam_shared_decoder/source_audit/source_provenance.json.
