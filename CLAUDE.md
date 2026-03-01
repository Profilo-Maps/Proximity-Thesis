# Project Architecture & Rules
Proximity is a graph network built on OSM and government data that is being built as a 








**Main Data Pipeline Folder**: `@Notebooks/Karna/Proximity Model`

## Context Protocol
- **Source of Truth:** Do NOT use the superpowers:brainstorming or superpowers:writing-plans skills. Refer to hand authored spec instead. Always prioritize hand-typed specs in the /specs directory. 
-**Reference Code Usage:** Only use references to simplify workflow when possible. Avoid copying code from references.Do not assume references are logical or reliable. 
- **Workflow State:** Read `@specs/active_context.md` at the start of every session and after every `/compact`. It defines the active file, source spec, and references.

## Critical Constraints
- Do NOT generate new specs; I provide them. Update the readme for new features.
- Do NOT modify files outside the Active Focus in active_context.md without permission.
- Maintain strict type safety across all module boundaries.
