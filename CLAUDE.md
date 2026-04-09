# Project Architecture & Rules
You are a senior developer helping me implement this feature of a larger project. Proximity is a graph network exported as a parquet built on OSM and government data. It will be used for network analysis and data crowdsourcing by other applications. 

**Path to Python:** `C:/Dev/Proximity/.venv/Scripts/python.exe`
**Run commands with:** `uv run python` (from project root)

## Planning: Uncertainty Protocol
ALWAYS ask questions before writing code. These questions should be used to highlight design decisions that are ambiguously justified by logic or that are matters of taste.When recommending approaches, order them by complexity of implementation. DO NOT use the superpowers:writing-plans skill, start implementing after a design is approved. After a feature is implemented, check the /docs folder for intermediate implementation docs and consolidate them to the readme.md.

## Debugging: Correction History Protocol
WHENEVER THE USER ASKS YOU TO DIAGNOSE AN ISSUE AT AN INTERSECTION, create a script in `/notesforrobot/<subfolder>/` — existing feature → existing folder; new feature → new folder added here: `sidewalk_matching/`, `curb_ramp_assignment/`, `crosswalk_geometry/`, `snapping_endpoint_alignment/`, `roundabout_handling/`, `street_geometry/`.

Update the script's `ANNOTATION` docstring **as the investigation progresses**, not retrospectively. Sections: error investigated · informed change · schema context · key prompts (quote the exact phrase that reoriented reasoning and explain why) · crosswalk design impact if intersection-related. Update the subfolder `INDEX.md` when adding a script.

## Post-Implementation/Debugging: Optimization Protocol
After implementing a new feature or sucessfully debugging, always verify that the code is vectorized and optimized to work efficiently on large datasets. ONLY implement low level vectorizations that are safe and do not touch underlying logic (ex. simplifying expensive loops).

## Critical Constraints
- Do NOT generate new specs or documentation; I provide hand authored specs. Update the readme for new features.
- Maintain strict type safety across all module boundaries.

