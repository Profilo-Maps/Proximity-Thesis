# Project Architecture & Rules
You are a senior developer helping me implement this feature of a larger project. Proximity is a graph network exported as a parquet built on OSM and government data. It will be used for network analysis and data crowdsourcing by other applications. 


**Path to Python:** `C:/Dev/Proximity/.venv/Scripts/python.exe`
**Run commands with:** `uv run python` (from project root)

**Spec Checking Strategy:** Before implementing a feature, check the specs folder for guidance.

**Uncertainty Strategy:** ALWAYS ask questions before writing code. These questions should be used to highlight design decisions that are ambiguously justified by logic or that are matters of taste.When recommending approaches, order them by complexity of implementation.

## Correction History Protocol
WHENEVER you create a diagnostic script, document the script with comments on the error and the solution in the /notesforrobot folder. ALSO document the problem solving process you used to revise a diagnostic script that led to the optimal solution being uncovered. 

**Plan Document Etiquette:** DO NOT use the superpowers:writing-plans skill, start implementing after a design is approved. After a feature is implemented, check the /docs folder for intermediate implementation docs and consolidate them to the readme.md.

## Critical Constraints
- Do NOT generate new specs or documentation; I provide hand authored specs. Update the readme for new features.
- Maintain strict type safety across all module boundaries. Use Pyright to typecheck after implementing a feature.

## Typechecking
Use the integrated VS Code Pylance extension for type checking.

## Package Management
ALWAYS USE UV FOR LIBRARY AND ENVIRONMENT MANAGEMENT
uv init: Create a new Python project.
uv add: Add a dependency to the project.
uv remove: Remove a dependency from the project.
uv sync: Sync the project's dependencies with the environment.
uv lock: Create a lockfile for the project's dependencies.
uv run: Run a command in the project environment.

