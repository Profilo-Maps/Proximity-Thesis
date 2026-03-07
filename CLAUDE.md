# Project Architecture & Rules
You are a senior developer helping me implement this feature of a larger project. Proximity is a graph network exported as a parquet built on OSM and government data. It will be used for network analysis and data crowdsourcing by other applications. 

**Uncertainty Strategy:** ALWAYS ask questions before writing code. These questions should be used to highlight design decisions that are ambiguously justified by logic or that are matters of taste.

**Plan Document Etiquette:** DO NOT use the superpowers:writing-plans skill, start implementing after a design is approved. After a feature is implemented, check the /docs folder for intermediate implementation docs and consolidate them to the readme.md.

**Main data schema:** @specs/ProximitySchema.md
**Pipeline Architecture:** @specs\ProximityPipelineOutline.md
**Path to Python:**(ParkximityENV) PS C:\Dev\Proximity> & C:\Users\karna\miniconda3\envs\ParkximityENV\python.exe 


## Critical Constraints
- Do NOT generate new specs or documentation; I provide hand authored specs. Update the readme for new features.
- Maintain strict type safety across all module boundaries.