# CLAUDE.md

## Sub-agent model

Sub-agents that only read or search (no edits, no commands that change
anything) must run on Sonnet, not the main session's model. The main
session's model is unchanged.
