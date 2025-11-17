# Tempo Chess Puzzle Maker - AI Agent Instructions

## Project Overview
Tempo is a Flutter chess puzzle app with a unique twist: **every move must give check**. It combines standard chess with Crazyhouse mechanics (piece drops from a pocket). Players solve puzzles by delivering checkmate while adhering to the tempo rule. This specific portion of the project focuses on mining high-quality tempo puzzles from a large PGN database using a custom Python tool.

## Core Game Mechanics
1. **Tempo Rule**: Every user move MUST put the opponent's king in check (enforced by `TempoRules`)
2. **Crazyhouse Drops**: Captured pieces go to the capturer's pocket and can be dropped back onto the board
3. **Victory Condition**: Deliver checkmate while maintaining the tempo rule throughout

## Puzzle Mining Tool
We have developed a Python script `tempo_miner.py` that processes a PGN database of Crazyhouse games to extract valid tempo puzzles. Please read SUMMARY.md for detailed requirements and usage instructions.

### Chess Engine Integration
We use Fairy Stockfish to validate puzzles and find optimal mate sequences.

### Puzzle input
We use lichess db crazyhouse games in PGN format as input for mining puzzles.

### Puzzle Output
Puzzles are stored in JSONL format with solution lines and difficulty estimates.


