# Tempo Puzzle Miner

A specialized tool for mining Crazyhouse chess puzzles that comply with the **Tempo rule**: every move must give check.

## Overview

Tempo is a Flutter chess puzzle app combining standard chess with Crazyhouse mechanics (piece drops from a pocket). Players solve puzzles by delivering checkmate while adhering to the tempo rule—every user move must put the opponent's king in check.

This tool mines puzzles from Crazyhouse PGN game databases and validates that:
1. Every attacking move gives check to the opponent's king
2. There exists a forced checkmate sequence
3. The solution includes proper Crazyhouse drops when applicable

## Features

### Core Mining Capabilities
- **Attacker-Perspective Forcedness**: Proves forced mate BEFORE pushing moves using Fairy-Stockfish
- **Checks-Only Tree**: Builds complete game trees with only checking moves
- **Crazyhouse Support**: Full support for piece drops with pocket notation
- **PV-Safe Defense**: Accurately extracts defender responses from engine PV

### Puzzle Data Exported

Each puzzle includes:

1. **`fenStart`**: Starting position in FEN notation with Crazyhouse pockets `[]`
   - Example: `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[QRbn] w KQkq - 0 1`
   - Pocket notation: `[UPPERCASE=white pieces, lowercase=black pieces]`

2. **`sideToMove`**: Which player starts the puzzle (`"w"` or `"b"`)

3. **`solutionSAN`**: Solution in Standard Algebraic Notation (human-readable)
   - Example: `["Qh5+", "Ke7", "Qxe5#"]`
   - Drops shown as `Q@h5+` (piece @ square)

4. **`solutionUCI`**: Solution in UCI notation (for engines/apps)
   - Example: `["d1h5", "e8e7", "h5e5"]`
   - Drops shown as `Q@h5` (piece @ square)

5. **`mateIn`**: Number of plies (half-moves) until checkmate
   - Mate in 3 plies = 2 moves by attacker (check, mate) with 1 defender reply

6. **`difficultyRating`**: Computed puzzle difficulty (800-2500 range)
   - Based on: mate distance, tree complexity, drops, player ELOs, solution length

7. **`tree`**: Complete move tree with all checking variations
   - Shows "force", "fail", and "mate" classifications for each move

8. **Player Context**: `whiteElo`, `blackElo`, `result`, `site`, `date`

### Difficulty Rating System

The difficulty rating (800-2500) is calculated from multiple factors:

1. **Mate-in Distance**: Longer forced sequences = harder (+0 to +400)
2. **Tree Complexity**: More alternative checking moves = harder (+0 to +300)
3. **Forcing Clarity**: Lower ratio of forcing moves = harder (+0 to +100)
4. **Crazyhouse Drops**: More drops in solution = harder (+40 per drop)
5. **Player ELO**: Higher rated games provide baseline adjustment (±200)
6. **Solution Length**: Longer solutions = more complex (+0 to +150)

Base rating starts at 1200, factors are added, and the result is clamped to 800-2500 range.

## Installation

### Prerequisites

1. **Python 3.8+**
2. **Python-Chess library** (with variant support)
3. **Fairy-Stockfish** chess engine

### Install Python Dependencies

```bash
pip install python-chess
```

### Install Fairy-Stockfish

Download the latest Fairy-Stockfish binary for your platform:
- https://github.com/fairy-stockfish/Fairy-Stockfish/releases

Make it executable and note its path:
```bash
chmod +x fairy-stockfish
# Example path: /usr/local/bin/fairy-stockfish
```

## Usage

### Basic Usage

Mine puzzles from a Crazyhouse PGN file:

```bash
python tempo_miner.py \
  --in lichess_db_crazyhouse_rated_2025-09.pgn \
  --verify-engine /usr/local/bin/fairy-stockfish \
  --out tempo_puzzles.jsonl \
  --preview puzzles_preview.csv
```

### Quick Sample Run (First 50 Puzzles)

```bash
python tempo_miner.py \
  --in games.pgn \
  --verify-engine /usr/local/bin/fairy-stockfish \
  --sample 50 \
  --tail-plies 16 \
  --progress-every-games 10
```

### Advanced Options

```bash
python tempo_miner.py \
  --in lichess_db_crazyhouse_rated_2025-09.pgn \
  --verify-engine $(which fairy-stockfish) \
  --out crazy_tempo.jsonl \
  --preview zh_tempo_preview.csv \
  --tail-plies 16 \
  --sample 100 \
  --movetime-ms 800 \
  --engine-timeout 4 \
  --max-mate-in 9 \
  --max-nodes 3000 \
  --per-puzzle-timeout-sec 20 \
  --progress-every-games 10
```

## Command Line Options

### Required
- `--in`: Input PGN file with Crazyhouse games (required)
- `--verify-engine`: Path to Fairy-Stockfish binary (required)

### Output
- `--out`: Output JSONL file (default: `zh_tempo.jsonl`)
- `--preview`: Preview CSV file (default: `zh_tempo_preview.csv`)

### Candidate Selection
- `--tail-plies`: Try root positions within N plies of game end (default: 16, 0=whole game)
- `--sample`: Stop after finding N puzzles (default: 0=no limit)
- `--max-games`: Stop after reading N games (default: 0=no limit)

### Engine Configuration
- `--movetime-ms`: Engine think time per call in milliseconds (default: 800)
- `--engine-timeout`: Max seconds to wait for engine response (default: 4.0)
- `--depth`: Engine search depth if movetime=0 (default: 0, uses movetime instead)

### Quality Constraints (rejection filters)
- `--max-mate-in`: Reject puzzles with mate distance exceeding N plies (default: 9, 0=no cap)
- `--max-nodes`: Reject puzzles with tree nodes exceeding N (default: 3000, 0=no cap)
- `--per-puzzle-timeout-sec`: Reject puzzles taking longer than N seconds (default: 20.0)

### Progress Reporting
- `--progress-every-games`: Print progress every N games (default: 200, 0=disable)
- `--progress-sec`: Print progress every N seconds (default: 0.0=off)

### Other
- `--seed`: Random seed for reproducibility (default: 0)

## Output Format

### JSONL Output (`--out`)

Each line is a complete puzzle JSON object:

```json
{
  "id": "zh:abc123:ply42",
  "variant": "crazyhouse",
  "fenStart": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R[P] w KQkq - 0 1",
  "sideToMove": "w",
  "mateIn": 5,
  "solutionSAN": ["Bxf7+", "Kxf7", "Ng5+", "Ke7", "Qh5", "Nf6", "Qf7#"],
  "solutionUCI": ["c4f7", "e8f7", "f3g5", "f7e7", "d1h5", "f6f6", "h5f7"],
  "difficultyRating": 1450,
  "whitePocket": {"P": 1, "N": 0, "B": 0, "R": 0, "Q": 0},
  "blackPocket": {"P": 0, "N": 0, "B": 0, "R": 0, "Q": 0},
  "tags": ["alwaysCheck", "dropOK", "fromGame", "forced", "fullChecks"],
  "whiteElo": "1654",
  "blackElo": "1612",
  "result": "1-0",
  "site": "https://lichess.org/abc123",
  "date": "2025.09.15",
  "tree": {
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R[P] w KQkq - 0 1": {
      "attacker": [
        {
          "uci": "c4f7",
          "class": "force",
          "defense": {"uci": "e8f7"},
          "nextFen": "r1bq3r/ppppkppp/2n2n2/2b1p3/4P3/5N2/PPPP1PPP/RNBQK2R[BP] b KQ - 0 1"
        }
      ]
    }
  }
}
```

### CSV Preview (`--preview`)

Quick overview of mined puzzles:

```csv
id,mateIn,sideToMove,fenStart,difficultyRating
zh:abc123:ply42,5,w,r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R[P] w KQkq - 0 1,1450
zh:def456:ply38,3,b,rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R[] b KQkq - 0 1,1280
```

## How It Works

### Mining Process

1. **Game Scanning**: Reads Crazyhouse games from PGN
2. **Position Selection**: Extracts candidate positions from the last `--tail-plies` of each game
3. **Forcedness Proof**: For each position, checks if at least one checking move forces mate
   - Uses `searchmoves` to prove forcedness BEFORE pushing the move
4. **Tree Building**: Recursively builds a checks-only game tree
   - "force" = move still forces mate
   - "fail" = move does not force mate (ends that branch)
   - "mate" = immediate checkmate
5. **Quality Filtering**: Applies constraints (`max-mate-in`, `max-nodes`, timeout)
6. **Difficulty Calculation**: Computes puzzle rating based on multiple factors
7. **Export**: Writes puzzle data to JSONL and CSV files

### Tree Structure

The `tree` field contains all attacking variations:

```json
{
  "fen_position": {
    "attacker": [
      {
        "uci": "d1h5",
        "class": "force",
        "defense": {"uci": "e8e7"},
        "nextFen": "next_position_fen"
      },
      {
        "uci": "f3g5",
        "class": "fail",
        "defense": {"uci": "d8f6"},
        "nextFen": "another_position_fen"
      },
      {
        "uci": "c4f7",
        "class": "mate",
        "defense": null,
        "nextFen": null
      }
    ]
  }
}
```

## Integration with Tempo Flutter App

To use these puzzles in your Flutter app:

1. **Load the JSONL file** and parse each line as JSON
2. **Set up the board** using `fenStart` (initialize Crazyhouse board with pockets)
3. **Orient the board** based on `sideToMove` (white or black on bottom)
4. **Validate moves** against `solutionUCI` or use `tree` for interactive hints
5. **Display difficulty** using `difficultyRating` for puzzle selection
6. **Show solution** using `solutionSAN` for human-readable move notation

### Example Flutter Usage

```dart
// Parse puzzle
final puzzle = jsonDecode(jsonLine);
final fen = puzzle['fenStart'];
final playerToMove = puzzle['sideToMove'] == 'w' ? chess.Color.WHITE : chess.Color.BLACK;
final solution = List<String>.from(puzzle['solutionUCI']);
final difficulty = puzzle['difficultyRating'];
final movesToMate = puzzle['mateIn'];

// Initialize board (using your Crazyhouse library)
final board = CrazyhouseBoard.fromFEN(fen);

// Orient board so player's pieces are on bottom
final orientation = playerToMove;

// Check user's move gives check
bool validateMove(Move move) {
  board.makeMove(move);
  bool givesCheck = board.isInCheck(!playerToMove);
  board.undoMove();
  return givesCheck;
}
```

## Example Puzzle

Here's a sample puzzle from the output:

```json
{
  "id": "zh:sample:ply20",
  "fenStart": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R[P] w KQkq - 0 1",
  "sideToMove": "w",
  "mateIn": 5,
  "solutionSAN": ["Bxf7+", "Kxf7", "Ng5+", "Ke7", "Qh5", "Nf6", "Qf7#"],
  "solutionUCI": ["c4f7", "e8f7", "f3g5", "f7e7", "d1h5", "f6f6", "h5f7"],
  "difficultyRating": 1450
}
```

**Explanation:**
1. White sacrifices the bishop with check: `Bxf7+`
2. Black must capture: `Kxf7`
3. White checks with the knight: `Ng5+`
4. Black's king moves: `Ke7`
5. White brings the queen into the attack: `Qh5`
6. Black blocks with the knight: `Nf6`
7. White delivers checkmate: `Qf7#`

Every white move gives check, satisfying the Tempo rule!

## Where to Get Crazyhouse Games

### Lichess Database
- Monthly Crazyhouse game databases: https://database.lichess.org/
- Download the `.pgn.zst` file and decompress:
  ```bash
  zstd -d lichess_db_crazyhouse_rated_2025-09.pgn.zst
  ```

### Filter by Rating
You can preprocess the PGN to include only high-rated games for better puzzle quality:
```bash
# Extract games where both players are rated 1800+
pgn-extract --minelo 1800 input.pgn -o filtered.pgn
```

## Performance Tips

1. **Use `--movetime-ms`**: Provides consistent runtime (default 800ms is good)
2. **Set `--sample`**: For quick testing, mine just 50-100 puzzles
3. **Limit `--tail-plies`**: Focusing on endgame (last 16 plies) finds more forced mates
4. **Adjust `--max-mate-in`**: Lower values (5-7) find simpler puzzles faster
5. **Use `--max-nodes`**: Prevents extremely complex puzzles that take too long
6. **Enable progress**: Use `--progress-every-games 100` to monitor progress

## Troubleshooting

### "ERROR: --verify-engine is required"
- Provide the full path to Fairy-Stockfish: `--verify-engine /path/to/fairy-stockfish`

### No puzzles found
- Games may not have positions with forced checking sequences
- Try increasing `--tail-plies` to 20 or 30
- Try different PGN files with higher-rated games
- Reduce `--max-mate-in` if looking for simpler puzzles

### Mining is too slow
- Reduce `--movetime-ms` (e.g., 400-600ms)
- Reduce `--engine-timeout` to 2-3 seconds
- Increase `--tail-plies` to skip early game positions
- Use `--sample` to stop early

### Puzzles are too hard/easy
- Difficulty rating auto-adjusts based on multiple factors
- Filter output by `difficultyRating` range in your app
- For easier puzzles: use `--max-mate-in 5`
- For harder puzzles: use `--max-mate-in 12` and increase `--max-nodes`

## Technical Details

### FEN with Crazyhouse Pockets
Standard chess FEN with pocket notation appended:
```
<piece placement>/<active color>/<castling>/<en passant>/<halfmove>/<fullmove>[pockets]
```

Pockets: `[UPPERCASE=white, lowercase=black]`
- Empty pockets: `[]`
- White has Queen + Rook, Black has knight + pawn: `[QRnp]`

### Move Notation

**UCI**: `<from><to>[promotion]` or `<piece>@<square>` for drops
- Regular: `e2e4`, `e7e8q`
- Drop: `N@f3`, `P@h7`

**SAN**: Standard Algebraic Notation with `@` for drops
- Regular: `e4`, `Nf3`, `O-O`
- Drop: `N@f3+`, `P@h7`
- Check: `+`
- Checkmate: `#`

## Contributing

To improve the difficulty rating algorithm, puzzle quality filters, or add new features, please submit issues or pull requests to the repository.

## License

See LICENSE file for details.
