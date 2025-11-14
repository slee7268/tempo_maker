# Quick Start Guide

Get started mining Tempo puzzles in 5 minutes!

## Prerequisites

1. **Python 3.8+** (check: `python3 --version`)
2. **Fairy-Stockfish** engine

## Step 1: Install Fairy-Stockfish

### macOS (Homebrew)
```bash
brew install fairy-stockfish
```

### Linux
```bash
# Download latest release
wget https://github.com/fairy-stockfish/Fairy-Stockfish/releases/latest/download/fairy-stockfish-largeboard_x86-64
chmod +x fairy-stockfish-largeboard_x86-64
sudo mv fairy-stockfish-largeboard_x86-64 /usr/local/bin/fairy-stockfish
```

### Windows
1. Download from https://github.com/fairy-stockfish/Fairy-Stockfish/releases
2. Extract and note the full path to the .exe file

## Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install python-chess
```

## Step 3: Get Crazyhouse Games

### Option A: Download from Lichess
```bash
# Download a month of Crazyhouse games (warning: large file ~500MB+)
wget https://database.lichess.org/standard/lichess_db_crazyhouse_rated_2024-12.pgn.zst
zstd -d lichess_db_crazyhouse_rated_2024-12.pgn.zst
```

### Option B: Create a Test PGN
Create a small test file `test.pgn` with a few Crazyhouse games:
```pgn
[Event "Rated Crazyhouse game"]
[Site "https://lichess.org/abc123"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[WhiteElo "1650"]
[BlackElo "1600"]
[Variant "Crazyhouse"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. d3 Nf6 5. Nc3 d6 6. Bg5 h6 7. Bh4 g5 
8. Nxg5 hxg5 9. Bxg5 Rg8 10. Qf3 Nd4 11. Qxf6 Qxf6 12. Bxf6 P@g2 
13. Rg1 Nxc2+ 14. Kd2 Nxa1 1-0
```

## Step 4: Mine Your First Puzzles

### Quick Test (First 10 Puzzles)
```bash
python3 tempo_miner.py \
  --in test.pgn \
  --verify-engine $(which fairy-stockfish) \
  --out puzzles.jsonl \
  --preview puzzles.csv \
  --sample 10 \
  --progress-every-games 1
```

### Production Run (100 Puzzles from Large Database)
```bash
python3 tempo_miner.py \
  --in lichess_db_crazyhouse_rated_2024-12.pgn \
  --verify-engine $(which fairy-stockfish) \
  --out tempo_puzzles.jsonl \
  --preview tempo_puzzles.csv \
  --sample 100 \
  --tail-plies 16 \
  --movetime-ms 800 \
  --max-mate-in 9 \
  --progress-every-games 50
```

## Step 5: View Your Puzzles

### Check the CSV Preview
```bash
cat puzzles.csv
```

Output:
```csv
id,mateIn,sideToMove,fenStart,difficultyRating
zh:abc123:ply28,5,w,r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R[P] w KQkq - 0 1,1450
```

### Look at a Full Puzzle
```bash
head -1 puzzles.jsonl | python3 -m json.tool
```

Output:
```json
{
  "id": "zh:abc123:ply28",
  "fenStart": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R[P] w KQkq - 0 1",
  "sideToMove": "w",
  "mateIn": 5,
  "solutionSAN": ["Bxf7+", "Kxf7", "Ng5+", "Ke7", "Qh5", "Nf6", "Qf7#"],
  "difficultyRating": 1450,
  ...
}
```

## Common Issues

### "ERROR: --verify-engine is required"
Make sure you provide the full path to fairy-stockfish:
```bash
which fairy-stockfish  # Find the path
python3 tempo_miner.py --verify-engine /usr/local/bin/fairy-stockfish --in test.pgn
```

### "No module named 'chess'"
Install the dependency:
```bash
pip install python-chess
```

### No puzzles found
- Your PGN might not have positions with forced checking sequences
- Try a larger PGN file with more games
- Try adjusting `--max-mate-in` to a higher value (e.g., 12)

## Understanding the Output

### Difficulty Ratings
- **800-1100**: Beginner (mate in 3-4, simple patterns)
- **1100-1400**: Intermediate (mate in 5-6, some complexity)
- **1400-1700**: Advanced (mate in 7-8, multiple variations)
- **1700-2000**: Expert (mate in 9+, complex trees)
- **2000-2500**: Master (very long forcing sequences)

### Solution Format
- **solutionSAN**: Human-readable notation (e.g., "Bxf7+", "N@f3")
- **solutionUCI**: Computer notation (e.g., "c4f7", "N@f3")
- **Drops**: Shown as "P@h7" (piece @ square)
- **Check**: Moves with "+" give check (Tempo rule)

### Tree Structure
The `tree` field shows all possible checking moves:
- **"force"**: This move leads to forced mate
- **"fail"**: This move doesn't force mate (wrong variation)
- **"mate"**: This move delivers immediate checkmate

## Next Steps

1. **Mine more puzzles**: Increase `--sample` or remove it to process all games
2. **Filter by difficulty**: Sort puzzles.csv by difficultyRating
3. **Integrate with Flutter**: Parse the JSONL file and load puzzles into your app
4. **Customize settings**: Adjust `--max-mate-in`, `--movetime-ms` for different puzzle types

## Advanced Usage

### High-Quality Puzzles Only
```bash
# Filter for high-rated games (1800+ ELO)
# Use pgn-extract or similar tool first, then:
python3 tempo_miner.py \
  --in high_rated.pgn \
  --verify-engine $(which fairy-stockfish) \
  --max-mate-in 7 \
  --movetime-ms 1000 \
  --sample 50
```

### Quick Testing (Fast Settings)
```bash
python3 tempo_miner.py \
  --in test.pgn \
  --verify-engine $(which fairy-stockfish) \
  --movetime-ms 400 \
  --max-mate-in 5 \
  --sample 5
```

## Integration with Flutter App

```dart
import 'dart:convert';
import 'dart:io';

void loadPuzzles() async {
  final file = File('puzzles.jsonl');
  final lines = await file.readAsLines();
  
  for (var line in lines) {
    final puzzle = jsonDecode(line);
    
    // Load puzzle data
    final fen = puzzle['fenStart'];
    final toMove = puzzle['sideToMove']; // 'w' or 'b'
    final solution = List<String>.from(puzzle['solutionUCI']);
    final difficulty = puzzle['difficultyRating'];
    final movesToMate = puzzle['mateIn'];
    
    // Initialize your Crazyhouse board with FEN
    // Orient board based on toMove
    // Validate moves against solution
    // Filter by difficulty for puzzle selection
  }
}
```

## Support

For issues or questions:
1. Check README.md for detailed documentation
2. Review example_puzzle_output.json for format reference
3. Test with test_difficulty_rating.py to verify setup

Happy puzzle mining! 🎯♟️
