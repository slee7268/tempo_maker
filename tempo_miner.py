#!/usr/bin/env python3
"""
Crazyhouse -> Tempo puzzle miner (Tempo variant = always checking + crazyhouse-style drops).

Takes a PGN of Crazyhouse games (e.g. lichess_db_crazyhouse_rated_2025-10.pgn),
uses Fairy-Stockfish largeboard to:
  * scan for “always-check” sequences (tempo-style forcing sequences),
  * verify they are correct and lead to mate within a fixed depth,
  * output them as JSONL suitable for a Tempo chess app.

High-level pipeline:

1. Read Crazyhouse PGN using python-chess.
2. For each game, walk moves and build:
   - board states,
   - crazyhouse pockets,
   - a search tree tracking attacker/defender moves.
3. When a promising “always-check” candidate is found:
   - verify with engine (UCI: Fairy-Stockfish)
   - ensure check on every move
   - ensure mate within `--max-mate-in`
4. Emit a record:
   - ID, start FEN, side to move
   - SAN/uci solution lines
   - simple difficulty estimate
   - optional tree for debugging/training.

This script is intentionally self-contained so it can be dropped into
a project and run directly.

Requirements:
  pip install python-chess tqdm

Engine:
  - Fairy-Stockfish largeboard binary, path given by --engine-path or --verify-engine
"""

import argparse
import json
import os
import sys
import time
import math
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import chess
import chess.pgn
from tqdm import tqdm
import subprocess

# -------------------- Engine Wrapper --------------------

@dataclass
class EngineConfig:
    path: str
    timeout_sec: float = 2.0
    concurrency: int = 1


class Engine:
    """
    Minimal UCI wrapper around Fairy-Stockfish (or any UCI engine).
    Only supports what the miner needs:
      * "uci" / "isready"
      * "position fen"
      * "go depth N" or "go nodes N"
      * parse bestmove and (optionally) info lines with score/mate.
    """

    def __init__(self, path: str, timeout_sec: float = 2.0):
        self.path = path
        self.timeout_sec = timeout_sec
        self.p = None

    def __enter__(self):
        # Ensure we use an absolute path if a bare filename was passed
        if not os.path.isabs(self.path):
            # Resolve relative to current working directory
            self.path = os.path.abspath(self.path)

        self.p = subprocess.Popen(
            [self.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok")
        # CRITICAL: Set variant to crazyhouse to enable drop moves
        self._send("setoption name UCI_Variant value crazyhouse")
        self._send("isready")
        self._wait_for("readyok")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self._send("quit")
        except Exception:
            pass
        if self.p:
            self.p.terminate()
            self.p = None

    def _send(self, cmd: str):
        assert self.p is not None
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def _read_line(self, timeout: Optional[float] = None) -> Optional[str]:
        # Simple blocking read with timeout via poll loop
        if timeout is None:
            return self.p.stdout.readline()

        t0 = time.time()
        while True:
            if time.time() - t0 > timeout:
                return None
            line = self.p.stdout.readline()
            if line:
                return line

    def _wait_for(self, token: str, timeout: Optional[float] = 5.0):
        while True:
            line = self._read_line(timeout)
            if line is None:
                raise RuntimeError(f"Engine did not respond with {token} in time")
            if token in line:
                return

    def analyze_fen(self, fen: str, nodes: int = 2000) -> Dict[str, Any]:
        """
        Run a single search from FEN, return:
          {
            "bestmove": "e2e4",
            "score_cp": <int or None>,
            "mate": <int or None>,
            "pv": ["e2e4", "e7e5", ...]
          }
        """
        self._send("ucinewgame")
        self._send(f"position fen {fen}")
        self._send(f"go nodes {nodes}")

        bestmove = None
        score_cp = None
        mate = None
        pv = []

        while True:
            line = self._read_line(self.timeout_sec)
            if line is None:
                break
            line = line.strip()
            if line.startswith("info "):
                # Parse score and PV if present
                tokens = line.split()
                if "score" in tokens:
                    idx = tokens.index("score")
                    if idx + 2 < len(tokens):
                        kind = tokens[idx + 1]
                        val = tokens[idx + 2]
                        if kind == "cp":
                            try:
                                score_cp = int(val)
                            except Exception:
                                pass
                        elif kind == "mate":
                            try:
                                mate = int(val)
                            except Exception:
                                pass
                if "pv" in tokens:
                    idx = tokens.index("pv")
                    pv = tokens[idx + 1 :]
            elif line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    bestmove = parts[1]
                break

        return {
            "bestmove": bestmove,
            "score_cp": score_cp,
            "mate": mate,
            "pv": pv,
        }


# -------------------- Tree Structures --------------------

@dataclass
class MoveInfo:
    uci: str
    san: str
    score_cp: Optional[int] = None
    mate: Optional[int] = None
    klass: str = "normal"  # "force" if checking move, etc.


@dataclass
class Node:
    fen: str
    attacker_moves: List[MoveInfo] = dataclasses.field(default_factory=list)
    defender_moves: List[MoveInfo] = dataclasses.field(default_factory=list)


class SearchTree:
    """
    Simple game tree keyed by FEN -> Node.
    Attacker moves = side creating checks (tempo).
    Defender moves = replies.
    """

    def __init__(self):
        self.nodes: Dict[str, Node] = {}

    def get_or_create(self, fen: str) -> Node:
        if fen not in self.nodes:
            self.nodes[fen] = Node(fen=fen)
        return self.nodes[fen]

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for fen, node in self.nodes.items():
            out[fen] = {
                "attacker": [dataclasses.asdict(m) for m in node.attacker_moves],
                "defender": [dataclasses.asdict(m) for m in node.defender_moves],
            }
        return out


# -------------------- Crazyhouse Helpers --------------------

def pockets_to_maps(board: chess.Board) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Convert python-chess crazyhouse pockets to simple piece-count maps
    for both sides: {"P": 2, "N": 1, ...}.
    """
    # python-chess crazyhouse: board.pockets[color] is a PieceMap-like
    # structure mapping a piece_type to count.
    w_map: Dict[str, int] = {}
    b_map: Dict[str, int] = {}

    if not hasattr(board, "pockets") or board.pockets is None:
        return w_map, b_map

    for color, target in ((chess.WHITE, w_map), (chess.BLACK, b_map)):
        pocket = board.pockets[color]
        for piece_type in range(1, 7):
            cnt = pocket.count(piece_type)
            if cnt > 0:
                piece_symbol = chess.Piece(piece_type, color).symbol()
                # We want 'P', 'N', etc. uppercased for consistency
                target[piece_symbol.upper()] = cnt

    return w_map, b_map


def is_checking_move(board: chess.Board, move: chess.Move) -> bool:
    """
    Return True if move gives check.
    """
    board_push = board.copy(stack=False)
    board_push.push(move)
    return board_push.is_check()


def crazyhouse_drop_san(board: chess.Board, move: chess.Move) -> str:
    """
    python-chess SAN will already show drops as "P@d5" for crazyhouse.
    Provided for completeness if any custom formatting is needed later.
    """
    return board.san(move)


# -------------------- Difficulty Rating --------------------

def calculate_difficulty_rating(mate_in, tree, start_board, game, solution_moves):
    """
    Calculate puzzle difficulty rating (800-2500 range) for Tempo puzzles.
    
    Key factors:
    1. Mate depth (exponential - longer forcing sequences are harder)
    2. Drop moves (adds tactical complexity unique to Crazyhouse)
    3. Pocket complexity (more pieces = more candidate moves to consider)
    4. Source game quality (higher rated games = better tactics)
    5. Endgame positions (fewer pieces = harder calculation)
    
    Note: In Tempo, all attacking moves are checks (forcing), so we don't
    distinguish quiet vs forcing moves.
    """
    base_rating = 1000
    
    # Factor 1: Mate depth (exponential - longer sequences much harder)
    # mate_in=1: 10, mate_in=3: 90, mate_in=5: 250, mate_in=9: 810
    mate_factor = mate_in * mate_in * 10
    
    # Factor 2: Crazyhouse drops (harder to visualize and calculate)
    drop_count = sum(1 for m in (solution_moves or []) if '@' in str(m))
    drop_factor = drop_count * 50
    
    # Factor 3: Pocket complexity (more pieces in pocket = more candidate moves)
    pocket_factor = 0
    if hasattr(start_board, 'pockets') and start_board.pockets is not None:
        try:
            white_pocket_count = sum(start_board.pockets[chess.WHITE].count(pt) for pt in range(1, 7))
            black_pocket_count = sum(start_board.pockets[chess.BLACK].count(pt) for pt in range(1, 7))
            total_pocket_pieces = white_pocket_count + black_pocket_count
            pocket_factor = total_pocket_pieces * 15
        except:
            pocket_factor = 0
    
    # Factor 4: Source game quality (higher rated games = better tactics)
    try:
        white_elo = int(game.headers.get("WhiteElo", "1500"))
        black_elo = int(game.headers.get("BlackElo", "1500"))
        avg_elo = (white_elo + black_elo) / 2
        # Adjust based on player strength (±150 range)
        elo_adjustment = (avg_elo - 1500) * 0.15
        elo_adjustment = max(-150, min(150, elo_adjustment))
    except:
        elo_adjustment = 0
    
    # Factor 5: Endgame bonus (fewer pieces = harder to calculate precisely)
    try:
        piece_count = len(start_board.piece_map())
        # More bonus as piece count decreases (endgames are tactically harder)
        endgame_bonus = max(0, (32 - piece_count) * 5)
    except:
        endgame_bonus = 0
    
    # Combine all factors
    difficulty = (
        base_rating
        + mate_factor
        + drop_factor
        + pocket_factor
        + elo_adjustment
        + endgame_bonus
    )
    
    # Clamp to reasonable range
    difficulty = max(800, min(2500, int(difficulty)))
    
    return difficulty


# -------------------- Miner --------------------
def make_record(game, pid, start_board, mate_in, solutionSAN, solutionUCI=None, tree=None):
    w_poc, b_poc = pockets_to_maps(start_board)
    
    # Calculate difficulty rating
    difficulty_rating = calculate_difficulty_rating(
        mate_in, tree, start_board, game, solutionSAN or []
    )
    
    rec = {
        "id": pid,
        "fenStart": start_board.fen(),
        "sideToMove": "w" if start_board.turn==chess.WHITE else "b",
        "mateIn": int(mate_in),
        "solutionSAN": solutionSAN[:],
        "solutionUCI": solutionUCI[:] if solutionUCI else None,
        "whitePocket": w_poc, 
        "blackPocket": b_poc,
        "difficultyRating": difficulty_rating,
        "tags": ["alwaysCheck","dropOK","fromGame","forced","fullChecks"],
        "whiteElo": game.headers.get("WhiteElo"),
        "blackElo": game.headers.get("BlackElo"),
        "result": game.headers.get("Result"),
        "site": game.headers.get("Site"),
    }
    # COMMENTED OUT: Tree is not needed for basic puzzle usage
    # if tree is not None:
    #     rec["tree"] = tree
    return rec

def make_puzzle_id(game, start_ply, puzzle_counter):
    """Generate unique puzzle ID from game site URL and puzzle counter."""
    site = game.headers.get("Site") or "unknown"
    # Extract just the game ID from the Lichess URL
    if "lichess.org/" in site:
        game_id = site.split("/")[-1]
    else:
        game_id = site
    return f"{game_id}_ply{start_ply}_p{puzzle_counter}"

def extract_always_check_sequence(game: chess.pgn.Game,
                                  engine: Engine,
                                  start_ply: int,
                                  max_mate_in: int,
                                  max_nodes: int,
                                  tail_plies: int,
                                  tree: SearchTree) -> Optional[Tuple[chess.Board, int, List[str], List[str]]]:
    """
    Look at game from ply `start_ply` onward. Try to find:
      - a sequence where attacking side gives check on every move (Tempo rule),
      - verified by engine as mating within `max_mate_in` plies,
      - bounded by `tail_plies` from game end.

    Returns (start_board, mate_in, solutionSAN, solutionUCI) or None.
    """
    # Reconstruct board up to start_ply
    board = game.board()
    mainline_moves = list(game.mainline_moves())
    for mv in mainline_moves[:start_ply]:
        board.push(mv)

    # Only search near the end for now
    remaining = len(mainline_moves) - start_ply
    if tail_plies > 0 and remaining > tail_plies:
        return None

    # Now start from this position and see if engine finds a mating always-check line
    # board.fen() for Crazyhouse includes pocket notation [QPnb] automatically
    fen0 = board.fen()
    engine_info = engine.analyze_fen(fen0, nodes=max_nodes)
    best = engine_info["bestmove"]
    mate = engine_info["mate"]

    if best is None or mate is None:
        return None

    # We only care about forced mates within max_mate_in
    if abs(mate) > max_mate_in:
        return None

    # Build the PV as a sequence of UCI moves
    pv_uci = engine_info["pv"]
    if not pv_uci:
        return None

    # Convert PV to SAN, enforce "always check"
    # CRITICAL: Attacker moves are at indices 0, 2, 4, ... (every other move starting from 0)
    # Defender moves are at indices 1, 3, 5, ...
    # We need EVERY attacker move to give check (including the final checkmate)
    tmp = board.copy(stack=False)
    solutionSAN: List[str] = []
    solutionUCI: List[str] = []
    attacker_turn = tmp.turn  # Remember who is the attacking side

    for i, u in enumerate(pv_uci):
        move = chess.Move.from_uci(u)
        if move not in tmp.legal_moves:
            break
        san = tmp.san(move)
        
        is_attacker_move = (tmp.turn == attacker_turn)
        
        # Push the move to see if it results in check
        tmp.push(move)
        
        # If this is an attacker move, it MUST result in check
        if is_attacker_move and not tmp.is_check():
            # Violates always-check rule
            return None

        solutionSAN.append(san)
        solutionUCI.append(u)

        if tmp.is_game_over():
            break

    if not solutionSAN:
        return None

    # Count only attacker moves for mate-in calculation
    # Attacker moves are at indices 0, 2, 4, ...
    attacker_move_count = (len(solutionSAN) + 1) // 2
    mate_in = attacker_move_count

    # Verify first move is by attacker and gives check
    test_board = board.copy(stack=False)
    first_move = chess.Move.from_uci(solutionUCI[0])
    if not is_checking_move(test_board, first_move):
        return None

    # Optional: record tree info (attacker moves, etc.)
    # COMMENTED OUT: Tree is not needed for basic puzzle usage
    # node0 = tree.get_or_create(fen0)
    # node0.attacker_moves.append(
    #     MoveInfo(
    #         uci=solutionUCI[0],
    #         san=solutionSAN[0],
    #         mate=mate,
    #         klass="force",
    #     )
    # )

    return board, mate_in, solutionSAN, solutionUCI


def iter_games(pgn_path: str):
    """
    Yield games from PGN file.
    """
    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            yield game


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Crazyhouse -> Tempo puzzle miner")
    ap.add_argument("--in", dest="input_pgn", required=True,
                    help="Input Crazyhouse PGN file (e.g. lichess_db_crazyhouse_rated_2025-10.pgn)")
    ap.add_argument("--out", dest="output_jsonl", required=True,
                    help="Output JSONL file for puzzles")
    ap.add_argument("--preview", dest="preview_csv", default=None,
                    help="Optional CSV preview file for quick inspection")
    ap.add_argument("--engine-path", dest="engine_path", default=None,
                    help="Path to Fairy-Stockfish largeboard engine binary")
    ap.add_argument("--verify-engine", dest="verify_engine", default=None,
                    help="Alternate engine path used to verify sequences")
    ap.add_argument("--max-mate-in", dest="max_mate_in", type=int, default=9,
                    help="Maximum mate-in plies allowed for a puzzle (abs(mate) <= this)")
    ap.add_argument("--max-nodes", dest="max_nodes", type=int, default=3000,
                    help="Max engine nodes per search")
    ap.add_argument("--tail-plies", dest="tail_plies", type=int, default=16,
                    help="Only look for puzzles within this many plies of game end (0 = whole game).")
    ap.add_argument("--progress-every-games", dest="progress_every_games", type=int, default=10,
                    help="Progress bar step: update every N games")
    ap.add_argument("--max-games", dest="max_games", type=int, default=0,
                    help="Optional cap on number of games to process (0 = all)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    engine_path = args.engine_path or args.verify_engine
    if not engine_path:
        print("Need --engine-path or --verify-engine", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(engine_path):
        print(f"Engine not found: {engine_path}", file=sys.stderr)
        sys.exit(1)

    # Set up engine
    engine_cfg = EngineConfig(path=engine_path, timeout_sec=2.0)

    # Prepare outputs
    out_f = open(args.output_jsonl, "w", encoding="utf-8")
    prev_f = None
    if args.preview_csv:
        prev_f = open(args.preview_csv, "w", encoding="utf-8")
        prev_f.write("id,fenStart,sideToMove,mateIn,difficultyRating,whiteElo,blackElo,site\n")

    tree = SearchTree()
    total_games = 0
    puzzle_counter = 0
    all_puzzles = []  # Collect all puzzles in memory to sort by difficulty

    with Engine(engine_cfg.path, timeout_sec=engine_cfg.timeout_sec) as eng:
        for game in tqdm(iter_games(args.input_pgn), desc="Mining puzzles"):
            total_games += 1
            if args.max_games and total_games > args.max_games:
                break

            # Only crazyhouse games
            if game.headers.get("Variant", "").lower() != "crazyhouse":
                continue

            mainline_moves = list(game.mainline_moves())
            game_len = len(mainline_moves)

            # Collect all candidate puzzles from this game
            game_candidates = []
            
            # Search from a range of plies near the end
            start_range = list(range(max(0, game_len - args.tail_plies), game_len))
            for start_ply in start_range:
                result = extract_always_check_sequence(
                    game,
                    eng,
                    start_ply=start_ply,
                    max_mate_in=args.max_mate_in,
                    max_nodes=args.max_nodes,
                    tail_plies=args.tail_plies,
                    tree=tree,
                )
                if result is None:
                    continue

                start_board, mate_in, solutionSAN, solutionUCI = result
                puzzle_counter += 1
                pid = make_puzzle_id(game, start_ply, puzzle_counter)

                rec = make_record(
                    game=game,
                    pid=pid,
                    start_board=start_board,
                    mate_in=mate_in,
                    solutionSAN=solutionSAN,
                    solutionUCI=solutionUCI,
                    tree=None,  # Tree not needed for basic puzzle usage
                )
                game_candidates.append(rec)

            # Only keep the puzzle with the longest mate sequence from this game
            if game_candidates:
                best_puzzle = max(game_candidates, key=lambda p: p["mateIn"])
                all_puzzles.append(best_puzzle)

            if args.progress_every_games and (total_games % args.progress_every_games == 0):
                # tqdm already shows progress but we can add an extra line if desired
                pass

    # Sort puzzles by difficulty rating (easiest to hardest)
    print(f"\nSorting {len(all_puzzles)} puzzles by difficulty...", file=sys.stderr)
    all_puzzles.sort(key=lambda p: p["difficultyRating"])

    # Write sorted puzzles to output
    print(f"Writing sorted puzzles to {args.output_jsonl}...", file=sys.stderr)
    for rec in all_puzzles:
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write CSV preview if requested
    if prev_f:
        for rec in all_puzzles:
            start_fen = rec["fenStart"]
            prev_f.write(
                f"{rec['id']},{start_fen},{rec['sideToMove']},"
                f"{rec['mateIn']},{rec['difficultyRating']},"
                f"{rec.get('whiteElo', '')},"
                f"{rec.get('blackElo', '')},"
                f"{rec.get('site', '')}\n"
            )

    out_f.close()
    if prev_f:
        prev_f.close()

    print(f"Processed {total_games} games, emitted {len(all_puzzles)} puzzles.", file=sys.stderr)
    if all_puzzles:
        print(f"Difficulty range: {all_puzzles[0]['difficultyRating']} - {all_puzzles[-1]['difficultyRating']}", file=sys.stderr)


if __name__ == "__main__":
    main()
