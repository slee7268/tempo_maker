#!/usr/bin/env python3
"""
Crazyhouse -> Tempo puzzle miner (attacker-perspective forcedness, PV-safe)

What it does
- Scans late plies of each game (last --tail-plies) for a root position where an
  ATTACKER checking move forces mate (proven BEFORE pushing, via searchmoves).
- Builds a checks-only tree:
    * "force": still forces mate -> store one best defense -> recurse
    * "fail": does not force mate -> store one best defense -> leaf
    * "mate": attacker move delivers mate immediately
- Stores: fenStart (with pockets), sideToMove, mateIn (fastest attacker mate),
  principal solution SAN/UCI, pockets, and the FEN-keyed attacker tree.

Key details/fixes
- Forcedness proof is done BEFORE pushing the attacker move (correct perspective).
- Defender reply is taken from PV[1] (since PV[0] is the attacker move); if PV too short
  or mismatched, we fall back to a bestmove query from the post-attack position.
- Movetime-based probing by default for predictable runtime.

Example
  python tempo_miner.py \
    --in lichess_db_crazyhouse_rated_2025-09.pgn \
    --out crazy_tempo.jsonl --preview zh_tempo_preview.csv \
    --tail-plies 16 --sample 50 \
    --verify-engine "$(which fairy-stockfish)" \
    --movetime-ms 800 --engine-timeout 4 \
    --max-mate-in 9 --max-nodes 3000 \
    --per-puzzle-timeout-sec 20 \
    --progress-every-games 10
"""

import sys, io, csv, json, argparse, random, subprocess, time
import chess, chess.pgn, chess.variant

# -------------------- CLI --------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_jsonl", default="zh_tempo.jsonl")
    ap.add_argument("--preview", dest="out_preview", default="zh_tempo_preview.csv")

    # Candidate roots (near end of game)
    ap.add_argument("--tail-plies", type=int, default=16,
                    help="Try roots whose first ply is within this many plies of game end (0 = whole game).")

    # Sampling / caps
    ap.add_argument("--sample", type=int, default=0,
                    help="If >0, stop as soon as this many puzzles are kept (early stop).")
    ap.add_argument("--max-games", type=int, default=0,
                    help="Stop after reading this many games (0 = no limit).")
    ap.add_argument("--seed", type=int, default=0)

    # Progress printing
    ap.add_argument("--progress-every-games", type=int, default=200,
                    help="Print progress every N games (0 = disable).")
    ap.add_argument("--progress-sec", type=float, default=0.0,
                    help="Also print progress roughly every X seconds (0 = off).")

    # Engine
    ap.add_argument("--verify-engine", dest="engine_path", default="",
                    help="Path to Fairy-Stockfish (crazyhouse). Required.")
    ap.add_argument("--depth", type=int, default=0,
                    help="Depth to use if movetime is 0.")
    ap.add_argument("--movetime-ms", type=int, default=800,
                    help="Per-call movetime in ms (recommended for consistent speed).")
    ap.add_argument("--engine-timeout", type=float, default=4.0,
                    help="Max seconds to wait for each engine call before stopping it.")

    # Guard rails (reject, don't trim)
    ap.add_argument("--max-mate-in", type=int, default=9,
                    help="Reject if fastest forcing check at root exceeds this attacker ply length (0 = no cap).")
    ap.add_argument("--max-nodes", type=int, default=3000,
                    help="Reject if attacker nodes exceed this count (0 = no cap).")
    ap.add_argument("--per-puzzle-timeout-sec", type=float, default=20.0,
                    help="Reject if a single puzzle build exceeds this wall time (0 = no cap).")

    return ap.parse_args()

# -------------------- IO helpers --------------------
def open_text_stream(path):
    if path == "-":
        return io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", newline="") if hasattr(sys.stdin, "buffer") else sys.stdin
    return open(path, "r", encoding="utf-8", newline="")

def read_game_stream(fh):
    idx = 0
    while True:
        g = chess.pgn.read_game(fh)
        if g is None:
            break
        idx += 1
        yield g, idx

def extract_moves_san(game):
    moves, node = [], game
    while node.variations:
        node = node.variations[0]
        moves.append(node.san())
    return moves

# -------------------- Boards & helpers --------------------
def board_after_san_prefix(full_moves, n):
    b = chess.variant.CrazyhouseBoard()
    for san in full_moves[:n]:
        b.push(b.parse_san(san))
    return b

def legal_checking_moves(board):
    """All legal moves (incl. drops) that give check."""
    out = []
    for mv in board.legal_moves:
        board.push(mv)
        gives_check = board.is_check()
        board.pop()
        if gives_check:
            out.append(mv)
    return out

def pockets_to_maps(board):
    pw = board.pockets[chess.WHITE]
    pb = board.pockets[chess.BLACK]
    def counts(pocket):
        if hasattr(pocket, "count"):
            return {"P": pocket.count(chess.PAWN), "N": pocket.count(chess.KNIGHT),
                    "B": pocket.count(chess.BISHOP), "R": pocket.count(chess.ROOK),
                    "Q": pocket.count(chess.QUEEN)}
        attr = {"P": getattr(pocket,"pawns",0), "N": getattr(pocket,"knights",0),
                "B": getattr(pocket,"bishops",0), "R": getattr(pocket,"rooks",0),
                "Q": getattr(pocket,"queens",0)}
        try:
            return {"P": pocket[chess.PAWN], "N": pocket[chess.KNIGHT],
                    "B": pocket[chess.BISHOP], "R": pocket[chess.ROOK],
                    "Q": pocket[chess.QUEEN]}
        except Exception:
            return attr
    return counts(pw), counts(pb)

# -------------------- Engine (bounded; attacker-perspective) --------------------
class Engine:
    def __init__(self, path, timeout_sec=4.0):
        self.path, self.timeout, self.p = path, timeout_sec, None

    def __enter__(self):
        if not self.path:
            return self
        self.p = subprocess.Popen([self.path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._send("uci"); self._wait("uciok")
        self._send("setoption name UCI_Variant value crazyhouse")
        return self

    def __exit__(self, *_):
        try:
            if self.p:
                self._send("quit")
                self.p.communicate(timeout=0.2)
        except Exception:
            try: self.p.kill()
            except Exception: pass

    def _send(self, s):
        if self.p and self.p.stdin:
            self.p.stdin.write(s + "\n")
            self.p.stdin.flush()

    def _wait(self, token):
        start = time.time()
        while time.time() - start < self.timeout:
            line = self.p.stdout.readline()
            if not line:
                break
            if token in line:
                return True
        return False

    def analyze_forced_mate(self, fen, movetime_ms=0, depth=0, searchmove=None):
        """
        Returns (ok, pv_uci, mateN)
        - ok True with mateN>0 means side-to-move mates in 'mateN' plies (Stockfish semantics).
        - searchmove: restrict root to a single attacker check. MUST be called BEFORE pushing that move.
        """
        if not self.p:
            return (False, [], None)
        self._send("ucinewgame")
        self._send(f"position fen {fen}")

        if movetime_ms and movetime_ms > 0:
            go = f"go movetime {movetime_ms}"
        else:
            go = f"go depth {depth if depth>0 else 18}"
        if searchmove:
            go += f" searchmoves {searchmove}"

        best_pv, mate = [], None
        self._send(go)

        start = time.time()
        last_bestmove = None
        while True:
            if time.time() - start > self.timeout:
                self._send("stop")
                break
            line = self.p.stdout.readline()
            if not line:
                break
            s = line.strip()
            if s.startswith("info ") and " pv " in s and " score mate " in s:
                try:
                    after = s.split(" score mate ", 1)[1]
                    mate_part, pv_part = after.split(" pv ", 1)
                    m = int(mate_part.split()[0])   # mate in m plies for side-to-move
                    mate = m
                    best_pv = pv_part.split()
                except Exception:
                    pass
            elif s.startswith("bestmove"):
                parts = s.split()
                if len(parts) >= 2:
                    last_bestmove = parts[1]
                break

        if mate is None or mate <= 0:
            # No forced mate (or getting mated). Return a bestmove if any (useful for 'fail' defense).
            if not best_pv and last_bestmove:
                best_pv = [last_bestmove]
            return (False, best_pv, None)
        return (True, best_pv, mate)

    def bestmove(self, fen, movetime_ms=0, depth=0):
        """One-shot bestmove (for storing a 'fail' defense or PV fallback)."""
        if not self.p: return None
        self._send("ucinewgame")
        self._send(f"position fen {fen}")
        if movetime_ms and movetime_ms > 0:
            self._send(f"go movetime {movetime_ms}")
        else:
            self._send(f"go depth {depth if depth>0 else 18}")
        start = time.time()
        best = None
        while True:
            if time.time() - start > self.timeout:
                self._send("stop"); break
            line = self.p.stdout.readline()
            if not line: break
            s = line.strip()
            if s.startswith("bestmove"):
                parts = s.split()
                if len(parts) >= 2:
                    best = parts[1]
                break
        return best

# -------------------- Tree builder (all checks; forced proven pre-push) --------------------
class BuildBudget:
    def __init__(self, max_nodes=0, per_puzzle_timeout_sec=0.0):
        self.max_nodes = max_nodes
        self.start = time.time()
        self.timeout = per_puzzle_timeout_sec
        self.nodes = 0
    def tick_node(self):
        self.nodes += 1
        if self.max_nodes and self.nodes > self.max_nodes:
            return False
        if self.timeout and (time.time() - self.start) > self.timeout:
            return False
        return True

def pv_defense_after_attacker(mv_uci, pv_list):
    """
    Given engine PV beginning with the attacker's chosen move, return the defender reply (pv[1]).
    Return None if PV is missing/short or doesn't match the chosen move.
    """
    if not pv_list or len(pv_list) < 2:
        return None
    if pv_list[0] != mv_uci:
        return None
    return pv_list[1]

def build_tree_all_checks(start_board, eng, args):
    """
    Build an attacker-only (checks-only) tree from start_board.

    FORCEDNESS CHECK (critical):
    - For each attacker check 'mv' at position 'b', call
        eng.analyze_forced_mate(b.fen(), ..., searchmove=mv.uci())
      BEFORE pushing 'mv'. This proves/disproves a checks-only forced mate
      from the ATTACKER perspective for that specific move.

    Returns ((tree, principal_san, principal_uci, mate_in), ok) or (None, False).
    """
    def fen_key(b): return b.fen()

    tree = {}
    budget = BuildBudget(max_nodes=args.max_nodes, per_puzzle_timeout_sec=args.per_puzzle_timeout_sec)
    visited = set()

    # Root viability: at least one checking move that forces mate (pre-push)
    root_checks = legal_checking_moves(start_board)
    if not root_checks:
        return (None, False)

    forced_candidates = []
    for mv in root_checks:
        ok, pv, mateN = eng.analyze_forced_mate(
            start_board.fen(), movetime_ms=args.movetime_ms, depth=args.depth, searchmove=mv.uci()
        )
        if ok and mateN:
            forced_candidates.append((mv, mateN, pv))

    if not forced_candidates:
        return (None, False)

    forced_candidates.sort(key=lambda t: (t[1], t[0].uci()))
    fastest_mate = forced_candidates[0][1]
    if args.max_mate_in and fastest_mate > args.max_mate_in:
        return (None, False)

    # DFS expand attacker nodes
    def expand_attacker(b):
        if not budget.tick_node():
            return False
        k = fen_key(b)
        if k in visited:
            return False
        visited.add(k)

        if b.is_checkmate():
            visited.remove(k)
            return True

        checks = legal_checking_moves(b)
        if not checks:
            visited.remove(k)
            return False  # Tempo requires a check every attacker ply

        entries = []
        forced, fails = [], []

        # Classify each checking move by proving forcedness PRE-PUSH (attacker perspective)
        for mv in checks:
            ok, pv, mateN = eng.analyze_forced_mate(
                b.fen(), movetime_ms=args.movetime_ms, depth=args.depth, searchmove=mv.uci()
            )
            if ok and mateN:
                forced.append((mv, mateN, pv))
            else:
                fails.append((mv, pv))

        # Order: forced by (mate-in, uci), then fails by uci
        forced.sort(key=lambda t: (t[1], t[0].uci()))
        fails.sort(key=lambda t: t[0].uci())

        has_forced_path = False  # we will return True only if at least one branch truly forces mate

        # Expand forced branches (only commit as "force" if child proves out)
        for mv, mateN, pv in forced:
            b.push(mv)
            if b.is_checkmate():
                entries.append({"uci": mv.uci(), "class": "mate", "defense": None, "nextFen": None})
                has_forced_path = True
                b.pop()
                continue

            # Defender reply from PV[1] if PV starts with mv; otherwise ask engine
            def_uci = pv[1] if (pv and len(pv) > 1 and pv[0] == mv.uci()) else None
            if not def_uci:
                def_uci = eng.bestmove(b.fen(), movetime_ms=args.movetime_ms, depth=args.depth)
            if not def_uci:
                b.pop(); visited.remove(k); return False

            def_mv = chess.Move.from_uci(def_uci)
            if def_mv not in b.legal_moves:
                b.pop(); visited.remove(k); return False

            b.push(def_mv)
            child_k = fen_key(b)

            # Recurse: only if child proves a checks-only path to mate do we mark this edge as "force"
            if expand_attacker(b):
                entries.append({"uci": mv.uci(), "class": "force",
                                "defense": {"uci": def_uci}, "nextFen": child_k})
                has_forced_path = True
            else:
                # Engine said "mate", but not under checks-only constraint → treat as a 'fail' leaf for gameplay
                entries.append({"uci": mv.uci(), "class": "fail",
                                "defense": {"uci": def_uci}, "nextFen": child_k})

            b.pop(); b.pop()

        # Add explicit fail leaves for moves that never looked forced
        for mv, pv in fails:
            b.push(mv)
            if b.is_checkmate():
                entries.append({"uci": mv.uci(), "class": "mate", "defense": None, "nextFen": None})
                has_forced_path = True  # immediate mate is a valid terminal success
                b.pop()
                continue

            def_uci = pv[1] if (pv and len(pv) > 1 and pv[0] == mv.uci()) else None
            if not def_uci:
                def_uci = eng.bestmove(b.fen(), movetime_ms=args.movetime_ms, depth=args.depth)
            if not def_uci:
                b.pop(); visited.remove(k); return False

            def_mv = chess.Move.from_uci(def_uci)
            if def_mv not in b.legal_moves:
                b.pop(); visited.remove(k); return False

            b.push(def_mv)
            next_k = fen_key(b)
            entries.append({"uci": mv.uci(), "class": "fail",
                            "defense": {"uci": def_uci}, "nextFen": next_k})
            b.pop(); b.pop()

        tree[k] = {"attacker": entries}
        visited.remove(k)
        return has_forced_path

    # Now actually build the tree starting from the root position
    if not expand_attacker(start_board):
        return (None, False)

    # Extract principal solution from the fastest forced candidate
    fastest_mv, fastest_mate, fastest_pv = forced_candidates[0]
    
    # Convert PV to SAN notation
    principal_san = []
    principal_uci = []
    temp_board = start_board.copy()
    for uci_move in fastest_pv:
        try:
            mv = chess.Move.from_uci(uci_move)
            if mv in temp_board.legal_moves:
                principal_san.append(temp_board.san(mv))
                principal_uci.append(uci_move)
                temp_board.push(mv)
            else:
                break
        except:
            break
    
    return ((tree, principal_san, principal_uci, fastest_mate), True)

# -------------------- Difficulty Rating --------------------
def calculate_difficulty_rating(mate_in, tree, start_board, game, solution_len):
    """
    Calculate puzzle difficulty rating (800-2500 range) based on multiple factors:
    1. Mate-in distance (longer = harder)
    2. Number of alternative moves to consider (branching factor)
    3. Presence of drops in solution
    4. Average player rating from game
    5. Solution length complexity
    """
    base_rating = 1200  # Starting point
    
    # Factor 1: Mate-in distance (each ply adds difficulty)
    # Mate in 3 plies = +0, Mate in 9 plies = +300
    mate_factor = min((mate_in - 3) * 50, 400)
    
    # Factor 2: Tree complexity (branching factor)
    # Count total attacker moves across all nodes
    total_moves = 0
    forcing_moves = 0
    if tree:
        for fen_key, node_data in tree.items():
            attacker_moves = node_data.get("attacker", [])
            total_moves += len(attacker_moves)
            forcing_moves += sum(1 for m in attacker_moves if m.get("class") == "force")
    
    # More alternatives = harder puzzle
    complexity_factor = min(total_moves * 10, 300)
    
    # High ratio of forcing moves = slightly easier (clearer path)
    if total_moves > 0:
        forcing_ratio = forcing_moves / total_moves
        clarity_penalty = int((1.0 - forcing_ratio) * 100)  # Less forcing = harder
    else:
        clarity_penalty = 0
    
    # Factor 3: Drops in solution (Crazyhouse-specific complexity)
    drop_count = sum(1 for move in (solution_len or []) if '@' in str(move))
    drop_factor = drop_count * 40
    
    # Factor 4: Average player ELO (higher rated games = potentially harder positions)
    try:
        white_elo = int(game.headers.get("WhiteElo", "0"))
        black_elo = int(game.headers.get("BlackElo", "0"))
        if white_elo > 0 and black_elo > 0:
            avg_elo = (white_elo + black_elo) / 2
            # Use player ELO as a baseline adjustment (±200 range)
            elo_adjustment = int((avg_elo - 1500) * 0.2)
            elo_adjustment = max(-200, min(200, elo_adjustment))
        else:
            elo_adjustment = 0
    except:
        elo_adjustment = 0
    
    # Factor 5: Solution length
    solution_length_factor = min(solution_len * 15, 150)
    
    # Combine all factors
    difficulty = base_rating + mate_factor + complexity_factor + clarity_penalty + drop_factor + elo_adjustment + solution_length_factor
    
    # Clamp to reasonable range
    difficulty = max(800, min(2500, difficulty))
    
    return int(difficulty)

# -------------------- Miner --------------------
def make_record(game, pid, start_board, mate_in, solutionSAN, solutionUCI=None, tree=None):
    w_poc, b_poc = pockets_to_maps(start_board)
    
    # Calculate difficulty rating
    difficulty_rating = calculate_difficulty_rating(
        mate_in, tree, start_board, game, len(solutionSAN) if solutionSAN else 0
    )
    
    rec = {
        "id": pid,
        "variant": "crazyhouse",
        "fenStart": start_board.fen(),
        "sideToMove": "w" if start_board.turn==chess.WHITE else "b",
        "mateIn": int(mate_in),
        "solutionSAN": solutionSAN[:],
        "solutionUCI": solutionUCI[:] if solutionUCI else None,
        "difficultyRating": difficulty_rating,
        "whitePocket": w_poc, "blackPocket": b_poc,
        "tags": ["alwaysCheck","dropOK","fromGame","forced","fullChecks"],
        "whiteElo": game.headers.get("WhiteElo"),
        "blackElo": game.headers.get("BlackElo"),
        "result": game.headers.get("Result"),
        "site": game.headers.get("Site"),
        "date": game.headers.get("Date"),
    }
    if tree is not None:
        rec["tree"] = tree
    return rec

def make_puzzle_id(game, start_ply):
    site = game.headers.get("Site") or "unknown"
    token = site.rsplit("/", 1)[-1]
    return f"zh:{token}:ply{start_ply}"

def main():
    args = parse_args()
    if args.seed: random.seed(args.seed)

    fh = open_text_stream(args.in_path)
    out_f = open(args.out_jsonl, "w", encoding="utf-8")
    prev_f = open(args.out_preview, "w", encoding="utf-8", newline="")
    pw = csv.DictWriter(prev_f, fieldnames=["id","mateIn","sideToMove","fenStart","difficultyRating"])
    pw.writeheader()

    reservoir, accepted, games_seen = [], 0, 0

    start_time = time.time(); last_print = start_time
    def progress(force=False):
        nonlocal last_print
        out_count = len(reservoir) if (args.sample>0) else accepted
        do_print = force or (
            (args.progress_every_games and games_seen>0 and games_seen%args.progress_every_games==0) or
            (args.progress_sec and (time.time()-last_print)>=args.progress_sec)
        )
        if do_print:
            print(f"[tempo] games={games_seen:,} accepted={accepted:,} out={out_count:,}", file=sys.stderr, flush=True)
            last_print = time.time()

    if not args.engine_path:
        print("ERROR: --verify-engine is required.", file=sys.stderr); sys.exit(2)

    with Engine(args.engine_path, timeout_sec=args.engine_timeout) as eng:
        try:
            for game, _ in read_game_stream(fh):
                games_seen += 1
                progress()

                moves = extract_moves_san(game)
                if not moves:
                    if args.max_games and games_seen>=args.max_games: break
                    continue

                m = len(moves)
                start_min = max(0, m - args.tail_plies) if args.tail_plies > 0 else 0

                # Try roots one ply at a time toward the end
                for s in range(start_min, m + 1):
                    b0 = board_after_san_prefix(moves, s)

                    # Build full tree (will internally prove a forced checking move exists at root)
                    res, ok = build_tree_all_checks(b0, eng, args)
                    if not ok or res is None:
                        continue

                    tree_dict, principal_san, principal_uci, mate_in = res
                    pid = make_puzzle_id(game, s)
                    rec = make_record(game, pid, b0, mate_in, principal_san, principal_uci, tree=tree_dict)

                    accepted += 1
                    if args.sample > 0:
                        if len(reservoir) < args.sample:
                            reservoir.append(rec)
                            if len(reservoir) >= args.sample:
                                # flush and exit early
                                out_f.close(); prev_f.close()
                                with open(args.out_jsonl,"w",encoding="utf-8") as f:
                                    for r in reservoir: f.write(json.dumps(r)+"\n")
                                with open(args.out_preview,"w",encoding="utf-8",newline="") as ph:
                                    pw2=csv.DictWriter(ph, fieldnames=["id","mateIn","sideToMove","fenStart","difficultyRating"])
                                    pw2.writeheader()
                                    for r in reservoir:
                                        pw2.writerow({"id":r["id"],"mateIn":r["mateIn"],
                                                      "sideToMove":r["sideToMove"],"fenStart":r["fenStart"],
                                                      "difficultyRating":r["difficultyRating"]})
                                print(f"Done. Games read: {games_seen:,} | Puzzles found: {accepted:,}", file=sys.stderr)
                                print(f"[tempo] games={games_seen:,} accepted={accepted:,} out={len(reservoir):,}", file=sys.stderr)
                                sys.exit(0)
                    else:
                        out_f.write(json.dumps(rec) + "\n")
                        pw.writerow({"id": rec["id"], "mateIn": rec["mateIn"],
                                     "sideToMove": rec["sideToMove"], "fenStart": rec["fenStart"],
                                     "difficultyRating": rec["difficultyRating"]})
                    break  # at most one puzzle per game for speed

                if args.max_games and games_seen >= args.max_games:
                    break

        finally:
            try: fh.close(); out_f.close(); prev_f.close()
            except: pass

    # EOF without early stop: flush whatever we have (if sampling)
    if args.sample > 0:
        with open(args.out_jsonl, "w", encoding="utf-8") as f:
            for r in reservoir: f.write(json.dumps(r) + "\n")
        with open(args.out_preview, "w", encoding="utf-8", newline="") as ph:
            pw2=csv.DictWriter(ph, fieldnames=["id","mateIn","sideToMove","fenStart","difficultyRating"])
            pw2.writeheader()
            for r in reservoir:
                pw2.writerow({"id":r["id"],"mateIn":r["mateIn"],
                              "sideToMove":r["sideToMove"],"fenStart":r["fenStart"],
                              "difficultyRating":r["difficultyRating"]})

    print(f"Done. Games read: {games_seen:,} | Puzzles found: {accepted:,}", file=sys.stderr)
    out_count = len(reservoir) if (args.sample>0) else accepted
    print(f"[tempo] games={games_seen:,} accepted={accepted:,} out={out_count:,}", file=sys.stderr)

if __name__ == "__main__":
    main()
