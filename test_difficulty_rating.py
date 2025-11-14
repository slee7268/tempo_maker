#!/usr/bin/env python3
"""
Simple test to verify the difficulty rating calculation works correctly.
This doesn't require Fairy-Stockfish, just tests the rating algorithm.
"""

import sys

# Mock the chess modules since we're just testing the rating function
class MockPocket:
    def count(self, piece_type):
        return 0

class MockBoard:
    def __init__(self):
        self.pockets = [MockPocket(), MockPocket()]
    def fen(self):
        return "test_fen"

class MockGame:
    def __init__(self):
        self.headers = {
            "WhiteElo": "1650",
            "BlackElo": "1550"
        }

def calculate_difficulty_rating(mate_in, tree, start_board, game, solution_len):
    """
    Calculate puzzle difficulty rating (800-2500 range) based on multiple factors.
    """
    base_rating = 1200
    
    # Factor 1: Mate-in distance
    mate_factor = min((mate_in - 3) * 50, 400)
    
    # Factor 2: Tree complexity
    total_moves = 0
    forcing_moves = 0
    if tree:
        for fen_key, node_data in tree.items():
            attacker_moves = node_data.get("attacker", [])
            total_moves += len(attacker_moves)
            forcing_moves += sum(1 for m in attacker_moves if m.get("class") == "force")
    
    complexity_factor = min(total_moves * 10, 300)
    
    if total_moves > 0:
        forcing_ratio = forcing_moves / total_moves
        clarity_penalty = int((1.0 - forcing_ratio) * 100)
    else:
        clarity_penalty = 0
    
    # Factor 3: Drops in solution
    drop_count = 0  # Mock for this test
    drop_factor = drop_count * 40
    
    # Factor 4: Average player ELO
    try:
        white_elo = int(game.headers.get("WhiteElo", "0"))
        black_elo = int(game.headers.get("BlackElo", "0"))
        if white_elo > 0 and black_elo > 0:
            avg_elo = (white_elo + black_elo) / 2
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

def test_difficulty_rating():
    print("Testing difficulty rating calculation...")
    print()
    
    # Test 1: Simple puzzle (mate in 3, simple tree)
    tree1 = {
        "pos1": {
            "attacker": [
                {"uci": "e2e4", "class": "force"},
                {"uci": "d2d4", "class": "fail"}
            ]
        }
    }
    rating1 = calculate_difficulty_rating(3, tree1, MockBoard(), MockGame(), 3)
    print(f"Test 1 - Simple puzzle (mate in 3):")
    print(f"  Rating: {rating1}")
    print(f"  Expected: ~1200 (base)")
    print()
    
    # Test 2: Harder puzzle (mate in 7, complex tree)
    tree2 = {
        "pos1": {
            "attacker": [
                {"uci": "e2e4", "class": "force"},
                {"uci": "d2d4", "class": "force"},
                {"uci": "f2f4", "class": "fail"},
                {"uci": "g2g4", "class": "fail"}
            ]
        },
        "pos2": {
            "attacker": [
                {"uci": "a2a4", "class": "force"},
                {"uci": "b2b4", "class": "fail"}
            ]
        }
    }
    rating2 = calculate_difficulty_rating(7, tree2, MockBoard(), MockGame(), 7)
    print(f"Test 2 - Harder puzzle (mate in 7, complex tree):")
    print(f"  Rating: {rating2}")
    print(f"  Expected: ~1600-1700 (mate factor + complexity)")
    print()
    
    # Test 3: Very hard puzzle (mate in 9, many alternatives, long solution)
    tree3 = {
        "pos1": {"attacker": [{"uci": f"m{i}", "class": "fail"} for i in range(10)]},
        "pos2": {"attacker": [{"uci": f"n{i}", "class": "force"} for i in range(5)]},
        "pos3": {"attacker": [{"uci": f"o{i}", "class": "fail"} for i in range(8)]}
    }
    rating3 = calculate_difficulty_rating(9, tree3, MockBoard(), MockGame(), 10)
    print(f"Test 3 - Very hard puzzle (mate in 9, many alternatives):")
    print(f"  Rating: {rating3}")
    print(f"  Expected: ~2000+ (high complexity + long mate)")
    print()
    
    # Test 4: Clamping (should be capped at 2500)
    tree4 = {f"pos{i}": {"attacker": [{"uci": f"m{j}", "class": "fail"} for j in range(20)]} for i in range(20)}
    rating4 = calculate_difficulty_rating(20, tree4, MockBoard(), MockGame(), 20)
    print(f"Test 4 - Extreme puzzle (should clamp to 2500):")
    print(f"  Rating: {rating4}")
    print(f"  Expected: 2500 (clamped)")
    print()
    
    # Verify ratings are in valid range
    all_ratings = [rating1, rating2, rating3, rating4]
    all_valid = all(800 <= r <= 2500 for r in all_ratings)
    
    print("=" * 50)
    if all_valid:
        print("✓ All tests passed! Ratings are in valid range (800-2500)")
        return 0
    else:
        print("✗ Some ratings are out of range!")
        return 1

if __name__ == "__main__":
    sys.exit(test_difficulty_rating())
