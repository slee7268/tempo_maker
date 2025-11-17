#!/usr/bin/env python3
"""
Test script for new tempo_miner.py features:
- Theme detection
- Difficulty rating v2
- Fun score calculation
"""

import sys
import chess
import chess.pgn
from dataclasses import dataclass
from typing import List

# Import the new functions
sys.path.insert(0, '/home/runner/work/tempo_maker/tempo_maker')
from tempo_miner import (
    detect_themes, 
    calculate_fun_score, 
    calculate_difficulty_rating_v2,
    Node,
    EngineMove
)


class MockGame:
    def __init__(self):
        self.headers = {
            "WhiteElo": "1800",
            "BlackElo": "1750"
        }


def test_theme_detection():
    """Test theme detection with various positions."""
    print("=" * 60)
    print("TEST 1: Theme Detection")
    print("=" * 60)
    
    # Test back rank mate
    print("\n1. Testing back-rank mate detection...")
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    solution = [chess.Move.from_uci("a1a8")]
    final_board = board.copy()
    final_board.push(solution[0])
    
    themes = detect_themes(board, solution, final_board)
    print(f"   Themes detected: {themes}")
    assert "back_rank" in themes, "Should detect back-rank mate"
    print("   ✓ Back-rank mate detected correctly")
    
    # Test sacrifice
    print("\n2. Testing sacrifice detection...")
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1")
    solution = [
        chess.Move.from_uci("c4f7"),  # Bxf7+ (sacrifice)
        chess.Move.from_uci("e8f7"),  # Kxf7
    ]
    final_board = board.copy()
    for move in solution:
        final_board.push(move)
    
    themes = detect_themes(board, solution, final_board)
    print(f"   Themes detected: {themes}")
    assert "sacrifice" in themes, "Should detect sacrifice"
    print("   ✓ Sacrifice detected correctly")
    
    # Test promotion
    print("\n3. Testing promotion detection...")
    board = chess.Board("8/P7/8/8/8/8/8/4K2k w - - 0 1")
    solution = [chess.Move.from_uci("a7a8q")]  # Promotion to queen
    final_board = board.copy()
    final_board.push(solution[0])
    
    themes = detect_themes(board, solution, final_board)
    print(f"   Themes detected: {themes}")
    assert "promotion" in themes, "Should detect promotion"
    print("   ✓ Promotion detected correctly")
    
    print("\n✓ Theme detection tests passed!")
    return True


def test_fun_score():
    """Test fun score calculation."""
    print("\n" + "=" * 60)
    print("TEST 2: Fun Score Calculation")
    print("=" * 60)
    
    # Test 1: Simple puzzle with no special themes
    print("\n1. Testing simple puzzle (no themes)...")
    score = calculate_fun_score([], 0, 3, 2.0)
    print(f"   Fun score: {score:.2f}")
    assert 0 <= score <= 10, "Score should be in 0-10 range"
    print(f"   ✓ Score in valid range: {score:.2f}")
    
    # Test 2: Exciting puzzle with queen sacrifice
    print("\n2. Testing exciting puzzle (queen sacrifice + smothered)...")
    score = calculate_fun_score(["queen_sac", "smothered"], 9, 5, 2.5)
    print(f"   Fun score: {score:.2f}")
    assert score > 5, "Score should be high for exciting puzzle"
    print(f"   ✓ High score for exciting puzzle: {score:.2f}")
    
    # Test 3: Complex puzzle (high branching penalty)
    print("\n3. Testing complex puzzle (high branching)...")
    score = calculate_fun_score(["back_rank"], 0, 7, 8.0)
    print(f"   Fun score: {score:.2f}")
    print(f"   ✓ Score calculated with complexity penalty: {score:.2f}")
    
    # Test 4: Maximum score cap
    print("\n4. Testing score cap...")
    score = calculate_fun_score(
        ["queen_sac", "smothered", "double_check", "drop_mate"], 
        15, 4, 2.0
    )
    print(f"   Fun score: {score:.2f}")
    assert score <= 10, "Score should not exceed 10"
    print(f"   ✓ Score capped at maximum: {score:.2f}")
    
    print("\n✓ Fun score tests passed!")
    return True


def test_difficulty_v2():
    """Test difficulty rating v2 calculation."""
    print("\n" + "=" * 60)
    print("TEST 3: Difficulty Rating V2")
    print("=" * 60)
    
    # Create mock solution nodes
    print("\n1. Creating mock solution nodes...")
    
    # Simple puzzle (mate in 3)
    nodes_simple = []
    for i in range(3):
        node = Node(fen=f"fen_{i}")
        node.to_move = "attacker" if i % 2 == 0 else "defender"
        node.best_move = chess.Move.from_uci("e2e4")
        node.engine_moves = [
            EngineMove(chess.Move.from_uci("e2e4"), 1000),
            EngineMove(chess.Move.from_uci("d2d4"), 500),
        ]
        node.checking_moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
        node.legal_moves = [chess.Move.from_uci("e7e5"), chess.Move.from_uci("d7d5")]
        nodes_simple.append(node)
    
    board = chess.Board()
    game = MockGame()
    
    rating = calculate_difficulty_rating_v2(3, None, board, game, nodes_simple)
    print(f"   Simple puzzle (mate in 3): {rating}")
    assert 800 <= rating <= 2500, "Rating should be in valid range"
    print(f"   ✓ Rating in valid range: {rating}")
    
    # Complex puzzle (mate in 7 with many variations)
    print("\n2. Testing complex puzzle...")
    nodes_complex = []
    valid_moves = ["e2e4", "d2d4", "f2f4", "g1f3", "b1c3"]
    for i in range(7):
        node = Node(fen=f"fen_{i}")
        node.to_move = "attacker" if i % 2 == 0 else "defender"
        node.best_move = chess.Move.from_uci(valid_moves[0])
        node.engine_moves = [
            EngineMove(chess.Move.from_uci(valid_moves[0]), 2000),
            EngineMove(chess.Move.from_uci(valid_moves[1]), 1800),
        ]
        # More checking moves for complexity
        node.checking_moves = [
            chess.Move.from_uci(valid_moves[0]), 
            chess.Move.from_uci(valid_moves[1]),
            chess.Move.from_uci(valid_moves[2]),
        ]
        node.legal_moves = [chess.Move.from_uci(m) for m in valid_moves]
        nodes_complex.append(node)
    
    rating_complex = calculate_difficulty_rating_v2(7, None, board, game, nodes_complex)
    print(f"   Complex puzzle (mate in 7): {rating_complex}")
    assert rating_complex > rating, "Complex puzzle should have higher rating"
    print(f"   ✓ Complex puzzle has higher rating: {rating_complex} > {rating}")
    
    print("\n✓ Difficulty rating v2 tests passed!")
    return True


def run_all_tests():
    """Run all test suites."""
    print("\n" + "=" * 60)
    print("RUNNING ALL TESTS FOR NEW FEATURES")
    print("=" * 60)
    
    try:
        test_theme_detection()
        test_fun_score()
        test_difficulty_v2()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
