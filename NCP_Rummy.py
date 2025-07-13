import random
import math
from typing import List, Tuple, Set, Dict, Optional
from functools import lru_cache

# --- Constants ---
SUITS = ['♠', '♥', '♦', '♣']
RANKS = list(range(1, 14))  # 1=Ace, 13=King

# --- Core Data Structures ---
Card = Tuple[int, str]  # (rank, suit)

def generate_deck() -> List[Card]:
    return [(r, s) for r in RANKS for s in SUITS]

def card_value(rank: int) -> int:
    return min(rank, 10)

@lru_cache(maxsize=10000)
def find_all_melds(hand_tuple: Tuple[Card, ...]) -> List[Set[Card]]:
    hand = list(hand_tuple)
    melds = []

    # Enhanced set detection with Ace-as-14
    rank_groups: Dict[int, List[Card]] = {}
    for r, s in hand:
        rank_groups.setdefault(r, []).append((r, s))
        if r == 1:
            rank_groups.setdefault(14, []).append((r, s))

    for rank, cards in rank_groups.items():
        if len(cards) >= 3:
            melds.append(set((r if r != 14 else 1, s) for (r, s) in cards))

    # Enhanced sequence detection with Ace-high/low
    suit_groups: Dict[str, List[int]] = {}
    for r, s in hand:
        suit_groups.setdefault(s, []).append(r)
        if r == 1:
            suit_groups[s].append(14)

    for suit, ranks in suit_groups.items():
        sorted_ranks = sorted(set(ranks))
        for i in range(len(sorted_ranks)):
            current_rank = sorted_ranks[i]
            if current_rank in [13, 14] and 1 in sorted_ranks and 2 in sorted_ranks:
                seq = [
                    (13 if current_rank == 14 else current_rank, suit),
                    (1, suit),
                    (2, suit)
                ]
                if all(c in hand for c in seq):
                    melds.append(set(seq))
            if i <= len(sorted_ranks) - 3:
                if sorted_ranks[i+2] == sorted_ranks[i] + 2:
                    seq = [
                        (sorted_ranks[i] if sorted_ranks[i] != 14 else 1, suit),
                        (sorted_ranks[i+1], suit),
                        (sorted_ranks[i+2] if sorted_ranks[i+2] != 14 else 1, suit)
                    ]
                    if all(c in hand for c in seq):
                        melds.append(set(seq))

    # Remove duplicates
    unique_melds = []
    seen = set()
    for m in melds:
        frozen = frozenset(m)
        if frozen not in seen:
            seen.add(frozen)
            unique_melds.append(m)
    return unique_melds

def compute_deadwood(hand: List[Card], melds: List[Set[Card]]) -> List[Card]:
    used = set(card for meld in melds for card in meld)
    return [c for c in hand if c not in used]

def predict_opponent_melds(discards: List[Card]) -> Dict[Card, float]:
    model = {}
    for rank, suit in discards:
        for s in SUITS:
            model[(rank, s)] = model.get((rank, s), 0) + 1.5
        for offset in [-1, 1]:
            adj = rank + offset
            if 1 <= adj <= 13:
                model[(adj, suit)] = model.get((adj, suit), 0) + 1.0
    return model

def energy(hand: List[Card], opponent_model: Dict[Card, float], game_state: Dict) -> float:
    melds = find_all_melds(tuple(hand))
    deadwood = compute_deadwood(hand, melds)
    deadwood_value = sum(card_value(c[0]) for c in deadwood)
    risk = sum(opponent_model.get(c, 0) for c in deadwood)
    phase = len(game_state.get('discards', [])) / 52.0
    return deadwood_value * (1 + phase) + risk * (3 - phase)

def simulated_annealing(hand: List[Card], game_state: Dict) -> List[Card]:
    current = hand.copy()
    opponent_model = predict_opponent_melds(game_state['discards'])
    best = current.copy()
    best_energy = energy(current, opponent_model, game_state)
    temp = 2500.0

    for _ in range(5000):
        deadwood = compute_deadwood(current, find_all_melds(tuple(current)))
        if deadwood:
            i = current.index(max(deadwood, key=lambda c: card_value(c[0])))
            candidate_indices = [idx for idx, c in enumerate(current) if c not in deadwood]
            if not candidate_indices:
                candidate_indices = list(range(len(current)))
            j = random.choice(candidate_indices)
        else:
            i, j = random.sample(range(len(current)), 2)

        neighbor = current.copy()
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        new_energy = energy(neighbor, opponent_model, game_state)
        if new_energy < best_energy or random.random() < math.exp((best_energy - new_energy)/temp):
            current = neighbor.copy()
            best_energy = new_energy
            best = current.copy()

        temp *= 0.997

    return best

class ABPResult:
    def __init__(self, score: float, move: Optional[Card]):
        self.score = score
        self.move = move

transposition = {}

def alpha_beta_search(hand: List[Card], game_state: Dict, depth: int = 4,
                      alpha: float = -float('inf'), beta: float = float('inf'), maximizing: bool = True) -> ABPResult:
    key = tuple(sorted(hand)) + (depth, maximizing)
    if key in transposition:
        return transposition[key]

    if depth == 0 or len(hand) < 6:
        current_score = -energy(hand, predict_opponent_melds(game_state['discards']), game_state)
        return ABPResult(current_score, None)

    best_score = -float('inf') if maximizing else float('inf')
    best_move = None
    deadwood = compute_deadwood(hand, find_all_melds(tuple(hand)))
    move_candidates = sorted(deadwood, key=lambda c: -card_value(c[0])) if deadwood else hand

    for card in move_candidates[:5]:
        new_hand = [c for c in hand if c != card]
        opponent_hand = new_hand.copy()
        opponent_melds = find_all_melds(tuple(opponent_hand))
        opponent_deadwood = compute_deadwood(opponent_hand, opponent_melds)
        if opponent_deadwood:
            opponent_discard = max(opponent_deadwood, key=lambda c: card_value(c[0]))
            opponent_hand.remove(opponent_discard)

        result = alpha_beta_search(opponent_hand, game_state, depth - 1, alpha, beta, not maximizing)

        if maximizing:
            if result.score > best_score:
                best_score, best_move = result.score, card
                alpha = max(alpha, best_score)
        else:
            if result.score < best_score:
                best_score, best_move = result.score, card
                beta = min(beta, best_score)

        if beta <= alpha:
            break

    transposition[key] = ABPResult(best_score, best_move)
    return transposition[key]

def hybrid_move(hand: List[Card], game_state: Dict) -> Card:
    optimized_hand = simulated_annealing(hand, game_state)
    result = alpha_beta_search(optimized_hand, game_state)
    return result.move if result.move else optimized_hand[0]

def card_to_str(card: Card) -> str:
    rank_map = {1: 'A', 11: 'J', 12: 'Q', 13: 'K'}
    return f"{rank_map.get(card[0], str(card[0]))}{card[1]}"

def str_to_card(card_str: str) -> Card:
    rank_map = {'A': 1, 'J': 11, 'Q': 12, 'K': 13}
    rank_part = card_str[:-1].upper()
    suit = card_str[-1]
    if rank_part in rank_map:
        return (rank_map[rank_part], suit)
    elif rank_part.isdigit():
        return (int(rank_part), suit)
    else:
        raise ValueError(f"Invalid rank '{rank_part}' in card: {card_str}")

def find_melds(hand: List[str]) -> List[List[str]]:
    melds = []
    rank_groups = {}
    for card in hand:
        rank_groups.setdefault(card[:-1], []).append(card)
    for group in rank_groups.values():
        if len(group) >= 3:
            melds.append(group[:3])
    suit_groups = {}
    for card in hand:
        suit_groups.setdefault(card[-1], []).append(card)
    for suit, cards in suit_groups.items():
        cards_sorted = sorted(cards, key=lambda c: str_to_card(c)[0])
        for i in range(len(cards_sorted) - 2):
            run = [cards_sorted[i]]
            for j in range(i + 1, len(cards_sorted)):
                current_rank = str_to_card(run[-1])[0]
                next_rank = str_to_card(cards_sorted[j])[0]
                if next_rank == current_rank + 1:
                    run.append(cards_sorted[j])
                    if len(run) >= 3:
                        melds.append(run.copy())
                else:
                    break
    unique_melds = []
    seen = set()
    for m in melds:
        key = tuple(sorted(m))
        if key not in seen:
            seen.add(key)
            unique_melds.append(m)
    return unique_melds

def traditional_move(hand: List[str]) -> str:
    melds = find_melds(hand)
    used = set()
    for meld in melds:
        used.update(meld)
    deadwood = [card for card in hand if card not in used]
    if deadwood:
        return max(deadwood, key=lambda c: card_value(str_to_card(c)[0]))
    return hand[0]
