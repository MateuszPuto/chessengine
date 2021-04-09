import bitboards
import autoencoder
import chess
import math
import copy
import operator

class Node:
    def __init__(self, position):
        self.state = position
        self.childNodes = [] 
        
    def get_children(self):
        moves = self.state.legal_moves
        for move in moves:
            board = copy.deepcopy(self.state)
            board.push(move)
            self.childNodes.append(Node(board))
    
    def get_bitboard(self):
        encoder = autoencoder.autoencoder()
        return encoder.encode(bitboards.bitboard_to_cnn_input(bitboards.bitboard(self.state)).unsqueeze(0)).view(1, -1)

    
    
class Score:
    def __init__(self, value, node):
        self.value = value
        self.node = node
    
    def get_val(self):
        return self.value
    
    def get_node(self):
        return self.node
    
    def __gt__(self, other):
        return self.value > other.value
        
def alphabeta(node, depth, alpha, beta, valueNet):
    '''Basic alpha-beta search procedure'''
    
    if depth == 0 or node.state.is_game_over():
        value, policy = valueNet(node.get_bitboard())
        
        return Score(value, node)

    node.get_children()
    
    score = Score(-math.inf, node)
    for child in node.childNodes:
        currScore = Score(-alphabeta(child, depth-1, -beta, -alpha, valueNet).get_val(), child)
        if currScore > score: score = currScore
        alpha = max(alpha, score.get_val())
            
        if alpha >= beta:
            return Score(beta, None) #fail hard beta-cutoff
            
    return Score(score.get_val(), score.get_node())