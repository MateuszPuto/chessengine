import chess
import torch
import math


def move_list(board):
    ''' List legal moves in "board" position'''
    moves = board.legal_moves
    x = []

    for move in moves:
        if(move.uci() != '0000'):
            x.append(move.uci())

    return x

def probability_distribution(move, board):
    ''' Returns possible moves with their relative weights assigned by policy network'''
    distr = []
    move1 = move[0].squeeze(0).tolist()[0]
    move2 = move[1].squeeze(0).tolist()[0]
    
    rank = {'a': 0,'b': 1,'c': 2,'d': 3,'e': 4,'f': 5,'g': 6,'h': 7}
    file = {'1': 7, '2': 6, '3': 5, '4': 4, '5': 3, '6': 2, '7': 1, '8': 0}

    legalMoves = move_list(board)
    
    for move in legalMoves:
        m = str(move)

        sq_from = move1[rank[m[0]] + 8 * file[m[1]]]
        sq_to = move2[rank[m[2]] + 8 * file[m[3]]]
        
        distr.append([move, sq_from * sq_to])
        
    return distr

def policy_from_probability(distr):
    '''Takes list of move probabilities, elements in format [move, probability] and returns respective policy
    in the format of "from" and "to" matrix each of size 64x64. Uses square root to perform decomposition.'''
    policy = [[0 for i in range(64)], [0 for i in range(64)]]
    
    rank = {'a': 0,'b': 1,'c': 2,'d': 3,'e': 4,'f': 5,'g': 6,'h': 7}
    file = {'1': 7, '2': 6, '3': 5, '4': 4, '5': 3, '6': 2, '7': 1, '8': 0}
    
    for square, val in distr:
        pos_from = rank[square[0]] + 8 * file[square[1]]
        pos_to = rank[square[2]] + 8 * file[square[3]]
        
        policy[0][pos_from] += math.sqrt(val)
        policy[1][pos_to] += math.sqrt(val)
        
    #normalization procedure
    p1_sum = sum(policy[0])
    for i, elem in enumerate(policy[0]):
            policy[0][i] = elem / (p1_sum + 0.001)
            
    p2_sum = sum(policy[1])
    for i, elem in enumerate(policy[1]):
            policy[1][i] = elem / (p2_sum + 0.001)
                
    return torch.Tensor(policy)