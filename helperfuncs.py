import chess
import torch


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
    move1 = move[0].tolist()
    move2 = move[1].tolist()
    
    rank = {'a': 0,'b': 1,'c': 2,'d': 3,'e': 4,'f': 5,'g': 6,'h': 7}
    file = {'1': 7, '2': 6, '3': 5, '4': 4, '5': 3, '6': 2, '7': 1, '8': 0}

    legalMoves = move_list(board)
    
    for move in legalMoves:
        m = str(move)
        
        sq_from = move1[rank[m[0]] + 8 * file[m[1]]]
        sq_to = move2[rank[m[2]] + 8 * file[m[3]]]
        
        distr.append([move, sq_from * sq_to])
        
    return distr