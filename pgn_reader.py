import chess.pgn
import bitboards
import copy

pgn = open("/home/mputo/CCRL/CCRL-4040.pgn")

def next_game():
    """Returns next game in a pgn file"""
    return chess.pgn.read_game(pgn)

def game_moves(game):
    """ Returns list of uci moves in a given game """
    moves = []
    for move in game.mainline_moves():
        moves.append(move)

    return moves

def get_dataset(size):
    """Generates simple dataset for autoencoder training with size equal or greater to specified 'size'"""
    dataset = []
    while len(dataset) < size:
        board = chess.Board()
        game = next_game()
        if game == None:
            dataset = None
            break
            
        moves = game_moves(game)
        for move in moves:
            board.push(move)
            cnn = bitboards.bitboard_to_cnn_input(bitboards.bitboard(copy.deepcopy(board)))
            dataset.append(cnn)
        
    return dataset