import chess
import torch
import copy

def bitboard(board):
    '''Converts chess.Board representation of position into bitboard containing 16 planes: 12 for pieces,
    2 for castling rights, 1 for en passant and 1 for side to move. Returns list of numbers where each one
    of them should be interpreted as bitfield where each bit represents square on the chess board 
    and contains information about single piece type'''
    
    bitboard = 16 * [0x0]
    
    whitePawn = list(board.pieces(chess.PAWN, chess.WHITE))
    blackPawn = list(board.pieces(chess.PAWN, chess.BLACK))
    whiteKnight = list(board.pieces(chess.KNIGHT, chess.WHITE))
    blackKnight = list(board.pieces(chess.KNIGHT, chess.BLACK))
    whiteBishop = list(board.pieces(chess.BISHOP, chess.WHITE))
    blackBishop = list(board.pieces(chess.BISHOP, chess.BLACK))
    whiteRook = list(board.pieces(chess.ROOK, chess.WHITE))
    blackRook = list(board.pieces(chess.ROOK, chess.BLACK))
    whiteQueen = list(board.pieces(chess.QUEEN, chess.WHITE))
    blackQueen = list(board.pieces(chess.QUEEN, chess.BLACK))
    whiteKing = list(board.pieces(chess.KING, chess.WHITE))
    blackKing = list(board.pieces(chess.KING, chess.BLACK))

    for pawn in whitePawn:
        bitboard[0] += 1 << pawn
    for pawn in blackPawn:
        bitboard[1] += 1 << pawn
    for knight in whiteKnight:
        bitboard[2] += 1 << knight        
    for knight in blackKnight:
        bitboard[3] += 1 << knight
    for bishop in whiteBishop:
        bitboard[4] += 1 << bishop        
    for bishop in blackBishop:
        bitboard[5] += 1 << bishop
    for rook in whiteRook:
        bitboard[6] += 1 << rook        
    for rook in blackRook:
        bitboard[7] += 1 << rook        
    for queen in whiteQueen:
        bitboard[8] += 1 << queen        
    for queen in blackQueen:
        bitboard[9] += 1 << queen
    for king in whiteKing:
        bitboard[10] += 1 << king        
    for king in blackKing:
        bitboard[11] += 1 << king

    if(bool(board.castling_rights & chess.BB_H1)):
        bitboard[12] += 1 << 7
    if(bool(board.castling_rights & chess.BB_A1)):
        bitboard[12] += 1 << 0
        
    if(bool(board.castling_rights & chess.BB_H8)):
        bitboard[13] += 1 << 63
    if(bool(board.castling_rights & chess.BB_A8)):
        bitboard[13] += 1 << 56
        
    if(board.ep_square != None):
        bitboard[14] = 1 << board.ep_square
    else:
        bitboard[14] = 0
    
    if(board.turn == chess.BLACK):
        bitboard[15] = (1 << 64) - 1
    else:
        bitboard[15] = 0
        
    return bitboard

def bitboard_to_tensor(bitboard):
    '''Converts bitboard to pytorch tensor'''
    li = []
    
    for plane in bitboard:
        string = format(plane, 'b').rjust(64, '0')
        li += [int(x) for x in list(string)]
        
    return torch.Tensor(li)

# board = chess.Board('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1')
# print([bin(bitboard(board)[i]) for i in range(0, 16)])

# print(bitboard_to_tensor(bitboard(board)))
