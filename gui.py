import chess
import chess.svg
import net
import torch
import autoencoder
import mctsAZ
from flask import Flask, request

app, engine_pt, encd_pt, chessboard, nodes = None, None, None, None, None

def start_server(*args):    
    page = lambda x: f"<html><body><div style='margin: 100px 100px 100px 100px;'> {str(x)} </div><div style='margin-left: 225px;'><p>Input move in SAN notation:</p><form method='POST'><input name='text' type='text'></form><p>Or let engine generate a move:</p><form method='POST'><input name='button' type='submit' value='Engine move'></form></div></body></html>"
    global app, engine_pt, encd_pt, chessboard, nodes 
    
    app = args[0]
    engine_pt = args[1]
    encd_pt = args[2]
    chessboard = args[3]
    nodes = args[4]
    
    @app.route('/')
    def print_board():
        svg = chess.svg.board(chessboard, size=500)

        return page(svg)

    @app.route('/new')
    def new_game():
        global chessboard
        chessboard = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        svg = chess.svg.board(chessboard, size=500)

        return page(svg)

    @app.route('/', methods=['POST'])
    @app.route('/new', methods=['POST'])
    def play_move():
        global chessboard

        if request.form.get('button'):
            nnet = net.Net().cuda()
            nnet.load_state_dict(torch.load(engine_pt))
            nnet.eval()

            encoder = autoencoder.autoencoder().cuda()
            encoder.load_state_dict(torch.load(encd_pt))
            encoder.eval()

            mcts = mctsAZ.Mcts(chessboard, nnet, encoder)
            result = mcts.search(nodes)
            chessboard = result.state
            svg = chess.svg.board(chessboard, size=500)
        elif request.form.get('text'):
            text = request.form['text']
            try:
                chessboard.push_san(text)
            except ValueError:
                pass
            svg = chess.svg.board(chessboard, size=500)

        return page(svg)

    return app
