import gui
import chess
from flask import Flask

'''
RUN SERVER FROM COMMAND LINE:
    export FLASK_APP="play_in_browser"
    flask run
'''

app = Flask(__name__)
engine_pt = "nnet_mcts.pt"
encd_pt = "autoencoderftest2.pt"
board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
nodes = 50

gui.start_server(app, engine_pt, encd_pt, board, nodes)