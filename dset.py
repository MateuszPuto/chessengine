from enum import Enum
import torch
from torch.utils.data import Dataset
import chess
import chess.engine
from chess.engine import Cp
import helperfuncs
import bitboards
import net
import autoencoder
import alphabeta
import mctsAZ
import mcts_custom

engine = chess.engine.SimpleEngine.popen_uci("/bin/stockfish")
SearchType = Enum('SearchType', 'MINIMAX MCTS CUSTOM')
ReinforcementType = Enum('ReinforcementType', 'MC TD PARAM')
winner_to_num = {chess.WHITE: 1, chess.BLACK: 0, None: 0.5}
    
class SearchDataset(Dataset):
    def __init__(self, size, transoform, reinf, game_generator, *args):
        self.data = game_generator.get_dataset(size, reinf, *args)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
        
class GameGenerator:
    def __init__(self, max_moves, draw_cutoff, param, reinf_type):
        """
        MAX_MOVES in halfmoves
        DRAW_CUTOFF in centipawns
        PARAM number in range (0, 1) used in PARAM 'ReinforcementType (simple linear combination of TD and Monte-Carlo learning)
        """
        
        self.MAX_MOVES = max_moves
        self.DRAW_CUTOFF = draw_cutoff
        self.PARAM = param
        self.reinf_type = reinf_type

    def get_dataset(self, size, reinf, *args):
        """
        Get dataset for NN training no smaller than specified 'size'.
        Args are the 'generate_game' function parameters.
        """
        
        dataset = []

        while len(dataset) < size:
            #generate a new game
            game = self.generate_game(*args)
            
            winner, state =  -1, game[-1].state

            #score the game with engine and determine winner based on engine score and draw cutoff
            score = engine.analyse(state, chess.engine.Limit(time=1))["score"].white()
            if score > Cp(self.DRAW_CUTOFF):
                winner = chess.WHITE
            elif score < Cp(self.DRAW_CUTOFF):
                winner = chess.BLACK
            else:
                winner = None
                
            #calculate learning targets
            values = []
            
            if self.reinf_type == ReinforcementType.MC:
                for nd in game:
                    values.append(winner_to_num[winner])
                    
            elif self.reinf_type == ReinforcementType.TD:
                td_values = []
                
                for nd in game:
                    td_values.append(self.get_evaluation(args[3], nd))
                
                for i in range(len(td_values)):
                    if((i+1) == len(td_values)):
                        values.append(winner_to_num[winner])
                    else:
                        values.append(td_values[i+1])
            
            elif self.reinf_type == ReinforcementType.PARAM:
                for nd in game:
                    values.append(self.PARAM * winner_to_num[winner])
                
                td_values = []
                for nd in game:
                    td_values.append(self.get_evaluation(args[3], nd))
                
                for i in range(len(td_values)):
                    if((i+1) == len(td_values)):
                        values[i] = winner_to_num[winner]
                    else:
                        values[i] += (1 - self.PARAM) * td_values[i+1]
                
                    
            #create dataset for search type
            for i, nd in enumerate(game):
                if args[3] == SearchType.MINIMAX:
                    position = bitboards.bitboard_to_cnn_input(bitboards.bitboard(nd.get_node().state)).unsqueeze(0).cuda()
                    dataset.append([position, values[i]])

                elif args[3] == SearchType.MCTS:
                    position = bitboards.bitboard_to_cnn_input(bitboards.bitboard(nd.state)).unsqueeze(0).cuda()
                    moves = [move.uci() for move in nd.moves]
                    policy = helperfuncs.policy_from_probability([[moves[i], child.actionValue] for i, child in enumerate(nd.childNodes)])
                    dataset.append([position, values[i], policy.cuda()])

                elif args[3] == SearchType.CUSTOM:
                    position = bitboards.bitboard_to_cnn_input(bitboards.bitboard(nd.state)).unsqueeze(0).cuda()
                    moves = [move.uci() for move in nd.moves]
                    choiceProbability = nd.choiceProbability
                    policy = helperfuncs.policy_from_probability([[moves[i], choiceProbability.value(i)] for i in range(len(choiceProbability.x))])
                    dataset.append([position, values[i], policy.cuda()])

        return dataset

    def generate_game(self, board, nnet, encoder, search_tree, *args):
        """Generate the chess game given the starting 'board' position. Args depend on chosen search type. 
        Three parameters for MINIMAX: depth, lower bound and higher bound of aspiration window.
        One parameter for MCTS and CUSTOM: number of rollouts."""

        game, moves = [], 0

        while not self.stop_cond(board, moves):

            if search_tree == SearchType.MINIMAX:
                node = alphabeta.alphabeta(alphabeta.Node(board), args[0], args[1], args[2], nnet, encoder)
                board = node.get_node().state

            elif search_tree == SearchType.MCTS:
                tree = mctsAZ.Mcts(board, nnet, encoder)
                node = tree.search(args[0])
                board = node.state

            elif search_tree == SearchType.CUSTOM:
                tree = mcts_custom.Mcts(board, nnet, encoder)
                node = tree.search(args[0])
                board = node.state
                
            #tree.print_tree(2)

            game.append(node)
            moves += 1
            
        return game

    def stop_cond(self, board, moves):
        '''Stops the game when it has reached terminal position or more moves than allowed were played.'''
        end = False

        if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
            end = True
        elif board.can_claim_draw():
            end = True
        elif moves > self.MAX_MOVES:
            end = True

        return end

    
    def get_evaluation(self, search_type, node):
        '''Gets the node evaluation computed during search.'''            
        return node.evaluation