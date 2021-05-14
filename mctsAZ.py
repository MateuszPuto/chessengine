import bitboards
import autoencoder
import helperfuncs
import chess
import math
import copy

Cpuct = 5

def evaluate(position, nn, encoder):
    embedding = encoder.encode(bitboards.bitboard_to_cnn_input(bitboards.bitboard(position)).unsqueeze(0).cuda()).view(1, -1)
    return nn(embedding.cuda().unsqueeze(0))
    
class Node:
    def __init__(self, parentNode, position):
        self.parentNode = parentNode
        self.childNodes = []

        self.state = position
        self.moves = position.legal_moves
        self.noVisits = 0
        self.actionValue = 0
        self.priorProbability = None
        self.evaluation = None
        
    def set_value(self, value):
        self.evaluation = value
        
    def set_prior(self, policy):
        self.priorProbability = helperfuncs.probability_distribution(policy, self.state)
        
    def get_uct(self, i):
        return self.childNodes[i].actionValue + Cpuct * self.priorProbability[i][1] * math.sqrt(self.noVisits) / (1 + self.childNodes[i].noVisits)
    
    def add(self, node):
        self.childNodes.append(node)
        
class Mcts:
    def __init__(self, state, nn, encoder):
        self.root = Node(None, state)
        self.nnet = nn
        self.encoder = encoder
    
    def search(self, rollouts):
        for rollout in range(rollouts):
            node = self.choose()
            node = self.expand(node)
            self.backup(node)
           
        bestNode, visits = None, 0
        for child in self.root.childNodes:
            if child.noVisits > visits:
                bestNode = child
                visits = child.noVisits
                
        return bestNode
    
    def choose(self):
        n = self.root
        
        while len(n.childNodes) > 0:
            branching = len(n.childNodes)
            n = n.childNodes[sorted([[n.get_uct(i), i] for i in range(branching)], reverse=True)[0][1]]
            
        return n
    
    def expand(self, n):
        moves = n.moves
        
        value, policy = evaluate(n.state, self.nnet, self.encoder)
        n.set_value(value)
        n.set_prior(policy)
        
        for move in moves:
            board = copy.deepcopy(n.state)
            board.push(move)
                
            newNode = Node(n, board)
            n.add(newNode)
    
        return n
    
    def backup(self, n):
        val = n.evaluation.item()
        
        while n.parentNode != None:
            val = 1 - val
            n = n.parentNode
            
            n.actionValue = (n.noVisits * n.actionValue + val) / (n.noVisits + 1)
            n.noVisits += 1
    