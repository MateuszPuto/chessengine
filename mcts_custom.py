import bitboards
import autoencoder
import helperfuncs
import chess
import math
import random
import copy

cParam = 2
cLog = math.log(cParam)

class Fraction:
    def __init__(self, x, y, av):
        self.av = av
        self.x = x
        self.y = y
    
    def value(self):
        return self.x / self.y
    
    def modify_by_delta(self, dx):
        delta = cLog * dx / (self.av * pow(self.av, 2))
        self.x += delta
        self.y += delta

def evaluate(position, nn, encoder):
    embedding = encoder.encode(bitboards.bitboard_to_cnn_input(bitboards.bitboard(position)).unsqueeze(0).cuda()).view(1, -1)
    return nn(embedding.cuda())
    
class Node:
    def __init__(self, parentNode, position):
        self.parentNode = parentNode
        self.childNodes = []
        self.expanded = 0

        self.state = position
        self.moves = position.legal_moves
        
        self.actionValue = None
        self.choiceProbability = 1
        
        self.evaluation = None
        self.priorProbability = None
        
    def set_value(self, value):
        self.evalution = value
        self.actionValue = value
        
    def set_prior(self, policy):
        self.priorProbability = helperfuncs.probability_distribution(policy, self.state)
        
    def z(self, value):
        return 1/(-math.log(value, cParam))
        
    def set_choiceProbability(self, weights, actionValues, sumOfWeights):
        self.choiceProbability = []
        for i in range(len(weights)):
            self.choiceProbability.append(Fraction(weights[i], sumOfWeights, actionValues[i]))
        
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
    
        ##choose by highest actionValue for now
        bestNode, value = None, 0
        for child in self.root.childNodes:
            if child.actionValue and child.actionValue > value:
                bestNode = child
                value = child.actionValue
                
        return bestNode
    
    def choose(self):
        n = self.root
        
        while len(n.childNodes) > 0:
            branching = len(n.childNodes)
            weights = n.choiceProbability
            
            if weights == 1:
                if n.expanded == len(n.childNodes):
                    weights, actionValues, sumOfWeights = [], [], 0

                    for i in range(branching):
                        weight = n.z(n.childNodes[i].actionValue)
                        weights.append(weight)
                        actionValues.append(n.childNodes[i].actionValue)
                        sumOfWeights += weight

                    n.set_choiceProbability(weights, actionValues, sumOfWeights)
                else:
                    weights = [x for _, x in n.priorProbability]
            else:
                  weights = [x.value() for x in weights]
              
                 
            n = n.childNodes[random.choices(list(range(branching)), weights)[0]]
            
        return n
    
    def expand(self, n):
        moves = n.moves
        
        value, policy = evaluate(n.state, self.nnet, self.encoder)
        n.set_value(value)
        n.set_prior(policy)
        if n.parentNode != None:
            n.parentNode.expanded += 1
        
        for move in moves:
            board = copy.deepcopy(n.state)
            board.push(move)
                
            newNode = Node(n, board)
            n.add(newNode)
    
        return n
    
    def backup(self, n):
        while n.parentNode != None:
            probability = n.choiceProbability
            val = 1 - n.actionValue ##adversarial search
            n = n.parentNode
            
            delta = probability * (val - n.actionValue)
            n.actionValue += delta
            if type(n.choiceProbability) != int:
                ##kidof terrible programming
                n.choiceProbability[0].modify_by_delta(delta)
    