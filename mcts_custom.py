import bitboards
import autoencoder
import helperfuncs
import chess
import math
import random
import copy

cParam = 2
scale = 0.8
minimalActionValue = 0.001

cLog = math.log(cParam)
normParam = 1/(-math.log(scale, cParam))

def z(x):
    '''Assuming the input is number between 0 and 1'''
    return 1/(-normParam * math.log(x*scale, cParam))

class Fraction:
    def __init__(self, x, y, av):
        self.x = x
        self.y = y
        self.av = av
    
    def value(self, i):
        return self.x[i] / self.y
    
    def modify_by_delta(self, i, actionValue):
        actionValue = max(actionValue, minimalActionValue)
        delta = z(actionValue) - self.x[i]

        self.x[i] += delta
        self.y += delta
        self.av[i] = actionValue

def evaluate(position, nn, encoder):
    embedding = encoder.encode(bitboards.bitboard_to_cnn_input(bitboards.bitboard(position)).unsqueeze(0).cuda()).view(1, -1)
    return nn(embedding.cuda())
    
class Node:
    def __init__(self, parentNode, position, nodeNum):
        self.parentNode = parentNode
        self.childNodes = []
        self.nodeNum = nodeNum
        self.visits = 0

        self.state = position
        self.moves = position.legal_moves
        
        self.actionValue = None
        self.choiceProbability = 1
        
        self.evaluation = None
        self.priorProbability = None
        
    def set_value(self, value):
        self.evaluation = value
        self.actionValue = value
        
    def set_prior(self, policy):
        self.priorProbability = helperfuncs.probability_distribution(policy, self.state)
        
    def set_choiceProbability(self, weights, sumOfWeights, actionValues):
        self.choiceProbability = Fraction(weights, sumOfWeights, actionValues)
        
    def add(self, node):
        self.childNodes.append(node)
        
class Mcts:
    def __init__(self, state, nn, encoder):
        self.root = Node(None, state, 0)
        self.nnet = nn
        self.encoder = encoder
    
    def search(self, rollouts):
        for rollout in range(rollouts):
            node = self.choose()
            node = self.expand(node)
            self.backup(node)
    
        bestNode, value = None, 0
        
        for child in self.root.childNodes:
            ##print(self.root.choiceProbability.value(child.nodeNum), child.actionValue, child.visits)
            if child.actionValue and child.actionValue > value:
                bestNode = child
                value = child.actionValue
                
        return bestNode
    
    def choose(self):
        n = self.root
        
        while len(n.childNodes) > 0:
            branching = len(n.childNodes)
            choiceProb = n.choiceProbability
                 
            n = n.childNodes[random.choices(list(range(branching)), [choiceProb.value(i) for i in range(len(choiceProb.x))])[0]]
            
        return n
    
    def expand(self, n):
        moves = n.moves
        
        value, policy = evaluate(n.state, self.nnet, self.encoder)
        n.set_value(value.item())
        n.set_prior(policy)
        
        for i, move in enumerate(moves):
            board = copy.deepcopy(n.state)
            board.push(move)
            
            newNode = Node(n, board, i)
            n.add(newNode)
            
        weights = [prior[1] for prior in n.priorProbability]
        sumOfWeights = sum(weights)
        
        n.set_choiceProbability(list(map(lambda x: x/sumOfWeights, weights)), 1, [n.actionValue for weight in weights])
        
        return n
    
    def backup(self, n):
        while n.parentNode != None:
            val = 1 - n.actionValue
            nnum = n.nodeNum 
            
            n = n.parentNode
            probability = n.choiceProbability
            
            delta = val - n.actionValue
            n.actionValue += probability.value(nnum) * delta
            probability.modify_by_delta(nnum, n.actionValue)
            n.visits += 1