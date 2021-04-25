import bitboards
import autoencoder
import helperfuncs
import chess
import math
import random
import copy

cParam = 2
cLog = math.log(cParam)
cutoff = 0.9
normParam = 1/(-math.log(cutoff, cParam))
learningRate = 1
minimalProbability = 0.001

class Fraction:
    def __init__(self, x, y, av):
        self.av = av
        self.x = x
        self.y = y
    
    def value(self, i):
        return self.x[i] / self.y
    
    def modify_by_delta(self, i, dx):
        delta = cLog * dx / (max(self.av[i], minimalProbability) * pow(math.log(max(self.av[i], minimalProbability)), 2))
        diff = min(cutoff, self.x[i] + delta) - self.x[i]
        self.x[i] += diff
        self.y += diff
        self.av[i] += dx

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
        self.evalution = value
        self.actionValue = self.z(value)
        
    def set_prior(self, policy):
        self.priorProbability = helperfuncs.probability_distribution(policy, self.state)
        
    def z(self, x):
        '''Assuming the input is number between 0 and 1'''
        return 1/(-normParam * math.log(x*cutoff, cParam))
    
    def z_inverse(self, y):
        return pow(cParam, -1/y)
        
    def set_choiceProbability(self, weights, actionValues, sumOfWeights):
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
    
        ##choose by highest actionValue for now
        bestNode, value = None, 0
        for child in self.root.childNodes:
            print(child.choiceProbability.value(child.nodeNum), child.actionValue, child.visits)
            if child.actionValue and child.actionValue > value:
                bestNode = child
                value = child.actionValue
                
        return bestNode
    
    def choose(self):
        n = self.root
        
        while len(n.childNodes) > 0:
            branching = len(n.childNodes)
            weights = n.choiceProbability
            weights = [weights.value(i) for i in range(len(weights.x))]
                 
            n = n.childNodes[random.choices(list(range(branching)), weights)[0]]
            
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
        n.set_choiceProbability([weight / sumOfWeights for weight in weights], [n.z_inverse(weight) for weight in weights], 1)
    
        return n
    
    def backup(self, n):
        while n.parentNode != None:
            val = cutoff - n.actionValue ##adversarial search
            nnum = n.nodeNum
            
            n = n.parentNode
            probability = n.choiceProbability
#             print(probability.value(nnum), val, n.actionValue)
            delta = learningRate * probability.value(nnum) * (val - n.actionValue)
            n.actionValue += delta
            if type(n.choiceProbability) != int:
                probability.modify_by_delta(nnum, delta)
            n.visits += 1