import bitboards
import autoencoder
import chess

def evaluate(position, nn, encoder):
    embdding = encoder.encode(bitboards.bitboard_to_cnn_input(bitboards.bitboard(position)).unsqueeze(0)).view(1, -1)
    return nn(embedding)
    
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
        self.evalution = value
        
    def set_prior(self, policy):
        self.priorProbability = policy
        
    def get_uct(self):
        pass
    
    def add_child(self, node):
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
            self.backpropagate(node)
    
    def choose(self):
        n = self.root
        
        while len(n.childNodes) > 0:
            branching = len(n.childNodes)
            n = n.childNodes[sorted([[n.get_uct(), i] for i in range(branching)], reverse=True)[0][1]]
            
        return n
    
    def expand(self, n):
        moves = n.moves
        
        value, policy = evaluate(n.position, self.nnet, self.encoder)
        n.set_value(value)
        n.set_prior(policy)
        
        for move in moves:
            board = copy.deepcopy(n.position)
            board.push(move)
                
            newNode = Node(n, board)
            n.add(newNode)
    
        return n
    
    def backpropagate(self, n):
        val = n.evaluation
        
        while n.parentNode != None:
            val = 1 - val ##adversarial search
            n = n.parentNode
            
            ## some update to 'actionValue' and 'noVisits'
            ## node.actionValue += x
            ## node.noVisits += 1
    