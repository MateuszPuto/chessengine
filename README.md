#Chess engine project

Usefull literture:
[AlphaGo](https://www.researchgate.net/publication/292074166_Mastering_the_game_of_Go_with_deep_neural_networks_and_tree_search)
[AlphaZero](https://arxiv.org/pdf/1712.01815.pdf)
[Deep Pepper: alphazero clone](https://arxiv.org/pdf/1806.00683.pdf)
[DeepChess: comparison based engine with α-β search](https://www.researchgate.net/profile/Eli_David/publication/306081185_DeepChess_End-to-End_Deep_Neural_Network_for_Automatic_Learning_in_Chess/links/59fe615aaca272347a2796a8/DeepChess-End-to-End-Deep-Neural-Network-for-Automatic-Learning-in-Chess.pdf)
[Learning to evaluate chess positions with deep neural networks and limited lookahead](https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/ICPRAM_CHESS_DNN_2018.pdf)
[Best first minimax: principal variation](https://www.aaai.org/Papers/AAAI/1994/AAAI94-210.pdf)
[Chessprogramming](https://www.chessprogramming.org/Main_Page)


##Chess overview
Chess is a game for two players in which win, draw and loss results are possible. The game consists of a state which is defined as an 8 by 8 board, with 6 types of pieces with respecive starting quantities: 8 pawns, 2 knights, 2 bishop, 2 rooks, 1 queen and 1 king for each player. Transitions of the position are clearly defined. Players can perform board transition alternately, starting with player with white pieces and shifting to player with black ones in a cycle. Every piece has an unique way of moving and attacking called commonly capturing, some of them such as en passant or castling are conditional, based on the board state. Pieces can capture opposing pices with attack move by occuping their square. King cannot be capured. When king is attaced it must be moved to another empty square or game becomes lost for the player that coudn't do so. There exists additional rule for promoting a pawn in which pawn that reaches other side of the board becomes different piece. There are few different ways of drawing a game which include stalemate, threefold repetition of the position and fifty-move rule. Game can be also lost by resignation or drawn by agreement. In competitive enviroments it is played with time controls where additional rules apply.

##Solution approaches
Problem of solving a chess game consists of being able to find a best move for each position. Best move means here board transition that given that consecutively best moves are played until the game termination the result is no worse than that achieved by any other than optimal sequance of moves, taking into consideration that we have influance only on the moves of the player of the same color as this of a starting position. The game of chess is trivially solved given infinite computational resources. This could be achieved for example by precomputed lookup table. However when realistic time and memory constraints are applied problem becomes more interesting. Few possible approches have been proposed. They can be distilled to two main propositions. One points at esimating state value, which is result of the game from position given that best moves are played. This can potentially be done by some function v(state, ...) which takes as input some states and ranks them or outputs the numeric value which we use for ranking purposes. Such numeric value can be interpreted in varius ways such as material balance or probability of winning. What's important is that if function **v** produces the perfect ranking we can easily solve the game. The other approach notes that by applying all possible transitions to a position recoursively to the end of the game it is possible to check all lines of play and choose the most favourable one. This is achieved with some search algorithm, usually with searching in a tree. It is worth noting that this can also lead to perfect play if we can expand all nodes and don't have to use heuristic evaluation on non-terminal nodes. The current best performing agents combine this two approaches. Details of each program can vary significantly. It is important to point that top programs are the effect of applying to them long tweaking or more recently lots of compute.

##Goal of the project
The goal of this project is to check how strong of an agent can be produced given limited precomputation and with general algorithms. This is not to say that we aim at producing game independent agent, but rather one that does not use solutions that fix the agent behaviour with respect to specific rules. Superhuman play has been achieved in the domain of chess and we do not try to simply replicate it but rather use chess as a as testing environment for developing more general solutions. The basic idea is to use combination of neural networks and search tree techinques to produce a game playing agent.

#Implementation

##Board representation

