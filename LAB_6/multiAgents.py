from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    def getAction(self, gameState):
        """
        Collect legal moves and successor states,
        then choose the best action based on the evaluation function.
        """
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Evaluates a state and assigns a score based on proximity to food,
        distance to ghosts, and scared ghost timers.
        """
        # Generate successor state
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Start with the base score from the game state
        score = successorGameState.getScore()

        # Add a penalty for stopping to encourage movement
        if action == Directions.STOP:
            score -= 10

        # Distance to the closest food
        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        if foodDistances:
            score += 10 / min(foodDistances)  # Favor states closer to food

        # Distance to ghosts (penalize proximity to active ghosts, reward for scared ones)
        for i, ghost in enumerate(newGhostStates):
            ghostDistance = manhattanDistance(newPos, ghost.getPosition())
            if newScaredTimes[i] > 0:  # Ghost is scared
                score += 200 / (ghostDistance + 1)  # Favor being closer to scared ghosts
            else:  # Ghost is active
                if ghostDistance < 2:
                    score -= 1000  # Large penalty if too close to an active ghost
                else:
                    score -= 10 / ghostDistance  # Small penalty for proximity to active ghosts

        return score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want
    to add functionality to all your adversarial search agents.  Please do not
    remove anything, however.
    """
    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        return self.minimax(0, 0, gameState)[0]

    def minimax(self, curr_depth, agent_index, gameState):
        """
        Returns the best score for an agent using the minimax algorithm.
        """
        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            curr_depth += 1
        
        if curr_depth == self.depth:
            return None, self.evaluationFunction(gameState)

        best_action = None
        best_score = None
        if agent_index == 0:  # Pacman's turn (maximizing player)
            best_score = float('-inf')
            for action in gameState.getLegalActions(agent_index):
                next_game_state = gameState.generateSuccessor(agent_index, action)
                _, score = self.minimax(curr_depth, agent_index + 1, next_game_state)
                if score > best_score:
                    best_score = score
                    best_action = action
        else:  # Ghost's turn (minimizing player)
            best_score = float('inf')
            for action in gameState.getLegalActions(agent_index):
                next_game_state = gameState.generateSuccessor(agent_index, action)
                _, score = self.minimax(curr_depth, agent_index + 1, next_game_state)
                if score < best_score:
                    best_score = score
                    best_action = action

        return best_action, best_score


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using alpha-beta pruning.
        """
        return self.alphaBeta(0, 0, gameState, float('-inf'), float('inf'))[0]

    def alphaBeta(self, curr_depth, agent_index, gameState, alpha, beta):
        """
        Implements the Alpha-Beta Pruning algorithm.
        """
        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            curr_depth += 1

        if curr_depth == self.depth or gameState.isWin() or gameState.isLose():
            return None, self.evaluationFunction(gameState)

        best_action = None
        if agent_index == 0:  # Pacman's turn (maximizing player)
            best_score = float('-inf')
            for action in gameState.getLegalActions(agent_index):
                next_game_state = gameState.generateSuccessor(agent_index, action)
                _, score = self.alphaBeta(curr_depth, agent_index + 1, next_game_state, alpha, beta)
                if score > best_score:
                    best_score = score
                    best_action = action
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
        else:  # Ghost's turn (minimizing player)
            best_score = float('inf')
            for action in gameState.getLegalActions(agent_index):
                next_game_state = gameState.generateSuccessor(agent_index, action)
                _, score = self.alphaBeta(curr_depth, agent_index + 1, next_game_state, alpha, beta)
                if score < best_score:
                    best_score = score
                    best_action = action
                beta = min(beta, best_score)
                if beta <= alpha:
                    break

        return best_action, best_score


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction.
        """
        return self.expectimax(0, 0, gameState)[0]

    def expectimax(self, curr_depth, agent_index, gameState):
        """
        Implements the expectimax algorithm where ghosts are assumed to make random moves.
        """
        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            curr_depth += 1

        if curr_depth == self.depth or gameState.isWin() or gameState.isLose():
            return None, self.evaluationFunction(gameState)

        best_action = None
        if agent_index == 0:  # Pacman's turn (maximizing player)
            best_score = float('-inf')
            for action in gameState.getLegalActions(agent_index):
                next_game_state = gameState.generateSuccessor(agent_index, action)
                _, score = self.expectimax(curr_depth, agent_index + 1, next_game_state)
                if score > best_score:
                    best_score = score
                    best_action = action
        else:  # Ghost's turn (expectation)
            scores = []
            for action in gameState.getLegalActions(agent_index):
                next_game_state = gameState.generateSuccessor(agent_index, action)
                _, score = self.expectimax(curr_depth, agent_index + 1, next_game_state)
                scores.append(score)
            best_score = sum(scores) / len(scores) if scores else 0
            best_action = action

        return best_action, best_score


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    """
    pacman_pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghost_states = currentGameState.getGhostStates()

    score = currentGameState.getScore()

    # Add a penalty for stopping to encourage movement
    if Directions.STOP in currentGameState.getLegalActions():
        score -= 10

    # Distance to food
    food_distances = [manhattanDistance(pacman_pos, food) for food in food.asList()]
    if food_distances:
        score += 10 / min(food_distances)

    # Ghost penalties and bonuses
    for ghost in ghost_states:
        ghost_dist = manhattanDistance(pacman_pos, ghost.getPosition())
        if ghost.scaredTimer > 0:
            score += 200 / (ghost_dist + 1)  # Favor being close to scared ghosts
        elif ghost_dist < 2:
            score -= 1000  # Strong penalty if too close to an active ghost
        else:
            score -= 10 / ghost_dist  # Small penalty for being near a ghost

    return score


# Abbreviation
better = betterEvaluationFunction
