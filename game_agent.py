"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import numpy as np


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_winner(player):
        return float('inf')
    if game.is_loser(player):
        return -float('inf')

    open_move = float(len(game.get_legal_moves(player)))

    return open_move


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_winner(player):
        return float('inf')
    if game.is_loser(player):
        return -float('inf')

    improved_score = float(len(game.get_legal_moves(player))) - 2 * float(len(game.get_legal_moves(game.get_opponent(player))))
    return improved_score

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_winner(player):
        return float('inf')
    if game.is_loser(player):
        return -float('inf')

    improved_score = float(len(game.get_legal_moves(player))) -  float(len(game.get_legal_moves(game.get_opponent(player))))
    return improved_score


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.moves = []

class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def __repr__(self):
        """
        Object to string representation used for display
        :return: string 
        """
        return "MinimaxPlayer"

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        # always choose center if first move
        if game.move_count == 0:
            return (
                game.width // 2, game.height // 2
            )

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            best_move = self.minimax(game, self.search_depth)

        except SearchTimeout:
            best_move = max(self.moves, default=[-float('inf'), (-1, -1)])[1]
            self.moves = []
            print('Timed out brotato chip.')

        # Return the best move from the last completed search iteration
        return best_move

    def max_value(self, state, depth):
            """
            Evaluates board positions and returns the utility of the move
            :param state: isolation.Board
            :param depth: depth
            :return: float
            """
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            legal_moves = state.get_legal_moves(self)
            if not legal_moves:
                return state.utility(player=self)
            elif depth <= 0:
                return self.score(state, self)

            val = -float("inf")
            for move in legal_moves:
                val = max(val, self.min_value(state.forecast_move(move), depth=depth-1))
            return val

    def min_value(self, state, depth):
            """
            Evaluates board positions and returns the utility of the move
            :param state: isolation.Board
            :param depth: depth
            :return: float
            """
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            min_player = state.active_player
            legal_moves = state.get_legal_moves(player=min_player)

            if not legal_moves:
                return state.utility(player=min_player)
            elif depth <= 0:
                return self.score(state, self)

            val = float("inf")
            for move in legal_moves:
                val = min(val, self.max_value(state.forecast_move(move), depth=depth-1))
            return val


    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        self.moves = []
        legal_moves = game.get_legal_moves()
        for move in legal_moves:
            move, score = (self.min_value(game.forecast_move(move), depth=depth-1), move)
            self.moves.append(
                (move, score)
            )
        chosen = max(self.moves, default=[-float('inf'), (-1, -1)])[1]
        return chosen


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        # always choose center if first move
        if game.move_count == 0:
            return (
                game.width // 2, game.height // 2
            )

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            best_move = self.alphabeta(game=game, depth=self.search_depth, alpha=float("-inf"), beta=float("inf"))

        except SearchTimeout:
            best_move = max(self.moves, default=[-float('inf'), (-1, -1)])[1]
            self.moves = []
            print('Timed out guapanchita')

        # Return the best move from the last completed search iteration
        return best_move

    def max_value(self, state, depth, alpha, beta):
            """
            Evaluates board positions and returns the utility of the move
            :param state: isolation.Board
            :param depth: depth
            :return: float
            """
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            legal_moves = state.get_legal_moves(self)
            if not legal_moves:
                return state.utility(player=self)
            elif depth <= 0:
                return self.score(state, self)

            val = -float("inf")
            for move in legal_moves:
                val = max(val, self.min_value(state=state.forecast_move(move), depth=depth-1, alpha=alpha, beta=beta))
                if val >= beta:
                    return val
                alpha = max(alpha, val)
            return val

    def min_value(self, state, depth, alpha, beta):
            """
            Evaluates board positions and returns the utility of the move
            :param state: isolation.Board
            :param depth: depth
            :return: float
            """
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            min_player = state.active_player
            legal_moves = state.get_legal_moves(player=min_player)

            if not legal_moves:
                return state.utility(player=min_player)
            elif depth <= 0:
                return self.score(state, self)

            val = float("inf")
            for move in legal_moves:
                val = min(val, self.max_value(state=state.forecast_move(move), depth=depth-1, alpha=alpha, beta=beta))
                if val <= alpha:
                    return val
                beta = min(beta, val)
            return val


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        self.moves = []
        legal_moves = game.get_legal_moves()
        for move in legal_moves:
            move, score = (self.min_value(state=game.forecast_move(move), depth=depth-1, alpha=alpha, beta=beta), move)
            self.moves.append(
                (move, score)
            )
        chosen = max(self.moves, default=[-float('inf'), (-1, -1)])[1]
        return chosen