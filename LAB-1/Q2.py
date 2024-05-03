class GraphSearchProblem:
    def __init__(self, initial_state, goal_state):
        # Initialize the graph search problem with the initial and goal states
        self.initial_state = initial_state
        self.goal_state = goal_state
    
    def goal_test(self, state):
        # Check if the given state is the goal state
        return state == self.goal_state
    
    @staticmethod
    def get_possible_moves(empty_tile_index, size):
        # check all the possible moves of empty_tile
        possible_moves = []
        row, col = divmod(empty_tile_index, size)
        if row > 0:
            possible_moves.append(empty_tile_index - size)  # Move Up
        if row < size - 1:
            possible_moves.append(empty_tile_index + size)  # Move Down
        if col > 0:
            possible_moves.append(empty_tile_index - 1)  # Move Left
        if col < size - 1:
            possible_moves.append(empty_tile_index + 1)  # Move Right
        return possible_moves

    def get_successors(self, state):
        # Implement this method based on your specific problem representation
        pass

class Puzzle8Environment(GraphSearchProblem):
    def __init__(self, initial_state, goal_state):
        # Initialize Puzzle-8 environment with initial and goal states
        super().__init__(initial_state, goal_state)
        self.size = int(len(initial_state) ** 0.5)

    def get_possible_moves(self, empty_tile_index):
        # Get possible moves (indices) for the empty tile based on its current position
        possible_moves = []
        row, col = divmod(empty_tile_index, self.size)
        if row > 0:
            possible_moves.append(empty_tile_index - self.size)  # Move Up
        if row < self.size - 1:
            possible_moves.append(empty_tile_index + self.size)  # Move Down
        if col > 0:
            possible_moves.append(empty_tile_index - 1)  # Move Left
        if col < self.size - 1:
            possible_moves.append(empty_tile_index + 1)  # Move Right
        return possible_moves

    def get_successors(self, state):
        # Get successors of the current state by generating possible moves and resulting states
        successors = []
        empty_tile_index = state.index(0)
        possible_moves = self.get_possible_moves(empty_tile_index)
        
        # Generate successors with the corresponding action, resulting state, and cost
        for move in possible_moves:
            new_state = list(state)
            new_state[empty_tile_index], new_state[move] = new_state[move], new_state[empty_tile_index]
            successors.append((f'Move {state[move]}', tuple(new_state), 1))
        
        return successors
