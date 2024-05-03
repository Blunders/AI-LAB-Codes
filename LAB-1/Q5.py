import time
import memory_profiler 
from puzzle_solver import IterativeDeepeningAgent, Puzzle8Environment

def solve_puzzle(depth_d, max_depth):
    # Generate Puzzle-8 instance at depth "d"
    puzzle_instance = states_at_depth_dfs(Puzzle8Environment((1, 2, 3, 4, 5, 6, 7, 8, 0)), depth_d)
    
    goal_puzzle_state = (1, 2, 3, 4, 5, 6, 7, 8, 0)  # Example goal state
    
    # Create the Puzzle-8 problem
    puzzle_problem = Puzzle8Environment(puzzle_instance, goal_puzzle_state)
    
    # Solve the problem and measure time
    start_time = time.time()
    iterative_deepening_agent = IterativeDeepeningAgent()
    solution_path = iterative_deepening_agent.iterative_deepening_search(puzzle_problem)
    end_time = time.time()
    
    # Calculate execution time in milliseconds
    execution_time = (end_time - start_time) * 1000
    
    # Measure memory usage using memory_profiler
    memory_usage = memory_profiler.memory_usage((iterative_deepening_agent.iterative_deepening_search, (puzzle_problem,)))
    max_memory_usage = max(memory_usage)
    
    return len(solution_path) - 1, execution_time, max_memory_usage

# Example Usage:
max_depth = 15
print(f"{'Depth':<10}{'Path Length':<15}{'Execution Time (ms)':<25}{'Memory Usage (Bytes)':<25}")
for depth_d in range(1, max_depth + 1):
    result = solve_puzzle(depth_d, max_depth)
    print(f"{depth_d:<10}{result[0]:<15}{result[1]:<25.3f}{result[2]:<25}")
