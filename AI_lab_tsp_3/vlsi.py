import random
import math
import matplotlib.pyplot as plt

# Function to read .tsp file and extract city coordinates
def read_tsp_file(filename):
    cities = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        coord_section_index = lines.index('NODE_COORD_SECTION\n') + 1
        for line in lines[coord_section_index:]:
            if line.strip() == 'EOF':
                break
            parts = line.strip().split()
            city_id = int(parts[0])
            x_coord = float(parts[1])
            y_coord = float(parts[2])
            cities[city_id] = (x_coord, y_coord)
    return cities

# Function to calculate Euclidean distance between two cities
def distance(city1, city2):
    x1, y1 = city1
    x2, y2 = city2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

# Function to calculate total distance of a tour
def total_distance(tour, cities):
    total = 0
    for i in range(len(tour)):
        total += distance(cities[tour[i]], cities[tour[(i + 1) % len(tour)]])
    return total

# Function to perform Lin-Kernighan heuristic for TSP with branching and cyclic path assumption
def lin_kernighan_tsp_cyclic_branching(cities, start_city, num_candidates=10, num_iterations=1000):
    best_tour = None
    best_cost = float('inf')
    for _ in range(num_candidates):
        tour = [start_city]  # Start with the specified starting city
        current_city = start_city
        current_cost = 0
        # Generate a candidate tour
        for _ in range(len(cities) - 1):
            next_city = None
            min_distance = float('inf')
            for city_id, city_coords in cities.items():
                if city_id not in tour:
                    dist = distance(cities[current_city], city_coords)
                    if dist < min_distance:
                        min_distance = dist
                        next_city = city_id
            tour.append(next_city)
            current_cost += min_distance
            current_city = next_city
        tour.append(start_city)  # Ensure the tour ends with the start city
        current_cost += distance(cities[current_city], cities[start_city])
        # Improve the tour using Lin-Kernighan heuristic
        for _ in range(num_iterations):
            i, j = random.sample(range(1, len(tour) - 1), 2)  # Exclude the start city from swapping
            segment = (tour[i], tour[j])  # Define the segment as (tour[i], tour[j]) and its reverse (tour[j], tour[i])
            reverse_segment = (tour[j], tour[i])
            candidates = []
            for k in range(1, len(tour) - 1):
                if tour[k] not in segment:
                    gain = distance(cities[tour[i]], cities[tour[k]]) + distance(cities[tour[j]], cities[tour[(k + 1) % len(tour)]]) - distance(cities[tour[i]], cities[tour[j]]) - distance(cities[tour[k]], cities[tour[(k + 1) % len(tour)]])
                    candidates.append((k, gain))
            candidates.sort(key=lambda x: x[1], reverse=True)  # Sort candidates by gain
            for k, gain in candidates:
                if gain > 0:
                    new_tour = tour[:k + 1] + list(reversed(tour[k + 1:j + 1])) + tour[(j + 1) % len(tour):]
                    new_cost = total_distance(new_tour, cities)
                    if new_cost < current_cost:
                        tour = new_tour[:]
                        current_cost = new_cost
            if current_cost < best_cost:
                best_tour = tour[:]
                best_cost = current_cost
    return best_tour, best_cost

# Read .tsp file
filename = 'xqg237.tsp'
cities = read_tsp_file(filename)

# Choose a starting city
start_city = next(iter(cities.keys()))

# Run Lin-Kernighan heuristic with cyclic path assumption and branching to find an optimized tour
best_tour, best_cost = lin_kernighan_tsp_cyclic_branching(cities, start_city)

# Plot cities and tour
plt.figure(figsize=(10, 8))
for city, (x, y) in cities.items():
    plt.plot(x, y, 'bo')
    plt.text(x, y, ' ' + str(city), fontsize=9, ha='right')
for i in range(len(best_tour)):
    city1 = cities[best_tour[i]]
    city2 = cities[best_tour[(i + 1) % len(best_tour)]]
    plt.plot([city1[0], city2[0]], [city1[1], city2[1]], 'r-')

plt.title('Optimized Tour with Cyclic Path Assumption and Branching Lin-Kernighan Heuristic')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.show()

# Print the results
print("Best tour:", best_tour)
print("Best cost:", best_cost)
