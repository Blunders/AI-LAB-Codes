import random
import math
import matplotlib.pyplot as plt

# Define the tourist locations in Rajasthan
rajasthan_locations = {
    "Ajmer": (26.4499, 74.6399),
    "Alwar": (27.5530, 76.6346),
    "Jaipur": (26.9124, 75.7873),
    "Udaipur": (24.5854, 73.7125),
    "Jodhpur": (26.2389, 73.0243),
    "Pushkar": (26.4897, 74.5511),
    "Jaisalmer": (26.9157, 70.9083),
    "Mount Abu": (24.5925, 72.7156),
    "Bikaner": (28.0229, 73.3119),
    "Ranthambore": (25.8667, 76.3),
    "Chittorgarh": (24.8887, 74.6269),
    "Bundi": (25.4415, 75.6454),
    "Bharatpur": (27.1767, 77.6844),
    "Kumbhalgarh": (25.152314, 73.590660),
    "Sawai Madhopur": (25.9928, 76.3526),
    "Sikar": (27.611195, 75.155155),
    "Dungarpur": (23.8363, 73.7143),
    "Nathdwara": (24.9339, 73.8226),
    "Mandawa": (28.0556, 75.1419),
    "Khetri": (28.001867, 75.789161)
}

# Calculate distance between two locations using Haversine formula
def distance(location1, location2):
    lat1, lon1 = location1
    lat2, lon2 = location2
    radius = 6371 # Radius of Earth in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c
    return d

# Calculate total distance of a tour
def total_distance(tour):
    total = 0
    for i in range(len(tour)):
        total += distance(rajasthan_locations[tour[i]],
                          rajasthan_locations[tour[(i + 1) % len(tour)]])
    return total

# Simulated Annealing algorithm with distance tracking
def simulated_annealing_with_distances(locations, initial_temperature=1000, cooling_rate=0.99, num_iterations=100000):
    current_solution = list(locations.keys())
    random.shuffle(current_solution)
    current_cost = total_distance(current_solution)
    best_solution = current_solution[:]
    best_cost = current_cost
    temperature = initial_temperature
    distances = []
    
    for _ in range(num_iterations):
        new_solution = current_solution[:]
        i, j = random.sample(range(len(new_solution)), 2)
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        new_cost = total_distance(new_solution)
        
        if new_cost < current_cost or random.random() < math.exp((current_cost - new_cost) / temperature):
            current_solution = new_solution[:]
            current_cost = new_cost
        
        if new_cost < best_cost:
            best_solution = new_solution[:]
            best_cost = new_cost
        
        distances.append(current_cost)
        temperature *= cooling_rate
    
    best_solution.append(best_solution[0])  # Ensure the tour forms a cycle
    return best_solution, best_cost, distances

# Run simulated annealing to find an optimized tour with distances
best_tour, best_cost, distances = simulated_annealing_with_distances(rajasthan_locations)

# Extract intermediary distances for the final route
intermediary_distances = [distance(rajasthan_locations[best_tour[i]], rajasthan_locations[best_tour[i + 1]])
                          for i in range(len(best_tour) - 1)]

# Print the results
print("Best tour:", best_tour)
print("Best cost:", best_cost, "km")
print("Intermediary distances:", intermediary_distances)

# Visualization
plt.figure(figsize=(10, 8))
for city, (lat, lon) in rajasthan_locations.items():
    plt.plot(lon, lat, 'o', markersize=8, label=city)

plt.plot([rajasthan_locations[best_tour[i]][1] for i in range(len(best_tour))],
         [rajasthan_locations[best_tour[i]][0] for i in range(len(best_tour))],
         color='blue', linestyle='-', linewidth=2)  

plt.title('Cost effective path')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)
plt.show()
