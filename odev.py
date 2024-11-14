import heapq
import numpy as np
import matplotlib.pyplot as plt

# 30x30 boyutunda bir harita oluşturma (0: boş, 1: engel)
grid_size = 30
grid = np.zeros((grid_size, grid_size))

# Haritaya engeller ekleyelim (örnek olarak)
grid[5:10, 5:10] = 1  # Engel 1
grid[15:20, 15:20] = 1  # Engel 2
grid[20:25, 25:30] = 1  # Engel 3
grid[2:7, 20:25] = 1  # Engel 4
grid[10:15, 10:15] = 1  # Engel 5

# A* algoritmasının kullanacağı yardımcı fonksiyonlar
def heuristic(a, b):
    # Manhattan mesafesi (x1 - x2) + (y1 - y2)
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal):
    # Hareket edilebilen yönler (yukarı, aşağı, sol, sağ)
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # Başlangıç ve hedef için gerekli parametreleri oluştur
    open_list = []
    closed_list = set()
    came_from = {}
    
    # Başlangıç noktası için f ve g değerlerini başlat
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    heapq.heappush(open_list, (f_score[start], start))
    
    while open_list:
        # En düşük f değeri olan noktayı al
        current_f, current = heapq.heappop(open_list)
        
        if current == goal:
            # Hedefe ulaşıldıysa, yolu geri izleyerek döndür
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        
        closed_list.add(current)
        
        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            
            if 0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size and grid[neighbor[1]][neighbor[0]] == 0:
                if neighbor in closed_list:
                    continue
                
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in [i[1] for i in open_list]:
                    heapq.heappush(open_list, (f_score.get(neighbor, float('inf')), neighbor))
                elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue
                
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
    
    return None  # Eğer yol bulunamazsa None döndür

# A* algoritmasını başlatma ve yol hesaplama
start = (10, 10)  # Başlangıç noktası (A* algoramasında (10, 10) kullanılacak)
goal = (20, 20)  # Hedef noktası
path = astar(grid, start, goal)

# Sonuçları yazdırma
if path:
    print("Bulunan yol:")
    for step in path:
        print(step)
else:
    print("Yol bulunamadı!")

# Haritayı görselleştirme
def visualize_path(grid, path):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Gridi çizme
    ax.imshow(grid, cmap='Greys', origin='upper')

    # Engelleri kırmızı renk ile gösterme
    ax.imshow(grid, cmap='Reds', alpha=0.5, origin='upper')

    # Yolu yeşil renkte çizme
    if path:
        path_x = [step[0] for step in path]
        path_y = [step[1] for step in path]
        ax.plot(path_x, path_y, color='green', linewidth=2, marker='o', markersize=5)

    # Başlangıç ve hedef noktalarını mavi ve kırmızı ile gösterme
    ax.scatter(start[0], start[1], color='blue', label="Başlangıç", s=100, edgecolor='black', zorder=5)
    ax.scatter(goal[0], goal[1], color='red', label="Hedef", s=100, edgecolor='black', zorder=5)

    # Görselleştirme detayları
    ax.set_title("A* Algoritması ile Yol Planlaması")
    ax.set_xlabel("X Konumu")
    ax.set_ylabel("Y Konumu")
    ax.legend(loc="upper right")

    plt.grid(True)
    plt.show()

# Haritayı ve yolu görselleştirme
visualize_path(grid,path)
