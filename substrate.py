import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from shapely.geometry import Polygon, Point
from scipy.spatial import Voronoi, voronoi_plot_2d
from descartes import PolygonPatch

from cell import Cell

N = 500
d = 7
D = 30

def force(distance) :
    if distance < D :   # interaction distance
        return (distance-d)*(distance-D)/(D-d)
    else :
        return 0        # no interaction
    
def get_intersection_shape(cell, polygon):  # return the intersection of a circle of raidus D and a polygon
    cell_point = Point(cell.x, cell.y)
    cell_circle = cell_point.buffer(D)
    polygon_shape = Polygon(polygon)

    if cell_circle.contains(polygon_shape):
        cell.border = False     # the cell is not on the border of the cell set
        return polygon_shape
    
    cell.border = True

    # Check if the polygon fully contains the circle
    if polygon_shape.contains(cell_circle):
        return cell_circle
    
    # Otherwise, return the intersection of the polygon and the circle
    cell.border = True
    intersection_shape = polygon_shape.intersection(cell_circle)
    return intersection_shape

class Substrate :   # the substrate is a bunch of cells
    def __init__(self, model, birth_rate, death_rate) :
        self.model = model     # object of class Model 
        self.birth_rate = birth_rate
        self.death_rate = death_rate
        first_cell = Cell(N//2, N//2, values = np.zeros(self.model.nb_channels))
        first_cell.values[0] = 1.0      # live cell = 1
        first_cell.values[1] = 0.25     # angle for replication = pi/2 : vertical replication
        first_cell.values[2] = birth_rate      # replication value at the beginning
        self.pop = [first_cell]
        # add also the four corners with 0 in replication, and 0 in color
            # the Voronoi function doesn't work if there are less than 5 points in the substrate
        corners = []
        corners.append(Cell(0, 0, values = np.zeros(self.model.nb_channels)))
        corners.append(Cell(0, N, values = np.zeros(self.model.nb_channels)))
        corners.append(Cell(N, 0, values = np.zeros(self.model.nb_channels)))
        corners.append(Cell(N, N, values = np.zeros(self.model.nb_channels)))

        for c in corners:
            self.pop.append(c)

    def compute_graph(self):
        points = np.array([(cell.x, cell.y) for cell in self.pop])
        vor = Voronoi(points)
        neighbors = [[] for _ in range(len(points))]
        # Loop through each ridge (edge) in the Voronoi diagram
        for point_indices in vor.ridge_points:
            p1, p2 = point_indices
            neighbors[p1].append(p2)
            neighbors[p2].append(p1)
        return vor, neighbors
    
    def population_step(self):
        new_pop = []
        to_remove = []
        for cell in self.pop :
            if cell.border and cell.values[2] >= self.birth_rate :          # treshold for replication
                new_pop.append(cell.replicate())
            if cell.values[0] < self.death_rate and not (cell.x == 0 or cell.x == N or cell.y == 0 or cell.y == N) :
                to_remove.append(cell)
            elif cell.x < 0 or cell.x > N or cell.y < 0 or cell.y > N:
                to_remove.append(cell)
        for cell in to_remove :
            self.pop.remove(cell)
        self.pop += new_pop

    def update_step(self, neighbors, update_rate = 0.01):
        for i, cell in enumerate(self.pop) :
            if cell.x <= 0 or cell.x >= N or cell.y <= 0 or cell.y >= N:
                continue
            sobel_x = np.zeros(self.model.nb_channels)
            sobel_y = np.zeros(self.model.nb_channels)
            sum_sobel_x = 0
            sum_sobel_y = 0
            for j in neighbors[i] :
                neighbor = self.pop[j]
                if neighbor.x <= 0 or neighbor.x >= N or neighbor.y <= 0 or neighbor.y >= N:
                    continue
                distance = cell.distance(neighbor)
                angle = np.arctan2(neighbor.y - cell.y, neighbor.x - cell.x)
                a = np.cos(angle)/distance
                b = np.sin(angle)/distance
                sobel_x += a*self.pop[j].values
                sobel_y += b*self.pop[j].values
                sum_sobel_x += a
                sum_sobel_y += b
            if sum_sobel_x != 0 :
                sobel_x /= sum_sobel_x
            if sum_sobel_y != 0 :
                sobel_y /= sum_sobel_y
            perception_vector = np.concatenate([cell.values, sobel_x, sobel_y])
            update_vector = self.model.update_function(perception_vector)
            #update_vector = np.clip(update_vector, -10, 10)
            cell.values += update_rate * update_vector
            cell.values[0] = np.clip(cell.values[0], 0, 1)
            cell.values[1] = cell.values[1] % 1
            cell.values[2:] = np.clip(cell.values[2:], 0, 1)
        return

    def position_step(self, neighbors):
        dico_force = {}
        for i, cell in enumerate(self.pop) :
            if cell.x <= 0 or cell.x >= N or cell.y <= 0 or cell.y >= N:
                continue
            force_x = 0
            force_y = 0
            for j in neighbors[i] :
                neighbor = self.pop[j]
                if neighbor.x <= 0 or neighbor.x >= N or neighbor.y <= 0 or neighbor.y >= N:
                    continue
                angle = np.arctan2(neighbor.y - cell.y, neighbor.x - cell.x)
                f = force(cell.distance(neighbor))
                force_x += f*np.cos(angle)
                force_y += f*np.sin(angle)
            dico_force[cell] = [force_x, force_y]
        for cell, force_vector in dico_force.items() :
            cell.x += force_vector[0]
            cell.y += force_vector[1]

    def anim(self, nb_frames, save = False):
        # run the simulation and display the result as an animation

        fig, ax = plt.subplots()
        fig.set_size_inches(N//100, N//100)   # square figure
        def update(frame):
            print(f"Animation : {frame+1}/{nb_frames}" , flush=True, end='\r')
            ax.clear()

            self.population_step()
            
            img, neigh = self.get_image()
            ax.imshow(img, cmap = 'gray')
            
            self.update_step(neigh)
            self.position_step(neigh)
            
            return ax

        # fps = 1 / interval
        anim = animation.FuncAnimation(fig, update, interval=20, frames=nb_frames)
        if save:
            anim.save('output.gif', writer='imagemagick', fps=20)
            print('Animation saved')
        else:
            plt.show()

    def display(self, target):
        img, _ = self.get_image()
        # plot the image of the substrate and the target in the same figure
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(10, 5)
        ax[0].imshow(img, cmap = 'gray')
        # ax[0].set_title('Model')
        # ax[0].axis('off')
        ax[1].imshow(target, cmap = 'gray')
        # ax[1].set_title('Target')
        # ax[1].axis('off')
        plt.show()


    def run(self, nb_steps, max_pop = 200):    # without computing the image at each step
        for i in range(nb_steps):

            self.population_step()
            _, neigh = self.compute_graph()
            self.update_step(neigh)
            self.position_step(neigh)

            nb_cells = len(self.pop)
            if nb_cells > max_pop or nb_cells == 4 :
                return i    # the simulation is stopped if there are too many cells or no cells
        return nb_steps
            
    def get_image(self):    # return the image of the substrate as a numpy array of shape (N, N)

        if len(self.pop) == 4 :
            img = np.ones((N, N))*255
            img[N//2, N//2] = 0   # the center is black to make contrast
            return img, []

        fig, ax = plt.subplots()
        # remove the axis

        ax.axis('off')
        fig.set_size_inches(N//100, N//100)
        vor, neigh = self.compute_graph()
        if vor:
            voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False, line_colors = 'none')
            for i, cell in enumerate(self.pop):
                if cell.x <= 0 or cell.x >= N or cell.y <= 0 or cell.y >= N:
                    continue
                region_index = vor.point_region[i]
                region = vor.regions[region_index]
                if not -1 in region and len(region) > 0:
                    polygon = [vor.vertices[j] for j in region]
                    intersection_shape = get_intersection_shape(cell, polygon)
                    if not intersection_shape.is_empty:
                        color = (1-cell.values[0], 1-cell.values[0], 1-cell.values[0])
                        patch = PolygonPatch(intersection_shape, facecolor=color, edgecolor=color)
                        ax.add_patch(patch)
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(N, N, 4)
        # keep only the first channel
        data = data[:,:,0].reshape(N, N)
        data = np.array(data, dtype = np.float64)       # very important line !! (don't remove it)
        plt.close(fig)
        return data, neigh
