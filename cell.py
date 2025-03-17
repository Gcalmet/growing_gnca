import numpy as np

class Cell :
    def __init__(self, x, y, values = None) :
        self.x = x  # not int
        self.y = y  # not int
        self.values = values

    def distance(self, other) :
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def replicate(self) :  # angle between 0 and 2*np.pi, distance = 1
        angle = 2*np.pi*self.values[1] 
        new_x = self.x + np.cos(angle)
        new_y = self.y + np.sin(angle)
        new_values = self.values.copy() # possibility to add mutations but it would make the model undeterministic
        #new_values += np.random.normal(0, 0.05, len(new_values))
        #new_values = np.clip(new_values, 0, 1)
        new_replication_rate = self.values[2]/2    # the replication rate is divided by 2 between the two cells
        new_values[2] = new_replication_rate
        self.values[2] = new_replication_rate
        return Cell(new_x, new_y, new_values)