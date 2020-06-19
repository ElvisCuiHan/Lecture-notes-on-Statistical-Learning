import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class MC_COVID(object):
    def __init__(self, initial_state=None):
        """

        :param initial_state: An array of form nrow x ncol giving the initial state of the whole system.
        :param nrow: row number
        :param ncol: column number
        """
        self.state = initial_state
        self.initial_state = initial_state.copy()
        self.nrow = initial_state.shape[0]
        self.ncol = initial_state.shape[1]
        self.markov_kernel = {}
        self._init_markov_kernel()

    def get_neighbor_state(self, row=0, col=0):
        """

        :param row: [int] the row coordinate of the point
        :param col: [int] the column coordinate of the point
        :return: A state ["H", "L", "C", "D", "I"] and an integer taking values in [0, 1, 2, 3, 4, 5, 6, 7, 8],
            which represents the number of covid points surrounding it.
        """
        count_covid_neighbor = 0
        # Check whether the point is in the corner

        if row == 0 and col == 0:
            neighborhoods = [self.state[row + 1, col], self.state[row, col + 1], self.state[row + 1, col + 1]]
            for neighbor in neighborhoods:
                if neighbor == "C" or neighbor == "L":
                    count_covid_neighbor += 1
            return self.state[row, col] + str(count_covid_neighbor)

        elif row == 0 and col == self.ncol - 1:
            neighborhoods = [self.state[row + 1, col], self.state[row, col - 1], self.state[row + 1, col - 1]]
            for neighbor in neighborhoods:
                if neighbor == "C" or neighbor == "L":
                    count_covid_neighbor += 1
            return self.state[row, col] + str(count_covid_neighbor)

        elif row == self.nrow - 1 and col == self.ncol - 1:
            neighborhoods = [self.state[row - 1, col], self.state[row, col - 1], self.state[row - 1, col - 1]]
            for neighbor in neighborhoods:
                if neighbor == "C" or neighbor == "L":
                    count_covid_neighbor += 1
            return self.state[row, col] + str(count_covid_neighbor)

        elif row == self.nrow - 1 and col == 0:
            neighborhoods = [self.state[row - 1, col], self.state[row, col + 1], self.state[row - 1, col + 1]]
            for neighbor in neighborhoods:
                if neighbor == "C" or neighbor == "L":
                    count_covid_neighbor += 1
            return self.state[row, col] + str(count_covid_neighbor)

        # Check whether the point is in the first row or the last row
        elif row == 0:
            neighborhoods = [self.state[row, col - 1], self.state[row, col + 1],
                             self.state[row + 1, col - 1], self.state[row + 1, col], self.state[row + 1, col + 1]]
            for neighbor in neighborhoods:
                if neighbor == "C" or neighbor == "L":
                    count_covid_neighbor += 1
            return self.state[row, col] + str(count_covid_neighbor)

        elif row == self.nrow - 1:
            neighborhoods = [self.state[row, col - 1], self.state[row, col + 1],
                             self.state[row - 1, col - 1], self.state[row - 1, col], self.state[row - 1, col + 1]]
            for neighbor in neighborhoods:
                if neighbor == "C" or neighbor == "L":
                    count_covid_neighbor += 1
            return self.state[row, col] + str(count_covid_neighbor)

        # Check whether the point is in the first column or the last column
        elif col == self.ncol - 1:
            neighborhoods = [self.state[row - 1, col], self.state[row + 1, col],
                             self.state[row + 1, col - 1], self.state[row, col - 1], self.state[row - 1, col - 1]]
            for neighbor in neighborhoods:
                if neighbor == "C" or neighbor == "L":
                    count_covid_neighbor += 1
            return self.state[row, col] + str(count_covid_neighbor)

        elif col == 0:
            neighborhoods = [self.state[row - 1, col], self.state[row + 1, col],
                             self.state[row + 1, col + 1], self.state[row, col + 1], self.state[row - 1, col + 1]]
            for neighbor in neighborhoods:
                if neighbor == "C" or neighbor == "L":
                    count_covid_neighbor += 1
            return self.state[row, col] + str(count_covid_neighbor)

        # The only possibility is that the point lies in the middle of the world
        else:
            neighborhoods = [self.state[row-1, col], self.state[row+1, col],
                             self.state[row-1,col-1], self.state[row,col-1], self.state[row+1,col-1],
                             self.state[row-1,col+1], self.state[row,col+1], self.state[row+1,col+1]]
            for neighbor in neighborhoods:
                if neighbor == "C" or neighbor == "L":
                    count_covid_neighbor += 1
            return self.state[row, col] + str(count_covid_neighbor)

    def _init_markov_kernel(self, user_defined_prob=None):
        """
        Initiating the Markov transition kernel
        :return: self
        """
        if user_defined_prob:
            self.markov_kernel = user_defined_prob
            return self
        rate = 0.8
        dead = 0.01
        immune = 0.12
        recover = 0.01
        self.markov_kernel['H0'] = np.array([rate**0, 1-rate**0, 0, 0., 0.])
        self.markov_kernel['H1'] = np.array([rate**1, 1-rate**1, 0, 0., 0.])
        self.markov_kernel['H2'] = np.array([rate**2, 1-rate**2, 0, 0., 0.])
        self.markov_kernel['H3'] = np.array([rate**3, 1-rate**3, 0, 0., 0.])
        self.markov_kernel['H4'] = np.array([rate**4, 1-rate**4, 0, 0., 0.])
        self.markov_kernel['H5'] = np.array([rate**5, 1-rate**5, 0, 0., 0.])
        self.markov_kernel['H6'] = np.array([rate**6, 1-rate**6, 0, 0., 0.])
        self.markov_kernel['H7'] = np.array([rate**7, 1-rate**7, 0, 0., 0.])
        self.markov_kernel['H8'] = np.array([rate**8, 1-rate**8, 0, 0., 0.])
        self.markov_kernel['C0'] = np.array([recover, 0, 1-(recover+dead+immune), dead, immune])
        self.markov_kernel['C1'] = np.array([recover, 0, 1-(recover+dead+immune), dead, immune])
        self.markov_kernel['C2'] = np.array([recover, 0, 1-(recover+dead+immune), dead, immune])
        self.markov_kernel['C3'] = np.array([recover, 0, 1-(recover+dead+immune), dead, immune])
        self.markov_kernel['C4'] = np.array([recover, 0, 1-(recover+dead+immune), dead, immune])
        self.markov_kernel['C5'] = np.array([recover, 0, 1-(recover+dead+immune), dead, immune])
        self.markov_kernel['C6'] = np.array([recover, 0, 1-(recover+dead+immune), dead, immune])
        self.markov_kernel['C7'] = np.array([recover, 0, 1-(recover+dead+immune), dead, immune])
        self.markov_kernel['C8'] = np.array([recover, 0, 1-(recover+dead+immune), dead, immune])
        self.markov_kernel['L']  = np.array([0, 0.8, 0.2, 0, 0])
        self.markov_kernel['D'] = np.array([0., 0., 0, 1, 0])
        self.markov_kernel['I'] = np.array([0.0, 0, 0.0, 0., 1])
        return self

    def transition_kernel(self, state=None):
        one_step_transition = self.markov_kernel[state]
        return one_step_transition

    def update_state_single(self, row=0, col=0):
        cur_state = self.state[row, col]
        if cur_state not in ["D", "I", "L"]:
            cur_neighbor_state = self.get_neighbor_state(row, col)
            trans_prob = self.transition_kernel(cur_neighbor_state)
        else:
            trans_prob = self.transition_kernel(cur_state)
        next_state = int(np.random.choice(5, 1, p=trans_prob))
        next_state = ["H", "C", "L", "D", "I"][next_state]
        return next_state

    def update_state_all(self):
        for i in range(self.nrow):
            for j in range(self.ncol):
                self.state[i, j] = self.update_state_single(i, j)
        return self.state

    def _recreate_world(self):
        self.state = self.initial_state

bb = np.random.choice(["H", "C", "L"], (88, 88), p=[0.996, 0.001, 0.003])

hi = MC_COVID(bb)

##### VISUALIZATION !!!!!!!!!!! ##############################

def dynamic_simulation():
    fig = plt.figure()
    i=0

    image_map = {"H":0.22, "C":0.99, "I": 0, "D":0.44, "L":0.73}
    bitch = [[image_map[dude] for dude in hi.state[0]]]
    for jj in range(1, hi.ncol):
        bitch.append([image_map[dude] for dude in hi.state[jj]])

    im = plt.imshow(bitch, animated=True)
    def updatefig(*args):
        hi.update_state_all()
        bitch = [[image_map[dude] for dude in hi.state[0]]]
        for jj in range(1, hi.nrow):
            bitch.append([image_map[dude] for dude in hi.state[jj]])
        im.set_array(bitch)
        return im,
    ani = animation.FuncAnimation(fig, updatefig,  blit=True, interval=99)
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.show()

dynamic_simulation()

