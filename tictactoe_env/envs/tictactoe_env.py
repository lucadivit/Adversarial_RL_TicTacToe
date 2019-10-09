import gym
from gym import error, spaces, utils
import numpy as np
import os

'''
Tab Game
[0,0,0,0,0,0,0,0,0]

action space {0,1,...,8}  una mossa per ogni casella

obs space min = [0,0,0,0,0,0,0,0,0] max = [2,2,2,2,2,2,2,2,2] . 
stati della tabella. 0 casella vuota, 1 casella con X, 2 casella con O

2 player 
1 per primo agente, 
2 per secondo agente.
Vengono scambiati in automatico
'''
class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    player_1_marker = "X"
    player_2_marker = "O"
    field_symbol = "*"
    free_space_symbol = ' '

    def __init__(self):
        self.actual_player = 1
        self.state = np.zeros(9)
        self.action_space = spaces.Discrete(9)
        self.obs_min = np.full((9,), 0)
        self.obs_max = np.full((9,), 3)
        self.observation_space = spaces.Box(low=self.obs_min, high=self.obs_max, dtype=np.int8)
        self.game_field = self.build_game_field()

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        done = False
        info = {}
        player = self.get_actual_player()
        if(not self.is_board_full()):
            placed = self.append_symbol(player, action)
            info["placed"] = placed
            if(placed is True):
                self.state[action] = player
                reward = -1
                self.print_game_field()
                tris, pos = self.check_if_tris_is_performed(player)
                if(tris is True):
                    reward = reward + 50
                    done = True
                    info["winner"] = player
                self.change_player()
            else:
                reward = -5
        else:
            reward = 0
            done = True
            info["placed"] = None
        return self.state, reward, done, info

    def reset(self):
        self.state = np.zeros(9)
        self.set_actual_player(1)
        self.game_field = self.build_game_field()
        self.print_game_field()
        return self.state

    def set_obs_space(self, obs):
        self.observation_space = obs

    def get_obs_space(self):
        return self.observation_space

    def set_action_space(self, actions):
        self.action_space = actions

    def get_action_space(self):
        return self.action_space

    def get_actual_player(self):
        return self.actual_player

    def set_actual_player(self, player):
        self.actual_player = player

    def change_player(self):
        if(self.get_actual_player() == 1):
            self.set_actual_player(2)
        elif(self.get_actual_player() ==  2):
            self.set_actual_player(1)
        else:
            pass

    def is_board_full(self):
        board_full = True
        for val in self.state:
            if(val == 0):
                board_full = False
                break
        return board_full

    def check_if_tris_is_performed(self, player):
        tris = False
        pos = None
        tris, pos = self.check_tris_for_rows(player)
        if(tris):
            return tris, pos
        tris, pos = self.check_tris_for_columns(player)
        if(tris):
            return tris, pos
        tris, pos = self.check_tris_for_diags(player)
        if(tris):
            return tris, pos
        return tris, pos

    def check_tris_for_rows(self, player):
        tris = False
        pos = None
        if(player == 1):
            if(self.game_field[0][0] == self.player_1_marker and self.game_field[0][3] == self.player_1_marker and self.game_field[0][6] == self.player_1_marker):
                tris = True
                pos = "first row"
            elif(self.game_field[3][0] == self.player_1_marker and self.game_field[3][3] == self.player_1_marker and self.game_field[3][6] == self.player_1_marker):
                tris = True
                pos = "second row"
            elif(self.game_field[6][0] == self.player_1_marker and self.game_field[6][3] == self.player_1_marker and self.game_field[6][6] == self.player_1_marker):
                tris = True
                pos = "third row"
            else:
                pass
        elif(player == 2):
            if(self.game_field[0][0] == self.player_2_marker and self.game_field[0][3] == self.player_2_marker and self.game_field[0][6] == self.player_2_marker):
                tris = True
                pos = "first row"
            elif(self.game_field[3][0] == self.player_2_marker and self.game_field[3][3] == self.player_2_marker and self.game_field[3][6] == self.player_2_marker):
                tris = True
                pos = "second row"
            elif(self.game_field[6][0] == self.player_2_marker and self.game_field[6][3] == self.player_2_marker and self.game_field[6][6] == self.player_2_marker):
                tris = True
                pos = "third row"
            else:
                pass
        return tris, pos

    def check_tris_for_columns(self, player):
        tris = False
        pos = None
        if(player == 1):
            if(self.game_field[0][0] == self.player_1_marker and self.game_field[3][0] == self.player_1_marker and self.game_field[6][0] == self.player_1_marker):
                tris = True
                pos = "first col"
            elif(self.game_field[0][3] == self.player_1_marker and self.game_field[3][3] == self.player_1_marker and self.game_field[6][3] == self.player_1_marker):
                tris = True
                pos = "second col"
            elif(self.game_field[0][6] == self.player_1_marker and self.game_field[3][6] == self.player_1_marker and self.game_field[6][6] == self.player_1_marker):
                tris = True
                pos = "third col"
            else:
                pass
        elif(player == 2):
            if(self.game_field[0][0] == self.player_2_marker and self.game_field[3][0] == self.player_2_marker and self.game_field[6][0] == self.player_2_marker):
                tris = True
                pos = "first col"
            elif(self.game_field[0][3] == self.player_2_marker and self.game_field[3][3] == self.player_2_marker and self.game_field[6][3] == self.player_2_marker):
                tris = True
                pos = "second col"
            elif(self.game_field[0][6] == self.player_2_marker and self.game_field[3][6] == self.player_2_marker and self.game_field[6][6] == self.player_2_marker):
                tris = True
                pos = "third col"
            else:
                pass
        return tris, pos

    def check_tris_for_diags(self, player):
        tris = False
        pos = None
        if(player == 1):
            if(self.game_field[0][0] == self.player_1_marker and self.game_field[3][3] == self.player_1_marker and self.game_field[6][6] == self.player_1_marker):
                tris = True
                pos = "l_to_r_diag"
            elif(self.game_field[0][6] == self.player_1_marker and self.game_field[3][3] == self.player_1_marker and self.game_field[6][0] == self.player_1_marker):
                tris = True
                pos = "r_to_l_diag"
            else:
                pass
        elif(player == 2):
            if(self.game_field[0][0] == self.player_2_marker and self.game_field[3][3] == self.player_2_marker and self.game_field[6][6] == self.player_2_marker):
                tris = True
                pos = "l_to_r_diag"
            elif(self.game_field[0][6] == self.player_2_marker and self.game_field[3][3] == self.player_2_marker and self.game_field[6][0] == self.player_2_marker):
                tris = True
                pos = "r_to_l_diag"
            else:
                pass

        return tris, pos

    def append_symbol(self, player, action):
        placed = False
        if (action == 0):
            if(player == 1):
                if(self.game_field[0][0] == self.free_space_symbol):
                    self.game_field[0][0] = self.player_1_marker
                    placed = True
            elif(player == 2):
                if (self.game_field[0][0] == self.free_space_symbol):
                    self.game_field[0][0] = self.player_2_marker
                    placed = True
        elif (action == 1):
            if (player == 1):
                if (self.game_field[0][3] == self.free_space_symbol):
                    self.game_field[0][3] = self.player_1_marker
                    placed = True
            elif (player == 2):
                if (self.game_field[0][3] == self.free_space_symbol):
                    self.game_field[0][3] = self.player_2_marker
                    placed = True
        elif (action == 2):
            if (player == 1):
                if (self.game_field[0][6] == self.free_space_symbol):
                    self.game_field[0][6] = self.player_1_marker
                    placed = True
            elif (player == 2):
                if (self.game_field[0][6] == self.free_space_symbol):
                    self.game_field[0][6] = self.player_2_marker
                    placed = True
        elif (action == 3):
            if (player == 1):
                if (self.game_field[3][0] == self.free_space_symbol):
                    self.game_field[3][0] = self.player_1_marker
                    placed = True
            elif (player == 2):
                if (self.game_field[3][0] == self.free_space_symbol):
                    self.game_field[3][0] = self.player_2_marker
                    placed = True
        elif (action == 4):
            if (player == 1):
                if (self.game_field[3][3] == self.free_space_symbol):
                    self.game_field[3][3] = self.player_1_marker
                    placed = True
            elif (player == 2):
                if (self.game_field[3][3] == self.free_space_symbol):
                    self.game_field[3][3] = self.player_2_marker
                    placed = True
        elif (action == 5):
            if (player == 1):
                if (self.game_field[3][6] == self.free_space_symbol):
                    self.game_field[3][6] = self.player_1_marker
                    placed = True
            elif (player == 2):
                if (self.game_field[3][6] == self.free_space_symbol):
                    self.game_field[3][6] = self.player_2_marker
                    placed = True
        elif (action == 6):
            if (player == 1):
                if (self.game_field[6][0] == self.free_space_symbol):
                    self.game_field[6][0] = self.player_1_marker
                    placed = True
            elif (player == 2):
                if (self.game_field[6][0] == self.free_space_symbol):
                    self.game_field[6][0] = self.player_2_marker
                    placed = True
        elif (action == 7):
            if (player == 1):
                if (self.game_field[6][3] == self.free_space_symbol):
                    self.game_field[6][3] = self.player_1_marker
                    placed = True
            elif (player == 2):
                if (self.game_field[6][3] == self.free_space_symbol):
                    self.game_field[6][3] = self.player_2_marker
                    placed = True
        elif (action == 8):
            if (player == 1):
                if (self.game_field[6][6] == self.free_space_symbol):
                    self.game_field[6][6] = self.player_1_marker
                    placed = True
            elif (player == 2):
                if (self.game_field[6][6] == self.free_space_symbol):
                    self.game_field[6][6] = self.player_2_marker
                    placed = True
        return placed

    def build_game_field(self):
        dim = 8
        field = []
        for row in range(0, dim):
            r = []
            for col in range(0, dim):
                if (row == 2 or row == 5):
                    r.append(self.field_symbol)
                elif (col == 2 or col == 5):
                    r.append(self.field_symbol)
                else:
                    r.append(self.free_space_symbol)
            field.append(r)
        return np.array(field)

    def print_game_field(self):
        os.system('clear')
        dimensions = self.game_field.shape
        y_dim = dimensions[0]
        x_dim = dimensions[1]
        for y in range(0, y_dim):
            for x in range(0, x_dim):
                print (self.game_field[y][x], end=' ')
            print ("\n", end='\r')

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass