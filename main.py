import gym
import tictactoe_env
import time
from QLearningAgent import  *
from GreedyPolicies import *

env = gym.make("tictactoe-v0")
policy = EGreedyPolicy()
agent_1_q_table_name = "q_learning_agent_1.pkl"
agent_2_q_table_name = "q_learning_agent_2.pkl"
agent_1 = QLearningAgent(env=env, policy=policy)
agent_2 = QLearningAgent(env=env, policy=policy)
agent_1.load_stored_q_table(agent_1_q_table_name)
agent_2.load_stored_q_table(agent_2_q_table_name)
time.sleep(1)

def print_winner(winner):
    if (winner is not None):
        print ("\nHa vinto il giocatore " + str(winner) + "\n")
    else:
        print ("\nPareggio\n")

def start_game(time_between_steps = 1, time_between_episodes=2):
    for i in range(0, 100):
        done = False
        print("Start episode ", i)
        time.sleep(time_between_episodes)
        state = env.reset()
        while (done is False):
            action_1 = agent_1.choose_action(state)
            new_state_1, reward_1, done, info = env.step(action_1)
            #Campo da gioco pieno. Non addestrare
            if(info.get("placed") == None):
                pass
            #Spazi disponibili e mossa valida. Addestra
            elif(info.get("placed") is True):
                agent_1.learn(state, action_1, reward_1, new_state_1, done)
            #Spazi disponibili e mossa non valida. Fai un'altra mossa e addestra
            else:
                while(info.get("placed") is False):
                    action_1 = agent_1.choose_action(state)
                    new_state_1, reward_1, done, info = env.step(action_1)
                    agent_1.learn(state, action_1, reward_1, new_state_1, done)
            if(done is True):
                print_winner(info.get("winner", None))
                time.sleep(2)
                break
            time.sleep(time_between_steps)
            state = new_state_1
            action_2 = agent_2.choose_action(state)
            new_state_2, reward_2, done, info = env.step(action_2)
            if(info.get("placed") == None):
                pass
            elif(info.get("placed") is True):
                agent_2.learn(state, action_2, reward_2, new_state_2, done)
            else:
                while(info.get("placed") is False):
                    action_2 = agent_2.choose_action(state)
                    new_state_2, reward_2, done, info = env.step(action_2)
                    agent_2.learn(state, action_2, reward_2, new_state_2, done)
            state = new_state_2
            if(done is True):
                print_winner(info.get("winner", None))
                time.sleep(2)
                break
            time.sleep(time_between_steps)
    agent_1.save_q_table(agent_1_q_table_name)
    agent_2.save_q_table(agent_2_q_table_name)

start_game(time_between_steps=0, time_between_episodes=0)
