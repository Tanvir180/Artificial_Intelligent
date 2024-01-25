import numpy as np 
import pandas as pd 

import os
'''
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
'''     
#-------------------------------
import random
import seaborn as sn
import matplotlib.pyplot as plt
#--------------------
BOARD_ROWS = 5                     
BOARD_COLS = 5                      
WIN_STATE = (4, 4)                  
START = (0, 0)                      
HOLES = [(1,0),(1,3),(3,1),(4,2)]   

#----------------------------------------
total_episodes = 10000       
learning_rate = 0.5          
max_steps = 99              
gamma = 0.9                  
epsilon = 0.1                

#------------------------

class State:
    def __init__(self,x,y): # Initialize state with provided coordinates
        self.cordinates = (x,y)
        self.isEnd = False
        
    def getCoordinates(self):
        return self.cordinates
        
    def getReward(self):
        if self.cordinates == WIN_STATE: # Rward at win state is 10
            return 10 
        elif self.cordinates in HOLES:   # Reward for failing in any hole is -5 ie punishment
            return -5
        else:                            # Reward for each transition to a non terminal state is -1
            return -1
    
    def isEndFunc(self):
        if (self.cordinates == WIN_STATE):
            self.isEnd = True
    
    def conversion(self):               # Function to convert a cell location (2d space) to 1d space for q learning table
        return BOARD_COLS*self.cordinates[0]+self.cordinates[1]
    
    def nxtCordinates(self, action):    # Provides next coordinates based on action provides        
        if action == "up":                
            nxtState = (self.cordinates[0] - 1, self.cordinates[1])                
        elif action == "down":
            nxtState = (self.cordinates[0] + 1, self.cordinates[1])
        elif action == "left":
            nxtState = (self.cordinates[0], self.cordinates[1] - 1)
        else:
            nxtState = (self.cordinates[0], self.cordinates[1] + 1)
           
        if (nxtState[0] >= 0) and (nxtState[0] <= BOARD_ROWS-1):
            if (nxtState[1] >= 0) and (nxtState[1] <= BOARD_COLS-1):                    
                    return nxtState # if next state legal
        return self.cordinates # Any move off the grid leaves state unchanged


#-------------------------------------------------------

class Environment:
    def __init__(self,length,width):
        self.BOARD_ROWS = length
        self.BOARD_COLS = width
        
    # Setters and Getters to define winning state/location , start state/location and holes in the environment/lake    
    def setWinState(self,x,y):
        self.WIN_STATE = (x,y)
        
    def setStart(self,x,y):
        self.START = (x,y)
        
    def setHoles(self,holesarray):
        self.HOLES = holesarray
        
    def getWinState(self):
        return self.WIN_STATE
    
    def getStart(self):
        return self.START
    
    def getHoles(self):
        return self.HOLES
        
    def getSize(self):
        return self.BOARD_ROWS,self.BOARD_COLS

#-----------------------------------------------------

class Agent:
    def __init__(self):
        self.actions = ["up", "down", "left", "right"]                              # Four possible movement for agent
        self.env = Environment(BOARD_ROWS,BOARD_COLS)                               # Defining environment for agent
        self.env.setWinState(WIN_STATE[0],WIN_STATE[1])
        self.env.setStart(START[0],START[1])
        self.env.setHoles(HOLES)
        self.state_size,self.action_size = BOARD_ROWS*BOARD_COLS,len(self.actions)  # Defining state and action space
        self.qtable = np.zeros((self.state_size,self.action_size))                  # Defining Q table for policy learning
        self.rewards = []                                                           # To store rewards per episode
                   
    def printTable(self):
        # Utility fucntion to print Q learning table
        print("------------------- Q LEARNING TABLE ------------------")
        print(self.qtable)
        print("-------------------------------------------------------")
        
    def printPath(self):
        rows,cols = self.env.getSize()
        data = np.ones((rows,cols))*150                                             # Create a matrix to display in heatmap
        for hole in self.env.getHoles():
            data[hole[0],hole[1]] = 300                                             # Mark all the holes to represent in heatmap
        
        START = self.env.getStart()
        state = State(START[0],START[1])      
        while True:
            print("::: ",state.getCoordinates())
            coerd = state.getCoordinates()
            data[coerd[0],coerd[1]] = 50                                            # Mark the movement path to represent in heatmap
            if state.getCoordinates()[0]==self.env.getWinState()[0] and state.getCoordinates()[1]==self.env.getWinState()[1]:
                break
            old_state = state.conversion()
            action = np.argmax(self.qtable[old_state, :])                           # Perform action which gives maximum Q value 
            nextstate = state.nxtCordinates(self.actions[action])                   # Get coordinates of next state
            state = State(nextstate[0],nextstate[1])                                # Update the state for next cycle
                
        hm = sn.heatmap(data = data,linewidths=1,linecolor="black",cmap='Blues',cbar=False)
        #plt.show()  
        plt.show(block=False)
        plt.pause(0.001)                                                                # displaying the plotted heatmap
        
                 
    def q_learning(self):
        # Q-learning, which is said to be an off-policy temporal difference (TD) control algorithm
        START = self.env.getStart()                                                 # reset the environment
        
        for episode in range(total_episodes):
            state = State(START[0],START[1])
            total_rewards = 0                                                       # total reward collected per episode 
            for step in range(max_steps):
                exp_exp_tradeoff = random.uniform(0, 1)                             # First we randomize a number
                old_state  = state.conversion()
        
                if exp_exp_tradeoff > epsilon:                                      # If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state) 
                    action = np.argmax(self.qtable[old_state,:])
                else:                                                               # Else doing a random choice --> exploration
                    action = random.randint(0,len(self.actions)-1)
                           
                nextState = state.nxtCordinates(self.actions[action])
                new_state  = State(nextState[0],nextState[1]).conversion()
                reward = state.getReward()
                total_rewards += reward                                            # Capture reard collected in this step in overall reward of episode
                # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]  :  Q learning equation
                self.qtable[old_state, action] = self.qtable[old_state, action] + learning_rate * (reward + gamma * np.max(self.qtable[new_state, :]) - self.qtable[old_state, action])
                state = State(nextState[0],nextState[1])                           # Update the state
            
            #epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) # Epsilon can be resuce with time to reduce exploration and focus on exploitation
            self.rewards.append(total_rewards)
            
            
    def plotReward(self):
        # Utility function to plot Reward collected wrt to episodes
        plt.figure(figsize=(12,5))
        plt.plot(range(total_episodes),self.rewards,color='red')
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward per Epidode')
        #plt.show()
        plt.show(block=False)
        plt.pause(0.001)
        
        
#-------------------------------------------------------------------------

ag = Agent()
ag.q_learning()
ag.printTable()


#-----------------------------------



print("The path for maximum reward:\n")
ag.printPath() 
ag.plotReward()
plt.show()
