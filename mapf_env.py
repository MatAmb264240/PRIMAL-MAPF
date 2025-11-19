import numpy as np
import random
import gym
from gym import spaces
from collections import deque

class SimpleMAPFEnv(gym.Env):
    def __init__(self, grid_size=10, num_agents=4, fov_size=10, obstacle_density=0.2, max_steps=256):
        self.GRID_SIZE = grid_size
        self.NUM_AGENTS = num_agents
        self.FOV_SIZE = fov_size
        self.OBSTACLE_DENSITY = obstacle_density
        self.ACTIONS = [(0,0),(0,1),(0,-1),(1,0),(-1,0)]
        self.REWARD_MOVE = -0.3
        self.REWARD_STAY_OFF = -0.5
        self.REWARD_FINISH = 20.0
        self.REWARD_COLLISION = -5.0
        self.POSITIONS = None
        self.GOALS = None
        self.OBSTACLES = None
        self.last_positions = None
        self.finished = False
        self.max_steps = max_steps
        self.steps = 0
        
        fov_shape = (self.FOV_SIZE, self.FOV_SIZE, 4)
        self.action_space = spaces.MultiDiscrete([len(self.ACTIONS)] * self.NUM_AGENTS)
        self.observation_space = spaces.Dict({
            "fov": spaces.Box(low=0.0, high=1.0, shape=fov_shape, dtype=np.float32),
            "goal_vec": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "action_mask": spaces.Box(low=0.0, high=1.0, shape=(len(self.ACTIONS),), dtype=np.float32)
        })

    def _valid(self, pos):
        x,y = pos
        if x < 0 or y < 0 or x>=self.GRID_SIZE or y>=self.GRID_SIZE:
            return False
        return self.OBSTACLES[x,y] == 0

    def _connected(self, start, goal):
        q = deque([start])
        seen = {start}
        while q:
            x,y = q.popleft()
            if (x,y)==goal:
                return True
            for dx,dy in self.ACTIONS[1:]:
                nx,ny = x+dx,y+dy
                if 0<=nx<self.GRID_SIZE and 0<=ny<self.GRID_SIZE:
                    if self.OBSTACLES[nx,ny]==0 and (nx,ny) not in seen:
                        seen.add((nx,ny))
                        q.append((nx,ny))
        return False

    # Fix me --- to jest slaba bardzo nieoptymalna funkcja do generowania bo za kazdym razem gdy chociaz jeden agent nie bedzie miał ściezki to losuje WSZYSTKO od nowa
    ## I do tego jest BFS który też jest nieoptymalny do sprawdzania czy jest ściezka, na pewno jest lepszy sposób
    def _generate_map(self):
        while True:
            self.OBSTACLES = (np.random.rand(self.GRID_SIZE,self.GRID_SIZE) < self.OBSTACLE_DENSITY).astype(int)
            free = []
            for x in range(self.GRID_SIZE):
                for y in range(self.GRID_SIZE):
                    if self.OBSTACLES[x, y]==0:
                        free.append((x, y))

            if len(free) < 2*self.NUM_AGENTS:
                continue
            s = random.sample(free, 2*self.NUM_AGENTS)
            starts = s[:self.NUM_AGENTS]
            goals = s[self.NUM_AGENTS:]
            ok = True
            for i in range(self.NUM_AGENTS):
                if not self._connected(starts[i], goals[i]):
                    ok = False
                    break
            if ok:
                self.POSITIONS = list(starts)
                self.GOALS = list(goals)
                break

    # The first three channels are binary matrices indicating: (i) the presence of obstacles, (ii) the presence of other agents, and (iii) a projection of the goal
    def _cut_fov(self, agent):
        ax,ay = agent
        f = np.zeros((self.FOV_SIZE,self.FOV_SIZE,4),dtype=np.float32)
        hf = self.FOV_SIZE//2
        for i in range(self.FOV_SIZE):
            for j in range(self.FOV_SIZE):
                x = ax + (i - hf)
                y = ay + (j - hf)
                if x<0 or y<0 or x>=self.GRID_SIZE or y>=self.GRID_SIZE:
                    f[i,j,0] = 1
                    continue
                if self.OBSTACLES[x,y]==1:
                    f[i,j,0] = 1
                for p in self.POSITIONS:
                    if p==(x,y):
                        f[i,j,1] = 1
                gx,gy = self.GOALS[self.agent_tmp]
                if (x,y)==(gx,gy):
                    f[i,j,3] = 1
        for k,(gx,gy) in enumerate(self.GOALS):
            if k==self.agent_tmp:
                continue
            dx = gx-ax
            dy = gy-ay
            d = max(abs(dx),abs(dy))
            if d <= hf:
                ix = dx + hf
                iy = dy + hf
                if 0<=ix<self.FOV_SIZE and 0<=iy<self.FOV_SIZE:
                    f[ix,iy,2] = 1
            else:
                if abs(dx)>abs(dy):
                    ix = hf + (hf if dx>0 else -hf)
                    iy = int((dy/d)*(hf if dx>0 else -hf))+hf
                else:
                    iy = hf + (hf if dy>0 else -hf)
                    ix = int((dx/d)*(hf if dy>0 else -hf))+hf
                ix = max(0,min(self.FOV_SIZE-1,ix))
                iy = max(0,min(self.FOV_SIZE-1,iy))
                f[ix,iy,2] = 1
        return f

    def _action_mask(self, agent_id):
        px,py = self.POSITIONS[agent_id]
        lp = self.last_positions[agent_id]
        mask = np.zeros(len(self.ACTIONS), dtype=np.float32)
        for i,(dx,dy) in enumerate(self.ACTIONS):
            nx,ny = px+dx,py+dy
            if (nx,ny)==lp:
                continue
            if self._valid((nx,ny)):
                mask[i] = 1
        if mask.sum()==0:
            mask[0] = 1
        return mask

    def get_observation(self, agent_id):
        self.agent_tmp = agent_id
        px,py = self.POSITIONS[agent_id]
        gx,gy = self.GOALS[agent_id]
        fov = self._cut_fov((px,py))
        dx = gx-px
        dy = gy-py
        dist = np.sqrt(dx*dx + dy*dy)
        if dist==0:
            goal_vec = np.array([0,0,0],dtype=np.float32)
        else:
            goal_vec = np.array([dx/dist,dy/dist,dist],dtype=np.float32)
        mask = self._action_mask(agent_id)
        return {"fov": fov, "goal_vec": goal_vec, "action_mask": mask}

    def reset(self):
        self._generate_map()
        self.last_positions = list(self.POSITIONS)
        self.steps = 0
        self.finished = False
        return [self.get_observation(i) for i in range(self.NUM_AGENTS)]

    def step(self, actions):
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()
        self.steps += 1
        rewards = [0]*self.NUM_AGENTS
        done = False

        old_positions = list(self.POSITIONS)
        new_positions = list(self.POSITIONS)
        occupied = set(self.POSITIONS)
        order = np.random.permutation(self.NUM_AGENTS)

        for i in order:
            cur = old_positions[i]
            goal = self.GOALS[i]
            a = int(actions[i])
            if a < 0 or a >= len(self.ACTIONS):
                a = 0
            dx,dy = self.ACTIONS[a]
            if a==0:
                if cur==goal:
                    r = 0
                else:
                    r = self.REWARD_STAY_OFF
                new_pos = cur
            else:
                cand = (cur[0]+dx, cur[1]+dy)
                move_valid = self._valid(cand) and cand not in occupied
                if not move_valid:
                    r = self.REWARD_COLLISION
                    new_pos = cur
                else:
                    new_pos = cand
                    if new_pos==goal:
                        r = self.REWARD_FINISH
                    else:
                        r = self.REWARD_MOVE
                    occupied.add(new_pos)
                    occupied.discard(cur)
            new_positions[i] = new_pos
            rewards[i] = r

        self.last_positions = list(self.POSITIONS)
        self.POSITIONS = new_positions

        all_finish = True
        for i in range(self.NUM_AGENTS):
            if self.POSITIONS[i] != self.GOALS[i]:
                all_finish = False
                break

        if all_finish:
            done = True
            self.finished = True

        if self.steps >= self.max_steps:
            done = True

        obs = [self.get_observation(i) for i in range(self.NUM_AGENTS)]
        return obs, rewards, done, {}
