#!/usr/bin/env python

import rvo2

from matplotlib import animation
import matplotlib.pyplot as plt

class state:
    def __init__(self):
        self.px = 0.0
        self.py = 0.0
        self.position = (self.px, self.py)

    def set_pos(self):
        self.position = (self.px, self.py)

sim = rvo2.PyRVOSimulator(1/60., 1.5, 5, 1.5, 2, 0.4, 2)

# Pass either just the position (the other parameters then use
# the default values passed to the PyRVOSimulator constructor),
# or pass all available parameters.
a0 = sim.addAgent((0, 0))
a1 = sim.addAgent((0, 4))
# a2 = sim.addAgent((7, 7))
# a3 = sim.addAgent((0, 8), 1.5, 5, 1.5, 2, 0.4, 2, (0, 0))

# Obstacles are also supported.
# o1 = sim.addObstacle([(6, 7), (-5, 4), (-8, -1)])
# sim.processObstacles()

sim.setAgentPrefVelocity(a0, (0, 1))
sim.setAgentPrefVelocity(a1, (0, -1))
# sim.setAgentPrefVelocity(a2, (-1, -1))
# sim.setAgentPrefVelocity(a3, (1, -1))

print('Simulation has %i agents and %i obstacle vertices in it.' %
      (sim.getNumAgents(), sim.getNumObstacleVertices()))

print('Running simulation')

state1 = list()
state2 = list()
state3 = list()
state4 = list()

fig, ax = plt.subplots(figsize=(7, 7))
ax.tick_params(labelsize=16)
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_xlabel('x(m)', fontsize=16)
ax.set_ylabel('y(m)', fontsize=16)

for step in range(500):
    sim.doStep()

    position1 = (sim.getAgentPosition(a0)[0], sim.getAgentPosition(a0)[1])
    state1.append(position1)
    position2 = (sim.getAgentPosition(a1)[0], sim.getAgentPosition(a1)[1])
    state2.append(position2)
    # position3 = (sim.getAgentPosition(a2)[0], sim.getAgentPosition(a2)[1])
    # state3.append(position3)
    # position4 = (sim.getAgentPosition(a3)[0], sim.getAgentPosition(a3)[1])
    # state4.append(position4)
    # states = [(sim.getAgentPosition(agent_no)[0], sim.getAgentPosition(agent_no)[1]) for agent_no in (a0, a1, a2, a3)]
    # print('step=%2i  t=%.3f  %s' % (step, sim.getGlobalTime(), '  '.join(positions)))
for k in range(len(state1)):
    if k % 4 == 0 or k == len(state1) - 1:
        robot = plt.Circle(state1[k], 0.2, fill=True, color="red")
        ax.add_artist(robot)

for k in range(len(state2)):
    if k % 4 == 0 or k == len(state2) - 1:
        robot = plt.Circle(state2[k], 0.2, fill=True, color="blue")
        ax.add_artist(robot)

# for k in state3:
#     robot = plt.Circle(k, 0.4, fill=True, color="red")
#     ax.add_artist(robot)
#
# for k in state4:
#     robot = plt.Circle(k, 0.4, fill=True, color="red")
#     ax.add_artist(robot)

plt.show()
# fig, ax = plt.subplots(figsize=(7, 7))
#             ax.tick_params(labelsize=16)
#             ax.set_xlim(-5, 5)
#             ax.set_ylim(-5, 5)
#             ax.set_xlabel('x(m)', fontsize=16)
#             ax.set_ylabel('y(m)', fontsize=16)
#
#             robot_positions = [self.states[i][0].position for i in range(len(self.states))]
#             human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))]
#                                for i in range(len(self.states))]
#             for k in range(len(self.states)):
#                 if k % 4 == 0 or k == len(self.states) - 1:
#                     robot = plt.Circle(robot_positions[k], self.robot.radius, fill=True, color=robot_color)
#                     humans = [plt.Circle(human_positions[k][i], self.humans[i].radius, fill=False, color=cmap(i))
#                               for i in range(len(self.humans))]
#                     ax.add_artist(robot)
#                     for human in humans:
#                         ax.add_artist(human)
#                 # add time annotation
#                 global_time = k * self.time_step
#                 if global_time % 4 == 0 or k == len(self.states) - 1:
#                     agents = humans + [robot]
#                     times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset,
#                                       '{:.1f}'.format(global_time),
#                                       color='black', fontsize=14) for i in range(self.human_num + 1)]
#                     for time in times:
#                         ax.add_artist(time)
#                 if k != 0:
#                     nav_direction = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
#                                                (self.states[k - 1][0].py, self.states[k][0].py),
#                                                color=robot_color, ls='solid')
#                     human_directions = [plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
#                                                    (self.states[k - 1][1][i].py, self.states[k][1][i].py),
#                                                    color=cmap(i), ls='solid')
#                                         for i in range(self.human_num)]
#                     ax.add_artist(nav_direction)
#                     for human_direction in human_directions:
#                         ax.add_artist(human_direction)
#             plt.legend([robot], ['Robot'], fontsize=16)
#             plt.show()

