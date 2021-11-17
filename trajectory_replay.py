import os
import glob
import csv
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection


def xy_from_rowcol(pos):
    return (pos[1], pos[0])

def goal_patch():
    patch = mpatches.Circle((0,0), 0.5)
    patch.set_color("green")
    patch.set_alpha(0.3)
    return patch

def update_goal_patch(goal, step_id):
    patch = goal["patch"]
    pos = goal["posns"][step_id]

    patch.set_center((pos[1] + 0.5, - pos[0] - 0.5))

def robot_patch():
    patch = mpatches.Rectangle((0,0), 0.75, -0.75)
    patch.set_alpha(0.3)
    return patch

def update_robot_patch(robot, step_id):
    patch = robot["patch"]
    pos = robot["posns"][step_id]

    patch.set_xy((pos[1] + 0.125,  - pos[0]  - 0.125))


def draw_state(fig, robots, goals, step_id):

    for goal in goals:
        update_goal_patch(goal, step_id)

    for robot in robots:
        update_robot_patch(robot, step_id)

    fig.canvas.draw()

def on_key(event):
    global fig, robots, goals, step_id, n_steps

    if event.key == "right":
        step_id += 1
        if step_id >= n_steps:
            step_id = n_steps - 1

        draw_state(fig, robots, goals, step_id)

    elif event.key == "left":
        step_id -= 1
        if step_id < 0:
            step_id = 0

        draw_state(fig, robots, goals, step_id)


traj_files = glob.glob("C:/Users/white/Documents/pCloud Sync/Research/rover_domain_data/MAG_TestB1/records/mhc/*.csv")

traj_file = traj_files[0]


robots = []
goals = []

with open(traj_file, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')

    mode = None

    for row in reader:
        if len(row) > 0:
            if row[0] == "n_rows":
                n_rows = int(row[1])
            elif row[0] == "n_cols":
                n_cols = int(row[1])
            elif row[0] == "n_steps":
                n_steps = int(row[1])
            elif row[0] == "n_goals":
                n_goals = int(row[1])
            elif row[0] == "n_robots":
                n_robots = int(row[1])

            elif row[0] == "Goal":
                mode = "Goal"
                goals.append({})
            elif row[0] == "Robot":
                mode = "Robot"
                robots.append({})

            elif mode == "Goal":
                if row[0] == "rows":
                    goals[-1]["rows"] = list(map(int, row[1:]))
                elif row[0] == "cols":
                    goals[-1]["cols"] = list(map(int, row[1:]))


            elif mode == "Robot":
                if row[0] == "rows":
                    robots[-1]["rows"] = list(map(int, row[1:]))
                elif row[0] == "cols":
                    robots[-1]["cols"] = list(map(int, row[1:]))


for goal in goals:
    rows = goal["rows"]
    cols = goal["cols"]
    goal["posns"] = [pos for pos in zip(rows, cols)]

    goal["patch"] = goal_patch()

for robot in robots:
    rows = robot["rows"]
    cols = robot["cols"]
    robot["posns"] = [pos for pos in zip(rows, cols)]

    robot["patch"] = robot_patch()



fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1], frameon=False)
ax.set_xlim(0, n_cols); ax.set_xticks([])
ax.set_ylim(-n_rows, 0); ax.set_yticks([])

step_id = 10

patches = [goal["patch"] for goal in goals] +  [robot["patch"] for robot in robots]
for patch in patches:
    ax.add_artist(patch)

draw_state(fig, robots, goals, step_id)
fig.canvas.mpl_connect('key_release_event', on_key)


plt.show()

