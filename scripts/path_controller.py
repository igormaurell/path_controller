#! /usr/bin/env python
from operator import pos
from threading import local
from typing import final
from PIL.Image import LINEAR
from numpy.lib.type_check import imag
import rospy 
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
import random
import math
from gazebo_msgs.msg import *
import numpy as np
import csv
import rospkg
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from environment import Env

X_POSITIVE_AXIS_SIZE = 3
X_NEGATIVE_AXIS_SIZE = 3
Y_POSITIVE_AXIS_SIZE = 3
Y_NEGATIVE_AXIS_SIZE = 3
VOXEL_SIZE = 0.05

DIST_TOLERANCE = VOXEL_SIZE/2

ANGLE_TOLERANCE = 5

ANGULAR_VEL = 0.2

LINEAR_VEL = 0.1

COLLISION_TOLERANCE = 0.13

def WIDTH():
    return X_NEGATIVE_AXIS_SIZE + X_POSITIVE_AXIS_SIZE

def HEIGTH():
    return Y_NEGATIVE_AXIS_SIZE + Y_POSITIVE_AXIS_SIZE

def mountMap():
    rows = int(HEIGTH()/VOXEL_SIZE)
    cols = int(WIDTH()/VOXEL_SIZE)

    return np.zeros(shape = (rows, cols), dtype = int)

def discretizePoint(point):
    i = int((Y_NEGATIVE_AXIS_SIZE + point[0])/VOXEL_SIZE)
    j = int((X_NEGATIVE_AXIS_SIZE - point[1])/VOXEL_SIZE)

    rows = int(HEIGTH()/VOXEL_SIZE)
    cols = int(WIDTH()/VOXEL_SIZE)
    
    if i >= rows or j >= cols or i < 0 or j < 0:
        rospy.loginfo('Point {} is beyond the map limits.'.format(point))
        #map growing
        pass

    return i, j

def markMapPoint(point, map, value = 1):
    i, j = discretizePoint(point)
    markMapDiscretizedPoint((i, j), map, value)

def markMapDiscretizedPoint(point_d, map, value = 1):
    if point_d[0] >= 0 and point_d[1] >= 0 and point_d[0] < map.shape[0] and point_d[1] < map.shape[1]:
        map[point_d[0], point_d[1]] = value

def markLaserScan(ranges, map, position, angle):
    angle_increment = 2*math.pi/ranges.shape[0]
    for i, r in enumerate(ranges):
        if r > 0 and r < 3.5:
            a = i*angle_increment
            if angle < 0:
                a = a + 2*math.pi + angle
            elif angle > 0:
                a = a + angle
        
            point = np.array([r*math.cos(a) + position[0], r*math.sin(a) + position[1]])

            point_d = discretizePoint(point)

            markMapDiscretizedPoint(point_d, map, value = 1)
            
            #marking danger-zone
            kernel_size = math.ceil(COLLISION_TOLERANCE/VOXEL_SIZE)
            kernel = list(range(-kernel_size,kernel_size+1))
            for dx in kernel:
                for dy in kernel:
                    if dx != 0 or dy != 0:
                        ni = point_d[0] + dy
                        nj = point_d[1] + dx
                        if (ni >= 0 and nj >= 0) and (ni < map.shape[0] and nj < map.shape[1]) and (map[ni, nj] == 0 or map[ni, nj] == 4):
                            markMapDiscretizedPoint((ni, nj), map, value = 2)

def calculateVoxelCentroidPoint(i, j):
    point_m = [-1, -1]
    point_M = [-1, -1]

    point_m[0] = i*VOXEL_SIZE - Y_NEGATIVE_AXIS_SIZE
    point_m[1] = X_NEGATIVE_AXIS_SIZE - j*VOXEL_SIZE

    point_M[0] = (i+1)*VOXEL_SIZE - Y_NEGATIVE_AXIS_SIZE
    point_M[1] = X_NEGATIVE_AXIS_SIZE - (j+1)*VOXEL_SIZE

    centroid = ((point_m[0] + point_M[0])/2, (point_m[1] + point_M[1])/2)
    
    return centroid

def euclidianDistance(A, B):
    return math.sqrt((B[0] - A[0])**2 + (B[1] - A[1])**2)

def computeHFunction(goal, shape):
    rows = shape[0]
    cols = shape[1]
    size = rows*cols
    h_function = np.ones(size)

    for i in range(0, size):
        pos_d = (i//cols, i%cols)
        pos = calculateVoxelCentroidPoint(pos_d[0], pos_d[1])
        h_function[i] = euclidianDistance(pos, goal)
    
    return h_function

def AStarPathSearch(position, goal, map, h_function):
    rospy.loginfo('Finding a Path.....')

    path = []

    position_d = discretizePoint(position)
    if position_d[0] == -1 or position_d[1] == -1:
        return []

    goal_d = discretizePoint(goal)
    if goal_d[0] == -1 or goal_d[1] == -1:
        return []

    rows = map.shape[0]
    cols = map.shape[1]

    open_map = np.ones(rows*cols)*np.inf
    open_map[position_d[0]*cols + position_d[1]] = 0.0
    open_map_size = 1

    g_function = np.ones(rows*cols)*np.inf
    g_function[position_d[0]*cols + position_d[1]] = 0.0

    closed_map = np.ones(rows*cols)*np.inf

    nodes = np.ones(rows*cols, dtype = int)*-1

    final_goal = -1
    
    while open_map_size > 0:
        p = np.argmin(open_map)
        posi = (p//cols, p%cols)
        g = g_function[p]
        f = open_map[p]
        g_function[p] = np.inf
        open_map[p] = np.inf

        if posi == goal_d:
            final_goal = p
            break

        neigh = []
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                if dx != 0 or dy != 0:
                    ni = posi[0] + dy
                    nj = posi[1] + dx
                    if (ni >= 0 and nj >= 0) and (ni < map.shape[0] and nj < map.shape[1]):
                        if (map[ni, nj] != 1 and map[ni, nj] != 2) or (ni, nj) == goal_d:
                            neigh.append((ni, nj))

        for n in neigh:
            n_p = n[0]*cols + n[1]

            #diagonal movement has a higher cost
            if posi[0] == n[0] or posi[1] == n[1]:
                n_g = g + VOXEL_SIZE
            else:
                n_g = g + math.sqrt(2)*VOXEL_SIZE

            n_h = h_function[n_p]

            n_f = n_g + n_h

            if n_f < open_map[n_p] and n_f < closed_map[n_p]:
                if open_map[n_p] == np.inf:
                    open_map_size += 1
                nodes[n_p] = p
                g_function[n_p] = n_g
                open_map[n_p] = n_f
        
        closed_map[p] = f
                            

    while final_goal != -1:
        index = (final_goal//cols, final_goal%cols)
        path.append(index)
        final_goal = nodes[final_goal]
    
    rospy.loginfo('Path Found.')
    return path[::-1][1:]

def isValidPath(path, map):
    if len(path) == 0:
        return False
    for p in path:
        if map[p[0], p[1]] == 1 or map[p[0], p[1]] == 2:
            return False
    return True

def robotAlign(position, angle, goal, last_ang_vel):
    pg = np.array([goal[0] - position[0], goal[1] - position[1]])
    x_axis = np.array([1, 0])

    final_angle = math.acos(np.dot(pg, x_axis)/(np.linalg.norm(pg, ord=2)*np.linalg.norm(x_axis, ord=2)))
    if pg[1] < 0:
        final_angle = -final_angle
    
    if angle*final_angle >= 0:
        diff = final_angle - angle
    else:
        if abs(angle) + abs(final_angle) >= math.pi:
            diff = 2*math.pi - (abs(angle) + abs(final_angle))
            if final_angle > 0:
                diff = -diff
        else:
            diff = abs(angle) + abs(final_angle)
            if final_angle < 0:
                diff = -diff
    
    if abs(diff) < ANGLE_TOLERANCE*math.pi/180 or diff*last_ang_vel < 0:
        return 0

    if diff < 0:
        return -ANGULAR_VEL
    
    return ANGULAR_VEL
 
def move(position, angle, goal, goal_dir, last_action):
    action = [0, 0]

    pg = np.array([goal[0] - position[0], goal[1] - position[1]])
    
    distance = np.dot(pg, goal_dir)

    if abs(distance) < DIST_TOLERANCE or distance < 0:
        return [0, 0]

    action[1] = robotAlign(position, angle, goal, last_action[1])
    if action[1] == 0:
        action[0] = 0.2 * distance
        if action[0] > 0.2:
            action[0] = 0.2
        if action[0] < 0.05:
            action[0] = 0.05

    return action

pos_goal_dir = None
def moveByPath(path, position, angle, last_action, map):
    global pos_goal_dir
    action = [0, 0]
    if len(path) == 0:
        return action        
    
    position_d = discretizePoint(position)

    goal_d = path[0]

    goal = calculateVoxelCentroidPoint(goal_d[0], goal_d[1])

    if pos_goal_dir is None:
        pos_goal_dir = np.array([goal[0] - position[0], goal[1] - position[1]])
        pos_goal_dir = pos_goal_dir/np.linalg.norm(pos_goal_dir, ord=2)

    action = move(position, angle, goal, pos_goal_dir, last_action)


    if action == [0, 0]:
        path.pop(0)
        pos_goal_dir = None
        return [0, 0]
    
    return action

def simplifyPath(path):
    print('SIMPLIFING PATH')
    if len(path) <= 1:
        return path
    simplified_path = [path[0]]
    last_direction = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    print('last_dir:', last_direction)
    i = 1
    while i < len(path) - 1:
        p1 = path[i]
        p2 = path[i+1]
        print('p1:', p1)
        print('p2:', p1)
        direction = (p2[0] - p1[0], p2[1] - p1[1])
        print('dir:', direction)
        if direction != last_direction:
            print('added')
            if simplified_path[-1] != p1:
                simplified_path.append(p1)
            simplified_path.append(p2)
        last_direction = direction
        i += 1
    simplified_path.append(path[-1])
    return simplified_path

if __name__ == "__main__": 
    rospy.init_node("path_controller_node", anonymous=False)
    
    env = Env()
    state_scan = env.reset()
    action = np.zeros(2)
    path = []
    simp_path = []

    h_function = None

    #0 = free | 1 = obstacle | 2 = danger zone | 3 = goal | 4 = path | 5 = current_position 
    local_map = mountMap()

    plt.axis([0,local_map.shape[0],0,local_map.shape[1]])
    plt.ion()
    plt.show()

    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)    
    r = rospy.Rate(5) # 10hz
    velocity = Twist()
    while not rospy.is_shutdown():
        if env.getGoalDistace() >= 0.2:
            position = (env.position.x, env.position.y)
            angle = env.yaw

            goal = (env.goal_x, env.goal_y)

            if action[0] == 0 and action[1] == 0:
                local_map = mountMap()

                markLaserScan(state_scan, local_map, position, angle)

            if h_function is None:
                h_function = computeHFunction(goal, local_map.shape)
    
            if not isValidPath(path, local_map):
                env.step([0, 0])
                path = AStarPathSearch(position, goal, local_map, h_function)
                simp_path = simplifyPath(path)
                if len(path) == 0:
                    rospy.logerr('There is no way to go to the point, cleaning the map.')

            for p in path:
                markMapDiscretizedPoint(p, local_map, value = 4)
            for p in simp_path:
                markMapDiscretizedPoint(p, local_map, value = 6)    
            
            markMapPoint(position, local_map, value = 5)
            markMapPoint(goal, local_map, value = 3)
            
            action = moveByPath(simp_path, position, angle, action, local_map)

            state_scan = env.step(action)

            plt.imshow(local_map)
            plt.draw()
            plt.pause(0.0001)

            markMapPoint(position, local_map, value = 0)
            markMapPoint(goal, local_map, value = 0)
            for p in path:
                markMapDiscretizedPoint(p, local_map, value = 0)
            for p in simp_path:
                markMapDiscretizedPoint(p, local_map, value = 0)    
            markMapPoint(goal, local_map, value = 0)

        else:
            rospy.loginfo('Arrive to the goal, waiting for the next.')
            path = []
            simp_path = []
            h_function = None
            action[0] = 0
            action[1] = 0
            state_scan = env.step(action)

        r.sleep()