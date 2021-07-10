#! /usr/bin/env python
from typing import final
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
VOXEL_SIZE = 0.1

COLLISION_TOLERANCE = 0.2

def ANGULAR_ACELERATION(diff):
    return 0.2*math.log(abs(diff))

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
        #map cleaning (remove past observations that appear to be free)

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
    
        # length = np.linalg.norm(point_e)
        # direction = point_e/length

        # distance, proj = distanceProjPointLine(p_target, direction)
        # if distance > max_dist:
        #     max_dist = distance

        # if proj > 0 and length + length_tolerance >= proj and distance < min_distance:
        #     possible_bins.append(start+i)

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


def AStarPathSearch(position, goal, map):
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
        #print('\nLOOP:')
        p = np.argmin(open_map)
        #print('p:',p)
        posi = (p//cols, p%cols)
        #print('posi:',posi)
        g = g_function[p]
        f = open_map[p]
        #print('f:', f)
        g_function[p] = np.inf
        open_map[p] = np.inf
        # print('open_list:', open_list)

        if posi == goal_d:
            final_goal = p
            break

        neigh = []
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                if dx != 0 or dy != 0:
                    ni = posi[0] + dy
                    nj = posi[1] + dx
                    if (ni >= 0 and nj >= 0) and (ni < map.shape[0] and nj < map.shape[1]) and (map[ni, nj] != 1 and map[ni, nj] != 2):
                        neigh.append((ni, nj))
        #print('neigh:', neigh)

        for n in neigh:
            #print('\tcurrent n:', n)
            n_p = n[0]*cols + n[1]

            #diagonal movement has a higher cost
            if posi[0] == n[0] or posi[1] == n[1]:
                n_g = g + VOXEL_SIZE
            else:
                n_g = g + math.sqrt(2)*VOXEL_SIZE

            n_centroid = calculateVoxelCentroidPoint(n[0], n[1])
            n_h = euclidianDistance(n_centroid, goal)

            n_f = n_g + n_h
            #print('\tncentroid:', n_centroid)

            #print('\tnp, ng, nh, n_f:', n_p, n_g, n_h, n_f)

            if n_f < open_map[n_p] and n_f < closed_map[n_p]:
                #print('\tAdded.')
                if open_map[n_p] == np.inf:
                    open_map_size += 1
                nodes[n_p] = p
                g_function[n_p] = n_g
                open_map[n_p] = n_f
            #print()
        
        
        closed_map[p] = f
                            

    while final_goal != -1:
        index = (final_goal//cols, final_goal%cols)
        path.append(index)
        final_goal = nodes[final_goal]
    
    rospy.loginfo('Path Found.')
    return path[::-1]

def validPath(path, map):
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

    return last_ang_vel**2 + 2*ANGULAR_ACELERATION(diff)*diff

def robotGoStraight(position, goal):
    #print(goal)
    pg = np.array([goal[0] - position[0], goal[1] - position[1]])
    distance = np.linalg.norm(pg, ord = 2)

    #print('distance:',distance)

    #TODO: here, the robot may pass the point
    if distance > 0.2:
        return 0.1
    else:
        return 0

def move(position, angle, goal, last_action):
    action = [0, 0]
    action[1] = robotAlign(position, angle, goal, last_action[1])
    if action[1] == 0:
        action[0] = robotGoStraight(position, goal)

    return action

def moveByPath(path, position, angle, last_action):
    action = [0, 0]
    if len(path) == 0:
        return action
    
    goal_d = path[0]
    goal = calculateVoxelCentroidPoint(goal_d[0], goal_d[1])
    
    action = move(position, angle, goal, last_action)
    if action == [0, 0]:
        path.pop(0)
        return moveByPath(path, position, angle, last_action)

    return action

if __name__ == "__main__": 
    rospy.init_node("path_controller_node", anonymous=False)
    
    env = Env()
    state_scan = env.reset()
    action = np.zeros(2)
    path = []
    
    #0 = free | 1 = obstacle | 2 = danger zone | 3 = goal | 4 = path | 5 = current_position
    map = mountMap()

    plt.axis([0,map.shape[0],0,map.shape[1]])
    plt.ion()
    plt.show()

    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)    
    r = rospy.Rate(5) # 10hz
    velocity = Twist()
    while not rospy.is_shutdown():

        # action[0] = 0.1
        # action[1] = 0.1

        position = (env.position.x, env.position.y)
        angle = env.yaw

        goal = (env.goal_x, env.goal_y)

        #print(angle*180/math.pi)

        markLaserScan(state_scan, map, position, angle)

        markMapPoint(position, map, value = 5)
        markMapPoint(goal, map, value = 3)

        if not validPath(path, map):
            path = AStarPathSearch(position, goal, map)

        for p in path:
            markMapDiscretizedPoint(p, map, value = 4)    
        
        action = moveByPath(path, position, angle, action)

        state_scan = env.step(action)

        plt.imshow(map)
        plt.draw()
        plt.pause(0.0001)

        markMapPoint(position, map, value = 0)
        markMapPoint(goal, map, value = 0)
        for p in path:
            if map[p[0], p[1]] == 4:
                markMapDiscretizedPoint(p, map, value = 0)

        r.sleep()