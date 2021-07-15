#! /usr/bin/env python
from logging import getLogger
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
from scipy.stats import circmean
from environment import Env

X_SEMI_AXIS_SIZE = 1
Y_SEMI_AXIS_SIZE = 1
VOXEL_SIZE = 0.05

DIST_TOLERANCE = 0.1

ANGLE_INCREMENT = 0

ANGLE_TOLERANCE = 8

ANGULAR_VEL_MIN = 0.05
ANGULAR_VEL_MAX = 0.5

ANGULAR_ACC = 0.2
ANGULAR_DCC = -0.3
DIFF_DCC = 20

LINEAR_VEL_MIN = 0.05
LINEAR_VEL_MAX = 0.15

LINEAR_ACC = 0.15
LINEAR_DCC = -0.15
DIST_DCC = 0.2

REACT_LINEAR_VEL_MAX = 0.3
REACT_LINEAR_ACC = 0.3

REACT_ANGULAR_VEL = 0.8

COLLISION_TOLERANCE = 0.13
DANGER_TOLERANCE = 0.28
REACT_TOLERANCE = 0.23

MAX_PATH_AGE = 15

REACT = False

MAP_RESIZE = None

env = Env()

#MAP

def WIDTH():
    return 2*Y_SEMI_AXIS_SIZE

def HEIGTH():
    return 2*X_SEMI_AXIS_SIZE

def mountMap():
    rows = int(HEIGTH()/VOXEL_SIZE)
    cols = int(WIDTH()/VOXEL_SIZE)

    return np.zeros(shape = (rows, cols), dtype = int)

def mapGrowth():
    global X_SEMI_AXIS_SIZE, Y_SEMI_AXIS_SIZE, MAP_RESIZE
    if MAP_RESIZE[0] > X_SEMI_AXIS_SIZE:
        X_SEMI_AXIS_SIZE = MAP_RESIZE[0]
    if MAP_RESIZE[1] > Y_SEMI_AXIS_SIZE:
        Y_SEMI_AXIS_SIZE = MAP_RESIZE[1]
    
    MAP_RESIZE = None
    
def discretizePoint(point):
    global MAP_RESIZE
    i = int((X_SEMI_AXIS_SIZE + point[0])/VOXEL_SIZE)
    j = int((Y_SEMI_AXIS_SIZE - point[1])/VOXEL_SIZE)

    rows = int(HEIGTH()/VOXEL_SIZE)
    cols = int(WIDTH()/VOXEL_SIZE)
    
    if i >= rows or j >= cols or i < 0 or j < 0:
        #rospy.loginfo('Point {} is beyond the map limits.'.format(point))
        x_s = int(math.ceil(abs(point[0])))
        y_s = int(math.ceil(abs(point[1])))
    
        if MAP_RESIZE is not None:
            if x_s > MAP_RESIZE[0]:
                MAP_RESIZE[0] = x_s
            if y_s > MAP_RESIZE[1]:
                MAP_RESIZE[1] = y_s
        else:
            MAP_RESIZE = [x_s, y_s]

    return i, j

def markMapPoint(point, map, value = 1):
    i, j = discretizePoint(point)
    markMapDiscretizedPoint((i, j), map, value)

def markMapDiscretizedPoint(point_d, map, value = 1):
    if point_d[0] >= 0 and point_d[1] >= 0 and point_d[0] < map.shape[0] and point_d[1] < map.shape[1]:
        map[point_d[0], point_d[1]] = value

def markLaserScan(ranges, map, position, angle, danger_zone_size = DANGER_TOLERANCE):
    for i, r in enumerate(ranges):
        if r > 0 and r < 3.5:
            a = i*ANGLE_INCREMENT
            if angle < 0:
                a = a + 2*math.pi + angle
            elif angle > 0:
                a = a + angle
        
            point = np.array([r*math.cos(a) + position[0], r*math.sin(a) + position[1]])

            point_d = discretizePoint(point)

            markMapDiscretizedPoint(point_d, map, value = 1)
            
            #marking danger-zone
            collision_kernel_size = math.ceil(COLLISION_TOLERANCE/VOXEL_SIZE)
            kernel_size = math.ceil(danger_zone_size/VOXEL_SIZE)
            kernel = list(range(-kernel_size,kernel_size+1))
            for dx in kernel:
                for dy in kernel:
                    if dx != 0 or dy != 0:
                        ni = point_d[0] + dy
                        nj = point_d[1] + dx
                        if (ni >= 0 and nj >= 0) and (ni < map.shape[0] and nj < map.shape[1]) and map[ni, nj] != 1:
                            if (abs(dx) > collision_kernel_size or abs(dy) > collision_kernel_size) and map[ni, nj] != 2:
                                markMapDiscretizedPoint((ni, nj), map, value = 3)
                            else:
                                markMapDiscretizedPoint((ni, nj), map, value = 2)

def calculateVoxelCentroidPoint(i, j):
    point_m = [-1, -1]
    point_M = [-1, -1]

    point_m[0] = i*VOXEL_SIZE - X_SEMI_AXIS_SIZE
    point_m[1] = Y_SEMI_AXIS_SIZE - j*VOXEL_SIZE

    point_M[0] = (i+1)*VOXEL_SIZE - X_SEMI_AXIS_SIZE
    point_M[1] = Y_SEMI_AXIS_SIZE - (j+1)*VOXEL_SIZE

    centroid = ((point_m[0] + point_M[0])/2, (point_m[1] + point_M[1])/2)
    
    return centroid


#PATH

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

def getBestFreeDiscretizedPosition(position_d, map, free_danger_zone):
    rows = map.shape[0]
    cols = map.shape[1]

    ks = 0
    new_position_d = None
    while map[position_d[0], position_d[1]] == 2 or (not free_danger_zone and map[position_d[0], position_d[1]] == 3):
        kernels = []
        if ks == 0:
            kernels.append([0])
        else:
            kernels.append([-ks,ks])
        kernels.append(list(range(-ks,ks+1)))
        for i in range(0, 2):
            for dx in kernels[i]:
                for dy in kernels[i-1]:
                    if REACT:
                        None
                    p_position_d = (position_d[0] + dy, position_d[1] + dx)
                    if map[p_position_d[0], p_position_d[1]] == 0:
                        if new_position_d is None:
                            new_position_d = p_position_d
                        elif h_function[p_position_d[0]*cols + p_position_d[1]] < h_function[new_position_d[0]*cols + new_position_d[1]]:
                            new_position_d = p_position_d
        if new_position_d is not None:
            position_d = new_position_d
        else:
            ks += 1
    if new_position_d is None:
        return position_d

    return new_position_d

def AStarPathSearch(position, goal, map, h_function, free_danger_zone = False):
    path = []

    rows = map.shape[0]
    cols = map.shape[1]

    position_d = discretizePoint(position)
    if position_d[0] == -1 or position_d[1] == -1:
        return [], False

    position_d = getBestFreeDiscretizedPosition(position_d, map, free_danger_zone)

    goal_d = discretizePoint(goal)
    if goal_d[0] == -1 or goal_d[1] == -1:
        return [], False
        
    new_goal_d = getBestFreeDiscretizedPosition(goal_d, map, free_danger_zone)

    if h_function[new_goal_d[0]*cols + new_goal_d[1]] < 0.2:
        goal_d = new_goal_d
    else:
        free_danger_zone = True
        new_goal_d = getBestFreeDiscretizedPosition(goal_d, map, free_danger_zone)
        if h_function[new_goal_d[0]*cols + new_goal_d[1]] < 0.2:
            goal_d = new_goal_d
        else:
            return [], False
    
    if position_d is None or goal_d is None:
        return [], False

    open_map = np.ones(rows*cols)*np.inf
    open_map[position_d[0]*cols + position_d[1]] = 0.0
    open_map_size = 1

    g_function = np.ones(rows*cols)*np.inf
    g_function[position_d[0]*cols + position_d[1]] = 0.0

    closed_map = np.ones(rows*cols)*np.inf

    nodes = np.ones(rows*cols, dtype = int)*-1

    final_goal = -1
    
    while open_map_size > 0:
        if REACT == True:
            return [], False
        p = np.argmin(open_map)
        posi = (p//cols, p%cols)
        g = g_function[p]
        f = open_map[p]
        g_function[p] = np.inf
        open_map[p] = np.inf
        open_map_size -= 1

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
                        if (map[ni, nj] != 1) and (map[ni, nj] != 2 or (ni, nj) == goal_d) and (map[ni, nj] != 3 or free_danger_zone):
                            neigh.append((ni, nj))

        for n in neigh:
            n_p = n[0]*cols + n[1]

            #using the same g for diagonal and straight
            n_g = g + VOXEL_SIZE
        
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
    
    return path[::-1][1:], True

def isValidPath(path, map):
    if len(path) == 0:
        return False
    for p in path:
        if map[p[0], p[1]] == 1 or map[p[0], p[1]] == 2:
            return False
    return True

def projPointInLine(A, P, v):
    PA = A - P

    proj_dist = PA.dot(v)

    A_proj = P + proj_dist*v

    return A_proj

def superSimplifyPath(path, map):
    global REACT
    if len(path) <= 1:
        return path

    path = simplifyPath(path)

    rows = map.shape[0]
    cols = map.shape[1]

    path_c = []

    for p in path:
        path_c.append(calculateVoxelCentroidPoint(p[0], p[1]))
    
    simp_path = [path[0]]

    current_position = path[0]
    current_position_c = np.array(path_c[0])
    current_i = 0

    while current_position != path[-1]:
        for i in range(len(path) - 1, current_i, -1):
            goal_c = np.array(path_c[i])
            if i == current_i + 1:
                simp_path.append(path[i])
                current_position = path[i]
                current_position_c = goal_c
                current_i = i
                break
            v = goal_c - current_position_c
            distance_p_g = np.linalg.norm(v, ord=2)
            v = v/distance_p_g
            POSSIBLE = True
            for k in range(0, rows):
                for j in range(0, cols):
                    if map[k, j] == 1:
                        if REACT:
                            return path
                        P = np.array(calculateVoxelCentroidPoint(k, j))
                        P_proj = projPointInLine(P, current_position_c, v)
                        distance = np.linalg.norm(P - P_proj, ord=2)
                        dp = np.linalg.norm(current_position_c - P_proj, ord=2)
                        dg = np.linalg.norm(goal_c - P_proj, ord=2)
                        if distance <= (COLLISION_TOLERANCE + DIST_TOLERANCE) and dp < distance_p_g and dg < distance_p_g:
                            POSSIBLE = False
                            break
                if POSSIBLE == False:
                    break
            if POSSIBLE:
                simp_path.append(path[i])
                current_position = path[i]
                current_position_c = goal_c
                current_i = i
                break
        
    return simp_path

def simplifyPath(path):
    if len(path) <= 1:
        return path
    simplified_path = [path[0]]
    last_direction = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    i = 1
    while i < len(path) - 1:
        p1 = path[i]
        p2 = path[i+1]
        direction = (p2[0] - p1[0], p2[1] - p1[1])
        if direction != last_direction:
            if simplified_path[-1] != p1:
                simplified_path.append(p1)
            simplified_path.append(p2)
        last_direction = direction
        i += 1
    simplified_path.append(path[-1])
    return simplified_path


#NAVIGATION

last_torr_time = 0
def computeAngularVelocity(diff, vzero, init_vel = ANGULAR_VEL_MIN, min_vel = ANGULAR_VEL_MIN, max_vel = ANGULAR_VEL_MAX, acc = ANGULAR_ACC, dcc = ANGULAR_DCC):
    global last_torr_time
    sign = 1
    if diff < 0:
        sign = -1
    vel = 0
    now = rospy.get_rostime()
    secs = now.to_sec()
    vzero = abs(vzero)
    if vzero != 0:
        torr_diff = vzero*(secs-last_torr_time)
        if abs(diff) < DIFF_DCC*math.pi/180:
            vel2 = vzero**2 + 2*dcc*torr_diff
        else:
            vel2 = vzero**2 + 2*acc*torr_diff
        if vel2 < 0:
            vel2 = 0
        vel = math.sqrt(vel2)
    else:
        vel = init_vel
    
    if vel > max_vel:
        vel = max_vel
    elif vel < min_vel:
        vel = min_vel

    last_torr_time = secs
    return sign*vel

def computeLinearVelocity(distance, vzero, init_vel = LINEAR_VEL_MIN, min_vel = LINEAR_VEL_MIN, max_vel = LINEAR_VEL_MAX, acc = LINEAR_ACC, dcc = LINEAR_DCC):
    global last_torr_time
    vel = 0
    now = rospy.get_rostime()
    secs = now.to_sec()
    if vzero != 0:
        torr_distance = vzero*(secs-last_torr_time)
        if abs(distance) < DIST_DCC:
            vel2 = vzero**2 + 2*dcc*torr_distance
        else:
            vel2 = vzero**2 + 2*acc*torr_distance
        if vel2 < 0:
            vel2 = 0
        vel = math.sqrt(vel2)
    else:
        vel = init_vel

    if vel > max_vel:
        vel = max_vel
    elif vel < min_vel:
        vel = min_vel
    
    last_torr_time = secs
    return vel

def computeMinDiffAngles(a, b):
    if a*b >= 0:
        diff = b - a
    else:
        if abs(a) + abs(b) >= math.pi:
            diff = 2*math.pi - (abs(a) + abs(b))
            if b > 0:
                diff = -diff
        else:
            diff = abs(a) + abs(b)
            if b < 0:
                diff = -diff
    return diff

def robotAlign(diff, last_ang_vel): 
    if diff*last_ang_vel < 0 or abs(diff) < ANGLE_TOLERANCE*math.pi/180:
        return 0

    if last_ang_vel == 0 and abs(diff) > DIFF_DCC*math.pi/180:
        computeAngularVelocity(diff, last_ang_vel, init_vel=ANGULAR_VEL_MAX/2)

    return computeAngularVelocity(diff, last_ang_vel)

def robotGo(distance, last_lin_vel):
    if last_lin_vel == 0 and abs(distance) < DIST_DCC:
        return computeLinearVelocity(distance, last_lin_vel, init_vel=LINEAR_VEL_MIN*2)
    
    return computeLinearVelocity(distance, last_lin_vel)
 
def move(position, angle, goal, goal_dir, last_action):
    action = [0, 0]

    pg = np.array([goal[0] - position[0], goal[1] - position[1]])
    
    distance = np.dot(pg, goal_dir)

    if abs(distance) < DIST_TOLERANCE or distance < 0:
        return [0, 0], True

    x_axis = np.array([1, 0])
    final_angle = math.acos(np.dot(pg, x_axis)/(np.linalg.norm(pg, ord=2)*np.linalg.norm(x_axis, ord=2)))

    if pg[1] < 0:
        final_angle = -final_angle
    
    diff = computeMinDiffAngles(angle, final_angle)

    if last_action[1] != 0:
        action[1] = robotAlign(diff, last_action[1])
    elif last_action[0] != 0:
        if abs(diff) >= 2*ANGLE_TOLERANCE*math.pi/180 and last_action[0] == LINEAR_VEL_MIN:
            action[0] = 0
        elif abs(diff) >= 2*ANGLE_TOLERANCE*math.pi/180:
            action[0] = robotGo(DIST_DCC - DIST_DCC/2, last_action[0])
        else:
            action[0] = robotGo(distance, last_action[0])
    else:
        if abs(diff) > ANGLE_TOLERANCE*math.pi/180:
            action[1] = robotAlign(diff, last_action[1])
        else:
            action[0] = robotGo(distance, last_action[0])

    return action, False

pos_goal_dir = None
def moveByPath(path, position, angle, last_action):
    global pos_goal_dir
    action = [0, 0]
    if len(path) == 0:
        return action        
    
    goal_d = path[0]

    goal = calculateVoxelCentroidPoint(goal_d[0], goal_d[1])

    if pos_goal_dir is None:
        pos_goal_dir = np.array([goal[0] - position[0], goal[1] - position[1]])
        pos_goal_dir = pos_goal_dir/np.linalg.norm(pos_goal_dir, ord=2)

    action, done = move(position, angle, goal, pos_goal_dir, last_action)

    if done:
        path.pop(0)
        pos_goal_dir = None
        return [0, 0]
    
    return action

#REACTIVE

def computeDangerAngles(ranges):
    nozero = ranges > 0
    danger_arr = ranges <= REACT_TOLERANCE

    danger_arr = danger_arr*nozero

    danger_indices = []
    c = 0
    s = -1
    e = -1
    for i in range(0, danger_arr.shape[0]):
        if danger_arr[i] == True and s == -1:
            s = i
            e = i
        elif danger_arr[i] == False and s != -1:
            if c > ANGLE_TOLERANCE/(ANGLE_INCREMENT*180/math.pi):
                e -= c
                danger_indices.append(s + int((e - s)/2))
                s = -1
                e = -1
                c = 0
            else:
                e += 1
                c += 1
        elif danger_arr[i] == True:
            e += 1
            c = 0
    if s != -1:
        e -= c
        danger_indices.append(s + int((e - s)/2))
    
    danger_indices = np.array(danger_indices)

    danger_indice = int(circmean(danger_indices, len(ranges), 0))

    danger_angle = danger_indice*ANGLE_INCREMENT
    
    if danger_angle > math.pi:
        danger_angle = danger_angle - 2*math.pi
    
    return danger_angle

action = np.zeros(2)
def computeReactAction(ranges):
    global action

    danger_angle = computeDangerAngles(ranges)

    diff = danger_angle

    diff_to_escape = 0
    if abs(diff) <= math.pi/4:
        if action[0] > 0:
            action[0] = 0
        action[0] = -computeLinearVelocity(ranges[int(180*ANGLE_INCREMENT*180/math.pi)], abs(action[0]), init_vel = LINEAR_VEL_MAX, max_vel=REACT_LINEAR_VEL_MAX, acc = REACT_LINEAR_ACC)
        diff_to_escape = math.pi/2 - abs(diff)
        if diff < 0:
            diff_to_escape = -diff_to_escape
    elif abs(diff) <= math.pi/2:
        diff_to_escape = math.pi/2 - abs(diff)
        if diff < 0:
            diff_to_escape = -diff_to_escape
        if action[0] > 0:
            action[0] =  computeLinearVelocity(ranges[0], abs(action[0]), init_vel = LINEAR_VEL_MAX/2, max_vel=REACT_LINEAR_VEL_MAX, acc = REACT_LINEAR_ACC)
            diff_to_escape = -diff_to_escape
        else:
            action[0] = -computeLinearVelocity(ranges[int(180*ANGLE_INCREMENT*180/math.pi)], abs(action[0]), init_vel = LINEAR_VEL_MAX/2, max_vel=REACT_LINEAR_VEL_MAX, acc = REACT_LINEAR_ACC)
    elif abs(diff) <= 3*math.pi/4:
        diff_to_escape = abs(diff) - math.pi/2
        if diff > 0:
            diff_to_escape = -diff_to_escape
        if action[0] < 0:
            action[0] = -computeLinearVelocity(ranges[int(180*ANGLE_INCREMENT*180/math.pi)], abs(action[0]), init_vel = LINEAR_VEL_MAX/2, max_vel=REACT_LINEAR_VEL_MAX, acc = REACT_LINEAR_ACC)
            diff_to_escape = -diff_to_escape
        else:
            action[0] = computeLinearVelocity(ranges[0], abs(action[0]), init_vel = LINEAR_VEL_MAX/2, max_vel=REACT_LINEAR_VEL_MAX, acc = REACT_LINEAR_ACC)
    else:
        if action[0] < 0:
            action[0] = 0
        action[0] = computeLinearVelocity(ranges[0], abs(action[0]), init_vel = LINEAR_VEL_MAX, max_vel=REACT_LINEAR_VEL_MAX, acc = REACT_LINEAR_ACC)
        diff_to_escape = abs(diff) - math.pi/2
        if diff > 0:
            diff_to_escape = -diff_to_escape
    
    if diff_to_escape*action[1] >= 0:
        action[1] = computeAngularVelocity(diff_to_escape, action[1], init_vel = REACT_ANGULAR_VEL, max_vel = REACT_ANGULAR_VEL)
    else:
        action[1] = computeAngularVelocity(-diff_to_escape, action[1], init_vel = REACT_ANGULAR_VEL, max_vel = REACT_ANGULAR_VEL)

def reactivity(data):
    global REACT, action
    ranges = []
    for i in range(len(data.ranges)):
        if data.ranges[i] == float('Inf'):
            ranges.append(3.5)
        elif np.isnan(data.ranges[i]):
            ranges.append(0)
        else:
            ranges.append(data.ranges[i])
    ranges = np.array(ranges)
    i = np.argmin(ranges)
    if ranges[i] <= REACT_TOLERANCE and ranges[i] > 0:
        REACT = True
        computeReactAction(ranges)
        env.step(action, scan_data = data)
    else:
        if REACT == True:
            action = [0, 0]
            env.step(action, scan_data = data)
        REACT = False

if __name__ == "__main__": 
    rospy.init_node("path_controller_node", anonymous=False)
    
    sub_scan = rospy.Subscriber('scan', LaserScan, reactivity)
    state_scan = env.reset()
    ANGLE_INCREMENT = 2*math.pi/len(state_scan)
    path = []
    simp_path = []

    h_function = None

    #0 = free | 1 = obstacle | 2 = collision zone | 3 = danger_zone
    local_map = mountMap()
    #0 = free | 1 = obstacle | 2 = collision zone | 3 = danger zone | 4 = current_position | 5 = goal | 6 = path | 6 = simplified_path | 
    view_map = np.copy(local_map)

    last_goal = None

    path_age = 0
    path_tries = 0

    plt.axis([0,local_map.shape[1],0,local_map.shape[0]])
    plt.ion()
    plt.show()

    r = rospy.Rate(5) # 10hz
    velocity = Twist()
    while not rospy.is_shutdown():
        if REACT == True:
            path = []
            simp_path = []
            path_age = 0
            path_tries = 0

        else:
            position = (env.position.x, env.position.y)
            position_d  = discretizePoint(position)
            
            angle = env.yaw

            goal = (env.goal_x, env.goal_y)
            goal_d  = discretizePoint(goal)

            if last_goal is None or goal != last_goal:
                path = []
                simp_path = []
                h_function = computeHFunction(goal, local_map.shape)
                path_age = 0
                path_tries = 0
                action[0] = 0
                action[1] = 0
                state_scan = env.step(action)
                last_goal = goal

            if action[0] == 0 and action[1] == 0:
                local_map = mountMap()

                markLaserScan(state_scan, local_map, position, angle)

                view_map = np.copy(local_map)

                path_age += 1
            
            if MAP_RESIZE is not None:
                rospy.loginfo('Increasing Map Size(x, y): {}'.format(tuple(MAP_RESIZE)))
                mapGrowth()

                local_map = mountMap()
                view_map = mountMap()

                plt.axis([0,local_map.shape[1],0,local_map.shape[0]])

                path = []
                simp_path = []
                path_age = 0
                path_tries = 0
                h_function = computeHFunction(goal, local_map.shape)

                action = np.array([0, 0])
                if not REACT:
                    env.step(action)

                continue
    
            if not isValidPath(path, local_map) or path_age > MAX_PATH_AGE:
                rospy.loginfo('Finding a Path.....')
                #stoping the robot and getting more reliable sensor readings
                action = np.array([0, 0])
                if not REACT:
                    env.step(action)

                #read when stopped
                if not REACT:
                    state_scan = env.step(action)

                position = (env.position.x, env.position.y)
                position_d  = discretizePoint(position)

                angle = env.yaw

                local_map = mountMap()
                if path_tries > 2:
                    markLaserScan(state_scan, local_map, position, angle, danger_zone_size=REACT_TOLERANCE)
                else:
                    markLaserScan(state_scan, local_map, position, angle)

                path, finish = AStarPathSearch(position, goal, local_map, h_function)
                if REACT == True:
                    continue
                path_tries += 1
                if len(path) == 0 and finish:
                    rospy.logerr('There is no way to go to the point.')
                    path, finish = AStarPathSearch(position, goal, local_map, h_function)
                    continue
                elif len(path) == 0:
                    rospy.logerr('Problem with map, path was not found.')
                    continue
                elif len(path) > 0:
                    rospy.loginfo('Path Found.')
                    path_age = 0
                    path_tries = 0
                    simp_path = superSimplifyPath(path, local_map)
                    if REACT:
                        continue

            for p in path:
                markMapDiscretizedPoint(p, view_map, value = 6)
            for p in simp_path:
                markMapDiscretizedPoint(p, view_map, value = 7)    
            markMapDiscretizedPoint(position_d, view_map, value = 4)
            markMapDiscretizedPoint(goal_d, view_map, value = 5)
            
            action = moveByPath(simp_path, position, angle, action)

            if not REACT:
                state_scan = env.step(action)

            plt.imshow(view_map)
            plt.draw()
            plt.pause(0.0001)

            markMapDiscretizedPoint(position_d, view_map, value = 0)
            markMapDiscretizedPoint(goal_d, view_map, value = 0)
            for p in path:
                markMapDiscretizedPoint(p, view_map, value = 0)
            for p in simp_path:
                markMapDiscretizedPoint(p, view_map, value = 0)
                      
        r.sleep()