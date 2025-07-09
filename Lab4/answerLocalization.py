from typing import List
import numpy as np
from utils import Particle

### 可以在这里写下一些你需要的变量和函数 ###
COLLISION_DISTANCE = 1
MAX_ERROR = 50000
def is_valid(x, y, walls):
    for wall in walls:
        if abs(x - wall[0]) < COLLISION_DISTANCE and abs(y - wall[1]) < COLLISION_DISTANCE:
            return False
    return True
calculate_k = 0.5
gauss_position_noise = 0.11
gauss_theta_noise = np.pi / 30
### 可以在这里写下一些你需要的变量和函数 ###

def generate_uniform_particles(walls, N):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    N: int, 采样点数量
    输出：
    particles: List[Particle], 返回在空地上均匀采样出的N个采样点的列表，每个点的权重都是1/N
    """
    all_particles: List[Particle] = []
    ### 你的代码 ###
    min_x, min_y = walls.min(axis=0)
    max_x, max_y = walls.max(axis=0)
    count = 0
    while count < N:
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        theta = np.random.uniform(0, np.pi * 2)
        if is_valid(x, y, walls):
            particle = Particle(x, y, theta, 1.0 / N)
            all_particles.append(particle)
            count += 1
    ### 你的代码 ###
    return all_particles


def calculate_particle_weight(estimated, gt):
    """
    输入：
    estimated: np.array, 该采样点的距离传感器数据
    gt: np.array, Pacman实际位置的距离传感器数据
    输出：
    weight, float, 该采样点的权重
    """
    weight = 1.0
    ### 你的代码 ###
    weight = np.exp(-calculate_k * np.linalg.norm(estimated - gt))
    ### 你的代码 ###
    return weight


def resample_particles(walls, particles: List[Particle]):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    particles: List[Particle], 上一次采样得到的粒子，注意是按权重从大到小排列的
    输出：
    particles: List[Particle], 返回重采样后的N个采样点的列表
    """
    resampled_particles: List[Particle] = []
    for _ in range(len(particles)):
        resampled_particles.append(Particle(1.0, 1.0, 1.0, 0.0))
    ### 你的代码 ###
    N = len(particles)
    counts = []
    total_count = 0
    for p in particles:
        count = int(N * p.get_weight())
        counts.append(count)
        total_count += count
    resampled_particles = []
    for i, p in enumerate(particles):
        for _ in range(counts[i]):
            x_noisy = np.random.normal(p.position[0], gauss_position_noise)
            y_noisy = np.random.normal(p.position[1], gauss_position_noise)
            theta_noisy = np.random.normal(p.theta, gauss_theta_noise)
            new_particle = Particle(x_noisy, y_noisy, theta_noisy, 1.0/N)
            resampled_particles.append(new_particle)
    remaining = N - len(resampled_particles)
    min_x, min_y = walls.min(axis=0)
    max_x, max_y = walls.max(axis=0)
    if remaining > 0:
        for _ in range(remaining):
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            theta = np.random.uniform(0, 2*np.pi)
            new_particle = Particle(x, y, theta, 1.0/N)
            resampled_particles.append(new_particle)
    ### 你的代码 ###
    return resampled_particles

def apply_state_transition(p: Particle, traveled_distance, dtheta):
    """
    输入：
    p: 采样的粒子
    traveled_distance, dtheta: ground truth的Pacman这一步相对于上一步运动方向改变了dtheta，并移动了traveled_distance的距离
    particle: 按照相同方式进行移动后的粒子
    """
    ### 你的代码 ###
    p.theta += dtheta
    p.theta %= 2 * np.pi
    p.position[0] += traveled_distance * np.cos(p.theta)
    p.position[1] += traveled_distance * np.sin(p.theta)
    ### 你的代码 ###
    return p

def get_estimate_result(particles: List[Particle]):
    """
    输入：
    particles: List[Particle], 全部采样粒子
    输出：
    final_result: Particle, 最终的猜测结果
    """
    final_result = Particle()
    ### 你的代码 ###
    final_result = max(particles, key=lambda p: p.weight)
    ### 你的代码 ###
    return final_result