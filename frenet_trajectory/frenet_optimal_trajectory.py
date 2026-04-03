import numpy as np
import copy
import os
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from QuinticPolynomialsPlanner.quintic_polynomials_planner import QuinticPolynomial
from CubicSpline import cubic_spline_planner

from enum import Enum
from cartesian_frenet_converter import (
    CartesianFrenetConverter,
)


class LateralMovement(str, Enum):
    HIGH_SPEED = "HIGH_SPEED"
    LOW_SPEED = "LOW_SPEED"


class LongitudinalMovement(str, Enum):
    MERGING_AND_STOPPING = "MERGING_AND_STOPPING"
    VELOCITY_KEEPING = "VELOCITY_KEEPING"


class ScenarioConfig:
    def __init__(self, lateral_movement=LateralMovement.HIGH_SPEED, longitudinal_movement=LongitudinalMovement.VELOCITY_KEEPING, **kwargs):
        self.lateral_movement = lateral_movement
        self.longitudinal_movement = longitudinal_movement

        self.max_speed = 50.0 / 3.6
        self.max_accel = 5.0
        self.max_curvature = 1.0
        self.dt = 0.2
        self.max_t = 5.0
        self.min_t = 4.0
        self.n_s_sample = 1

        self.k_j = 0.1
        self.k_t = 0.1
        self.k_s_dot = 1.0
        self.k_d = 1.0
        self.k_s = 1.0
        self.k_lat = 1.0
        self.k_lon = 1.0

        self.sim_loop = 500

        if self.lateral_movement == LateralMovement.LOW_SPEED:
            self.max_road_width = 1.0
            self.d_road_w = 0.2
            self.target_speed = 3.0 / 3.6
            self.d_t_s = 0.5 / 3.6
            self.wx = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
            self.wy = [0.0, 0.0, 1.0, 0.0, -1.0, -2.0]
            self.obstacles = np.array([[3.0, 1.0], [5.0, -0.0], [6.0, 0.5], [8.0, -1.5]])
            self.robot_radius = 0.5

            self.initial_speed = 1.0 / 3.6
            self.initial_accel = 0.0
            self.initial_lat_position = 0.5
            self.initial_lat_speed = 0.0
            self.initial_lat_acceleration = 0.0
            self.initial_course_position = 0.0

            self.animation_area = 5.0

            self.stop_s = 4.0
            self.d_s = 0.3
            self.n_stop_s_sample = 3
        else:
            self.max_road_width = 7.0
            self.d_road_w = 1.0
            self.target_speed = 30.0 / 3.6
            self.d_t_s = 5.0 / 3.6
            self.wx = [0.0, 10.0, 20.5, 35.0, 70.5]
            self.wy = [0.0, -6.0, 5.0, 6.5, 0.0]
            self.obstacles = np.array(
                [[20.0, 10.0], [30.0, 6.0], [30.0, 8.0], [35.0, 8.0], [50.0, 3.0]]
            )
            self.robot_radius = 2.0

            self.initial_speed = 10.0 / 3.6
            self.initial_accel = 0.0
            self.initial_lat_position = 2.0
            self.initial_lat_speed = 0.0
            self.initial_lat_acceleration = 0.0
            self.initial_course_position = 0.0

            self.animation_area = 20.0
            self.stop_s = 25.0
            self.d_s = 2
            self.n_stop_s_sample = 4

        # Override defaults if provided in kwargs
        for k, v in kwargs.items():
            if hasattr(self, k):
                if k == "obstacles" and isinstance(v, list):
                    setattr(self, k, np.array(v))
                else:
                    setattr(self, k, v)


class LateralMovementStrategy:
    def calc_lateral_trajectory(self, config, fp, di, c_d, c_d_d, c_d_dd, Ti):
        raise NotImplementedError("calc_lateral_trajectory not implemented")

    def calc_cartesian_parameters(self, config, fp, csp):
        raise NotImplementedError("calc_cartesian_parameters not implemented")


class HighSpeedLateralMovementStrategy(LateralMovementStrategy):
    def calc_lateral_trajectory(self, config, fp, di, c_d, c_d_d, c_d_dd, Ti):
        tp = copy.deepcopy(fp)
        s0_d = fp.s_d[0]
        s0_dd = fp.s_dd[0]
        lat_qp = QuinticPolynomial(
            c_d, c_d_d * s0_d, c_d_dd * s0_d**2 + c_d_d * s0_dd, di, 0.0, 0.0, Ti
        )

        tp.d = []
        tp.d_d = []
        tp.d_dd = []
        tp.d_ddd = []

        for i in range(len(fp.t)):
            t = fp.t[i]
            s_d = fp.s_d[i]
            s_dd = fp.s_dd[i]

            s_d_inv = 1.0 / (s_d + 1e-6) + 1e-6
            s_d_inv_sq = s_d_inv * s_d_inv

            d = lat_qp.calc_point(t)
            d_d = lat_qp.calc_first_derivative(t)
            d_dd = lat_qp.calc_second_derivative(t)
            d_ddd = lat_qp.calc_third_derivative(t)

            tp.d.append(d)
            tp.d_d.append(d_d * s_d_inv)
            tp.d_dd.append((d_dd - tp.d_d[i] * s_dd) * s_d_inv_sq)
            tp.d_ddd.append(d_ddd)

        return tp

    def calc_cartesian_parameters(self, config, fp, csp):
        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            i_yaw = csp.calc_yaw(fp.s[i])
            i_kappa = csp.calc_curvature(fp.s[i])
            i_dkappa = csp.calc_curvature_rate(fp.s[i])
            s_condition = [fp.s[i], fp.s_d[i], fp.s_dd[i]]
            d_condition = [
                fp.d[i],
                fp.d_d[i],
                fp.d_dd[i],
            ]
            x, y, theta, kappa, v, a = CartesianFrenetConverter.frenet_to_cartesian(
                fp.s[i], ix, iy, i_yaw, i_kappa, i_dkappa, s_condition, d_condition
            )
            fp.x.append(x)
            fp.y.append(y)
            fp.yaw.append(theta)
            fp.c.append(kappa)
            fp.v.append(v)
            fp.a.append(a)
        return fp


class LowSpeedLateralMovementStrategy(LateralMovementStrategy):
    def calc_lateral_trajectory(self, config, fp, di, c_d, c_d_d, c_d_dd, Ti):
        s0 = fp.s[0]
        s1 = fp.s[-1]
        tp = copy.deepcopy(fp)
        lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, s1 - s0)

        tp.d = [lat_qp.calc_point(s - s0) for s in fp.s]
        tp.d_d = [lat_qp.calc_first_derivative(s - s0) for s in fp.s]
        tp.d_dd = [lat_qp.calc_second_derivative(s - s0) for s in fp.s]
        tp.d_ddd = [lat_qp.calc_third_derivative(s - s0) for s in fp.s]
        return tp

    def calc_cartesian_parameters(self, config, fp, csp):
        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            i_yaw = csp.calc_yaw(fp.s[i])
            i_kappa = csp.calc_curvature(fp.s[i])
            i_dkappa = csp.calc_curvature_rate(fp.s[i])
            s_condition = [fp.s[i], fp.s_d[i], fp.s_dd[i]]
            d_condition = [fp.d[i], fp.d_d[i], fp.d_dd[i]]
            x, y, theta, kappa, v, a = CartesianFrenetConverter.frenet_to_cartesian(
                fp.s[i], ix, iy, i_yaw, i_kappa, i_dkappa, s_condition, d_condition
            )
            fp.x.append(x)
            fp.y.append(y)
            fp.yaw.append(theta)
            fp.c.append(kappa)
            fp.v.append(v)
            fp.a.append(a)
        return fp


class LongitudinalMovementStrategy:
    def calc_longitudinal_trajectory(self, config, c_speed, c_accel, Ti, s0):
        raise NotImplementedError("calc_longitudinal_trajectory not implemented")

    def get_d_arrange(self, config, s0):
        raise NotImplementedError("get_d_arrange not implemented")

    def calc_destination_cost(self, config, fp):
        raise NotImplementedError("calc_destination_cost not implemented")


class VelocityKeepingLongitudinalMovementStrategy(LongitudinalMovementStrategy):
    def calc_longitudinal_trajectory(self, config, c_speed, c_accel, Ti, s0):
        fplist = []
        for tv in np.arange(
            config.target_speed - config.d_t_s * config.n_s_sample, config.target_speed + config.d_t_s * config.n_s_sample, config.d_t_s
        ):
            fp = FrenetPath()
            lon_qp = QuarticPolynomial(s0, c_speed, c_accel, tv, 0.0, Ti)
            fp.t = [t for t in np.arange(0.0, Ti, config.dt)]
            fp.s = [lon_qp.calc_point(t) for t in fp.t]
            fp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
            fp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
            fp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]
            fplist.append(fp)
        return fplist

    def get_d_arrange(self, config, s0):
        return np.arange(-config.max_road_width, config.max_road_width, config.d_road_w)

    def calc_destination_cost(self, config, fp):
        ds = (config.target_speed - fp.s_d[-1]) ** 2
        return config.k_s_dot * ds


class MergingAndStoppingLongitudinalMovementStrategy(LongitudinalMovementStrategy):
    def calc_longitudinal_trajectory(self, config, c_speed, c_accel, Ti, s0):
        if s0 >= config.stop_s:
            return []
        fplist = []
        for s in np.arange(
            config.stop_s - config.d_s * config.n_stop_s_sample, config.stop_s + config.d_s * config.n_stop_s_sample, config.d_s
        ):
            fp = FrenetPath()
            lon_qp = QuinticPolynomial(s0, c_speed, c_accel, s, 0.0, 0.0, Ti)
            fp.t = [t for t in np.arange(0.0, Ti, config.dt)]
            fp.s = [lon_qp.calc_point(t) for t in fp.t]
            fp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
            fp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
            fp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]
            fplist.append(fp)
        return fplist

    def get_d_arrange(self, config, s0):
        if s0 < config.stop_s / 3:
            return np.arange(-config.max_road_width, config.max_road_width, config.d_road_w)
        else:
            return [0.0]

    def calc_destination_cost(self, config, fp):
        ds = (config.stop_s - fp.s[-1]) ** 2
        return config.k_s * ds


class QuarticPolynomial:
    def __init__(self, xs, vxs, axs, vxe, axe, time):
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time**2, 4 * time**3], [6 * time, 12 * time**2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time, axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        return self.a0 + self.a1 * t + self.a2 * t**2 + self.a3 * t**3 + self.a4 * t**4

    def calc_first_derivative(self, t):
        return self.a1 + 2 * self.a2 * t + 3 * self.a3 * t**2 + 4 * self.a4 * t**3

    def calc_second_derivative(self, t):
        return 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2

    def calc_third_derivative(self, t):
        return 6 * self.a3 + 24 * self.a4 * t


class FrenetPath:
    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.a = []
        self.ds = []
        self.c = []

    def pop_front(self):
        self.x.pop(0)
        self.y.pop(0)
        self.yaw.pop(0)
        self.v.pop(0)
        self.a.pop(0)
        self.s.pop(0)
        self.s_d.pop(0)
        self.s_dd.pop(0)
        self.s_ddd.pop(0)
        self.d.pop(0)
        self.d_d.pop(0)
        self.d_dd.pop(0)
        self.d_ddd.pop(0)


def calc_frenet_paths(config, c_s_d, c_s_dd, c_d, c_d_d, c_d_dd, s0, lat_strat, lon_strat):
    frenet_paths = []

    for Ti in np.arange(config.min_t, config.max_t, config.dt):
        lon_paths = lon_strat.calc_longitudinal_trajectory(
            config, c_s_d, c_s_dd, Ti, s0
        )

        for fp in lon_paths:
            for di in lon_strat.get_d_arrange(config, s0):
                tp = lat_strat.calc_lateral_trajectory(
                    config, fp, di, c_d, c_d_d, c_d_dd, Ti
                )

                Jp = sum(np.power(tp.d_ddd, 2))
                Js = sum(np.power(tp.s_ddd, 2))

                lat_cost = config.k_j * Jp + config.k_t * Ti + config.k_d * tp.d[-1] ** 2
                lon_cost = (
                    config.k_j * Js
                    + config.k_t * Ti
                    + lon_strat.calc_destination_cost(config, tp)
                )
                tp.cf = config.k_lat * lat_cost + config.k_lon * lon_cost
                frenet_paths.append(tp)

    return frenet_paths


def calc_global_paths(config, fplist, csp, lat_strat):
    return [
        lat_strat.calc_cartesian_parameters(config, fp, csp) for fp in fplist
    ]


def check_collision(config, fp, ob):
    for i in range(len(ob[:, 0])):
        d = [
            ((ix - ob[i, 0]) ** 2 + (iy - ob[i, 1]) ** 2)
            for (ix, iy) in zip(fp.x, fp.y)
        ]

        collision = any([di <= config.robot_radius**2 for di in d])

        if collision:
            return False

    return True


def check_paths(config, fplist, ob):
    path_dict = {
        "max_speed_error": [],
        "max_accel_error": [],
        "max_curvature_error": [],
        "collision_error": [],
        "ok": [],
    }
    for i, _ in enumerate(fplist):
        if any([v > config.max_speed for v in fplist[i].v]):
            path_dict["max_speed_error"].append(fplist[i])
        elif any([abs(a) > config.max_accel for a in fplist[i].a]):
            path_dict["max_accel_error"].append(fplist[i])
        elif any([abs(c) > config.max_curvature for c in fplist[i].c]):
            path_dict["max_curvature_error"].append(fplist[i])
        elif not check_collision(config, fplist[i], ob):
            path_dict["collision_error"].append(fplist[i])
        else:
            path_dict["ok"].append(fplist[i])
    return path_dict


def frenet_optimal_planning(config, csp, s0, c_s_d, c_s_dd, c_d, c_d_d, c_d_dd, ob, lat_strat, lon_strat):
    fplist = calc_frenet_paths(config, c_s_d, c_s_dd, c_d, c_d_d, c_d_dd, s0, lat_strat, lon_strat)
    fplist = calc_global_paths(config, fplist, csp, lat_strat)
    fpdict = check_paths(config, fplist, ob)

    min_cost = float("inf")
    best_path = None
    for fp in fpdict["ok"]:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp

    return [best_path, fpdict]


def generate_target_course(x, y):
    csp = cubic_spline_planner.CubicSpline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, csp


def iter_simulation_events(config: ScenarioConfig):
    if config.lateral_movement == LateralMovement.HIGH_SPEED:
        lat_strat = HighSpeedLateralMovementStrategy()
    else:
        lat_strat = LowSpeedLateralMovementStrategy()

    if config.longitudinal_movement == LongitudinalMovement.VELOCITY_KEEPING:
        lon_strat = VelocityKeepingLongitudinalMovementStrategy()
    else:
        lon_strat = MergingAndStoppingLongitudinalMovementStrategy()

    tx, ty, tyaw, tc, csp = generate_target_course(config.wx, config.wy)

    yield {
        "type": "meta",
        "target_course_x": tx,
        "target_course_y": ty,
        "obstacles": config.obstacles.tolist(),
        "animation_area": config.animation_area,
        "robot_radius": config.robot_radius,
    }

    c_s_d = config.initial_speed
    c_s_dd = config.initial_accel
    c_d = config.initial_lat_position
    c_d_d = config.initial_lat_speed
    c_d_dd = config.initial_lat_acceleration
    s0 = config.initial_course_position

    last_path = None
    step_count = 0

    for i in range(config.sim_loop):
        [path, fpdict] = frenet_optimal_planning(
            config, csp, s0, c_s_d, c_s_dd, c_d, c_d_d, c_d_dd, config.obstacles, lat_strat, lon_strat
        )

        if path is None:
            if last_path is not None:
                path = copy.deepcopy(last_path)
                path.pop_front()
            else:
                break
        if len(path.x) <= 1:
            break

        last_path = path
        s0 = path.s[1]
        c_d = path.d[1]
        c_d_d = path.d_d[1]
        c_d_dd = path.d_dd[1]
        c_s_d = path.s_d[1]
        c_s_dd = path.s_dd[1]

        step_count += 1
        yield {
            "type": "frame",
            "step": step_count,
            "path_x": list(path.x[1:]),
            "path_y": list(path.y[1:]),
            "vehicle_x": path.x[1],
            "vehicle_y": path.y[1],
            "vehicle_v": path.v[1],
            "vehicle_yaw": path.yaw[1]
        }

        if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 1.0:
            break

    yield {
        "type": "done",
        "steps": step_count,
    }


def run_simulation(config: ScenarioConfig):
    result = {
        "target_course_x": [],
        "target_course_y": [],
        "obstacles": [],
        "trace": [],
    }

    for event in iter_simulation_events(config):
        if event["type"] == "meta":
            result["target_course_x"] = event["target_course_x"]
            result["target_course_y"] = event["target_course_y"]
            result["obstacles"] = event["obstacles"]
        elif event["type"] == "frame":
            result["trace"].append({
                "path_x": event["path_x"],
                "path_y": event["path_y"],
                "vehicle_x": event["vehicle_x"],
                "vehicle_y": event["vehicle_y"],
                "vehicle_v": event["vehicle_v"],
                "vehicle_yaw": event["vehicle_yaw"],
            })

    return {
        "target_course_x": result["target_course_x"],
        "target_course_y": result["target_course_y"],
        "obstacles": result["obstacles"],
        "trace": result["trace"],
    }
