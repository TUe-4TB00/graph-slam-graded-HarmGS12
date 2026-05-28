import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global` which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function `add_landmark_measurement_from_global` which calculates the correct bearing and range from the global poses.``
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    # TODO: Initialize the optimizer 
    params = gtsam.LevenbergMarquardtParams()

    # TODO: Perform the optimization and print the result
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    return result

def _copy_graph(graph):
    try:    
        return gtsam.NonlinearFactorGraph(graph)
    except TypeError:
        copied_graph = gtsam.NonlinearFactorGraph()
        for i in range(graph.size()):
            copied_graph.add(graph.at(i))
        return copied_graph
    
def _copy_values(values):
    try:
        return gtsam.Values(values)
    except TypeError:
        copied_values = gtsam.Values()
        for key in values.keys():
            try:
                copied_values.insert(key,values.atPose2(key))
            except RuntimeError:
                copied_values.insert(key, values.atPoint2(key))
        return copied_values
    
def minimize_marginals(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest sum of marginals.
    best_pose = None      # chosen pose option
    best_landmark = None    # chosen landmark (1 or 2)
    lowest_choice_marginal = float("inf")
    best_returned_marginals = None

    for pose_name, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            test_graph = _copy_graph(graph)
            test_initial = _copy_values(initial_estimate)

            test_graph, test_initial = add_pose(test_graph, test_initial, pose_5)

            result = optimize(test_graph, test_initial)

            test_graph = add_landmark_measurement(
                test_graph,
                result,
                pose_5,
                landmark
            )

            result = optimize(test_graph, test_initial)

    # TODO: Calculate marginal covariances for the relevant variables and visualize the updated factor graph with covariances
            marginals = gtsam.Marginals(test_graph, result)
            choice_marginal = marginals.marginalCovariance(L(landmark)).sum()

    # The sum of the marginals for each landmark can be computed using marginals.marginalCovariance(L(x)).sum()
            total_marginals = (
                marginals.marginalCovariance(L(1)).sum()
                + marginals.marginalCovariance(L(2)).sum()
            )

            if choice_marginal < lowest_choice_marginal:
                lowest_choice_marginal = choice_marginal
                best_pose = pose_name
                best_landmark = landmark
                best_returned_marginals = total_marginals

    return best_pose, best_landmark, best_returned_marginals

def minimize_errors(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest resulting error.
    best_pose = None      # chosen pose option
    best_landmark = None    # chosen landmark (1 or 2)
    lowest_error = float("inf")
    list_of_errors = []

    for pose_name, pose_5 in pose_options.items():
        lowest_error_for_pose = float("inf")
        best_landmark_for_pose = None

        for landmark in [1, 2]:
            test_graph = _copy_graph(graph)
            test_initial = _copy_values(initial_estimate)

            test_graph, test_initial = add_pose(test_graph, test_initial, pose_5)

            result = optimize(test_graph, test_initial)

            test_graph = add_landmark_measurement(
                test_graph,
                result,
                pose_5,
                landmark
            )

            result = optimize(test_graph, test_initial)

            error = test_graph.error(result)

            if error < lowest_error_for_pose:
                lowest_error_for_pose = error
                best_landmark_for_pose = landmark

        list_of_errors.append(lowest_error_for_pose)

        if lowest_error_for_pose < lowest_error:
            lowest_error = lowest_error_for_pose
            best_pose = pose_name
            best_landmark = best_landmark_for_pose

    sum_of_errors = sum(list_of_errors)

    return best_pose, best_landmark, sum_of_errors
    