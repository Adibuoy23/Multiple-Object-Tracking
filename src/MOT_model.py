# Import the necessary libraries
import numpy as np
import sys
import turicreate as tc
from os import walk
import fnmatch
from scipy.optimize import linear_sum_assignment


######################################################
# Define the functions here
######################################################

def eccentricity_dependent_covariance(x, eye, eccentricity_scale):
    """ This function computes the noise in the observations that is dependent on the eye fixations.
    :param x: vector containing true coordinates (x,y) of the object
    :param eye: eye fixation coordinates (eye_x, eye_y)
    :param eccentricity_scale: number of pixels per 1 degree eccentricity
    :return: cov (covariance dependent on spatial noise)
    """
    cov = []
    for i in x:
        ans = 0.08 * (1 + (0.42 * np.linalg.norm(i - eye) / eccentricity_scale))
        cov.append(ans)
    cov = np.array(cov)
    return eccentricity_scale * cov

def velocity_dependent_covariance(vel):
    """
    This function computes the noise in the velocity channel.
    The noise generated is gaussian centered around 0, with sd = a + b*v;
     where a = 0.01; b = 0.05 (Vul, Frank, Tenenbaum, Alvarez 2009)
    :param vel:
    :return: covariance
    """
    cov = []
    for v in vel:
        ans = 0.01 + 0.05 * np.linalg.norm(vel)
        cov.append(ans)
    cov = np.array(cov)
    return cov


def generate_noise(Vision_noise):
    """
    This function takes in the standard deviations that are derived from eccentricity_dependent_covariance function.
    Using these standard deviations, it generates a 2D gaussian noise N(0, vnoise) centered at 0 with the specified s.d.
    :param Vision_noise:
    :return: noise
    """
    noise = np.stack([np.random.normal(np.zeros(2), vnoise) for vnoise in Vision_noise], axis=0)
    return noise


def covariance_matrix(Vision_noise):
    """
    This function computes the multi variate covariance matrix (each dimension
    representing the noise for a given object). In total, there are (num_targets +
    num_distractor) dimensions in the first axis of this N-D array.
    :param Vision_noise: covariance of the vision noise obtained from the above function.
    This is (num_target + num_distractor) dimensional vector with each element being the covariance of
    noise associated with that particular object.
    :return: cov_matrix (multi-variate) covariance matrix of shape (N,2,2) - where N = (num_target +
    num_distract)
    """
    cov_matrix = np.stack([np.eye(2) * v for v in Vision_noise], axis=0)
    return cov_matrix

def compute_cost_matrix(belief_frame, observation_frame):
    """
    This function computes the cost associated with allocating each observation to a belief (whether it
    is a target / non-target). The cost is computed by calculating the distance from each observation to all the
    beliefs.
    :param belief_frame: shape (N,2) - where N is (num_targets + num_distractors). First half of the elements
    in the first dimension are targets, the rest are distractors. Each entry in this array is a coordinate pair (x,y)
    indicating where the algorithm believes the targets are.
    :param observation_frame: (N,2) - where N is (num_targets + num_distractors). Each entry in this array
    is a coordinate pair (x,y) indicating the noisy observations made at a given time instance. These are un-ordered
    pairs (no such correspondences such as first half and second half in the belief frame)
    :return: cost matrix (N,N) with each element containing the distance between observation i and belief j.
    """
    cost = np.zeros([np.shape(belief_frame)[0], np.shape(observation_frame)[0]])
    for i, belief_val in enumerate(belief_frame):
        for j, obs_val in enumerate(observation_frame):
            cost[i][j] = np.linalg.norm(belief_val - obs_val)
    return cost

def compute_accuracy(corr_ind,num_targets):
    """
    This function computes the accuracy based on the given correspondences
    :param corr_ind: list containing the index values of the targets
    :param num_targets: num_targets present in that trial
    :return: accuracy
    """
    actual_targets = list(np.arange(1,num_targets+1))
    accuracy = len(list(set(corr_ind) - set(actual_targets)))/float(num_targets)
    return (1-accuracy)

if __name__ == "__main__":
    # Path variable (where the data is located)
    myPath = sys.argv[2]
    # Append the path variable to existing search path
    sys.path.append(myPath)
    # Get the file information in the directory
    file_list = []
    for root, dirs, files in walk(myPath):
        for filename in files:
            if fnmatch.fnmatch(filename.lower(), "*behavior*"):
                file_list.append(filename)
                # file_list.extend(filenames)
    file_list = [s for s in file_list if "downsampled" not in s]
    print(file_list)
    destpath = sys.argv[3]
    sub_idx = int(sys.argv[1]) # subject index
    # Load the data into an SFrame
    #tc.set_runtime_config('tc_DEFAULT_NUM_PYLAMBDA_WORKERS', 24)
    MOT_eye_data = tc.SFrame.read_json(myPath + file_list[sub_idx], orient='records')
    nruns = 100  # The number of times the algorithm is supposed to be run for each trial
    max_trials = 120
    trials = MOT_eye_data['Trial number'].unique().sort().to_numpy()  # Trials array
    correspondence = np.zeros([max_trials * nruns, 8])  # Initialize array to store correspondences
    true_dist_array = [[]] * max_trials * nruns # Initialize array to store the posterior noise
    correspondence_array = [[]] * max_trials * nruns  # Initialize array to store correspondences
    predicted_accuracy = np.zeros([max_trials*nruns,]) # Initialize array to store the accuracies predicted
    Kp_array = [[]] * max_trials * nruns # Kalman gain array
    trial_array = [[]] * max_trials * nruns # Initialize a list to keep track of all the trial numbers for which the correspondences have been computed
    num_target_array = [[]] * max_trials * nruns # Initialize a list to keep track of the number of targets "" "" ""
    speed_array = [[]] * max_trials * nruns # Initialize a list to keep track of speed "" "" ""
    avg_sampling_rate_per_run=np.zeros([max_trials*nruns,])
    avg_size_per_run = np.zeros([max_trials*nruns,])

    rate=20 #Hz
    sampling_rate = 60/rate  #(60 Hz is the monitor refresh rate, 12Hz is the human sampling rate)
    trial_duration = 10.0
    frame_duration = trial_duration / 600  # (sec)
    # Define the time step
    dt = frame_duration
    # Define the eccentricity (pixel per degree - ppd)
    eccentricity = 33.6

    np.random.seed(0) # set the random seed to 0 for consistency of random numbers

    # Define the state transition matrix A
    A = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]) # [4 X 4] array
    B = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]) #[2 X 4] array

    # Define the process noise covariance
    Process_noise = 1e-3
    # Vision_noise = 1e-4
    V_Noise = []

    for trial in trials:
        trial_data = MOT_eye_data[MOT_eye_data['Trial number'] == trial]
        for name in trial_data.column_names():
            if name not in ['T_Centroid.x','T_Centroid.y']:
                if not trial_data[name].all(): # Remove all empty columns
                    trial_data = trial_data.remove_column(name)
        # get the trajectory data
        # get the columns containing the co-ordinate information of the trajectory.
        word_list = trial_data.column_names()
        obj_columns = [s for s in word_list if "Object" in s]
        num_targets = int(trial_data['Number of Targets'].unique().to_numpy())
        num_distractors = num_targets
        s = float(trial_data['Speed (deg/sec)'].unique().to_numpy())
        trial_data = trial_data[0:len(trial_data):sampling_rate] # Down-sample the data to the given sampling rate
        trajectory_data = trial_data[obj_columns]
        eye_data = trial_data['T_Centroid.x', 'T_Centroid.y'] #fixation.x and fixation.y in Zheng's data

        for run in range(nruns):
            #Get the noisy estimates of the first frame.
            xhat = np.zeros([num_targets+num_distractors, 2, 4])  # a posteri estimate of x
            xhatminus = np.zeros([num_targets+num_distractors, 2, 4])  # a priori estimate of x
            observations = np.zeros([num_targets+num_distractors, int(600/sampling_rate), 2])

            # Initialize the xhat (state) matrix
            initial_x = np.array([trajectory_data[0].get(key) for key in obj_columns])
            initial_eye = np.array(list(eye_data[0].values()))
            initial_x_plus_1 = np.array([trajectory_data[1].get(key) for key in obj_columns])
            initial_vel = initial_x_plus_1 - initial_x
            initial_x = np.reshape(initial_x, [num_targets+num_distractors, 2])
            initial_vel = np.reshape(initial_vel, [num_targets+num_distractors, 2])
            xhat[:, 0, :] = np.concatenate((initial_x, initial_vel), axis=1)
            xhatminus[:,0,:] = xhat[:,0,:]
            ## Generate the noise based on eye fixations
            Vision_noise = eccentricity_dependent_covariance(xhat[:,0,0:2], initial_eye, eccentricity) #Generates the eccentricity based sigma
            noise = generate_noise(Vision_noise)
            observations[:,0,:] = xhat[:,0,0:2] + noise

            # Define the filter parameters
            ##################################
            P = np.zeros([num_targets+num_distractors, 2, 4, 4])  # a posteriori error estimate covariance matrix
            P[:, 0, :, :] = [[0.25 * (dt ** 4), 0, 0.5 * (dt ** 3), 0], [0, 0.25 * (dt ** 4), 0, 0.5 * (dt ** 3)],
                             [0.5 * (dt ** 2), 0, 1 * (dt), 0], [0, 0.5 * (dt ** 2), 0, 1 * (dt)]]
            Pminus = np.zeros([num_targets+num_distractors, 2, 4, 4])  # a priori error estimate covariance matrix
            Pminus[:, 0, :, :] = [[0.25 * (dt ** 4), 0, 0.5 * (dt ** 3), 0], [0, 0.25 * (dt ** 4), 0, 0.5 * (dt ** 3)],
                                  [0.5 * (dt ** 2), 0, 1 * (dt), 0], [0, 0.5 * (dt ** 2), 0, 1 * (dt)]]
            K = np.zeros([num_targets+num_distractors, 2, 4, 2])  # gain or blending factor
            K[:, 0, :, :] = [[1, 0], [0, 1], [0, 0], [0, 0]]
            ##################################

            # Define the stack of identity matrix
            arrays = [np.eye(4) for _ in range(num_targets+num_distractors)]
            identity_matrix_stack = np.stack(arrays, axis=0)

            row_ind = []
            col_ind = []
            true_dist_list = []
            Kp = []
            sampled_frame_list = []
            size_list = []
            current_sampled_frame = 1
            for frame in range(1,len(trajectory_data)):
                # ---> Prediction update
                ix = 1
                xhatminus[:, ix, :] = np.matmul(xhat[:, ix - 1, :], A.T) # verified
                matmulA = np.matmul(np.matmul(A, P[:, ix - 1, :, :]), A.T) # verified
                Pminus[:, ix, :, :] = matmulA + Process_noise * identity_matrix_stack # verified

                #Get the noisy observations from the true data and store it in a temporary variable
                x_temp = np.array([trajectory_data[frame].get(key) for key in obj_columns])
                x_temp = np.reshape(x_temp, [num_targets+num_distractors, 2])
                eye = np.array(list(eye_data[frame].values()))
                #x_temp_minus_1 = np.array([trajectory_data[frame-1].get(key) for key in obj_columns])
                #size=trial_data[frame]['size'] #not present in zheng's data
                ####################################################
                ## Generate the noise based on eye fixations
                Vision_noise = eccentricity_dependent_covariance(x_temp, eye, eccentricity) #Generates the eccentricity based sigma
                V_Noise.append(Vision_noise) #Change V_Noise to Vision_noise_array later
                noise = generate_noise(Vision_noise)
                ####################################################
                ## ---> Measurement update
                eccentricity_dependent_noisy_obs = x_temp + noise
                observations[:,frame,:] = eccentricity_dependent_noisy_obs

                ## ---> Correspondence update
                beta = 0.5 * (K[:, ix - 1, 0, 0] + K[:, ix - 1, 1, 1])
                #belief_frame = np.stack([(1-b) * xhm + b * xm for b,xhm,xm in
                #                         zip(beta, xhatminus[:,ix, 0:2], xhat[:, ix - 1, 0:2])], axis=0)
                belief_frame = xhatminus[:, ix, 0:2] ## Previously used version
                cost = compute_cost_matrix(belief_frame[0:num_targets], observations[:,frame, 0:2])
                row,col = linear_sum_assignment(cost)
                row_ind.append(list(row+1))
                col_ind.append(list(col+1))

                true_dist = list(cost[row,col])
                true_dist_list.append(true_dist)
                # Update the observations based on the correspondences
                non_beliefs = list(set(np.arange(0, num_targets+num_distractors)) - set(col))
                col = list(col)
                col.extend(non_beliefs)
                updated_observations = list(np.take(observations[:,frame,:], col, axis = 0))
                updated_observations_previous = list(np.take(observations[:,frame - 1,:], col, axis = 0))

                # store the size and sampling rate information (could be useful when dealing with variable sampling rate)
                sampled_frame_list.append(sampling_rate)
                #size_list.append(size)

                matmulP = np.matmul(np.matmul(B, Pminus[:,ix, :, :, ]), B.T)
                K[:,ix, :, :] = np.matmul(np.matmul(Pminus[:,ix, :, :], B.T),
                                          np.linalg.inv(matmulP + covariance_matrix(Vision_noise)))
                err_from_ = np.array(updated_observations) - np.matmul(xhatminus[:,ix, :], B.T)
                xhat[:, ix, 0:2] = xhatminus[:, ix, 0:2] + np.matmul(K[:, ix, :], err_from_[:, :, np.newaxis])[:, 0:2, 0]
                xhat[:, ix, 2:] = (np.array(updated_observations) - np.array(updated_observations_previous))
                P[:, ix, :, :] = np.matmul((identity_matrix_stack - np.matmul(K[:, ix, :, :], B)),
                                           Pminus[:, ix, :, :])
                Kp.append([np.linalg.norm(K[i, ix, 0:2, :]) for i in range(num_targets + num_distractors)])

                # For memory conservation purpose only
                xhat[:, ix-1, :] = xhat[:, ix, :]
                P[:, ix-1, :, :] = P[:, ix, :, :]

            correspondence[nruns * (trial - 1) + run][0:num_targets] = col_ind[-1][0:num_targets]
            correspondence_array[nruns * (trial - 1) + run] = col_ind
            true_dist_array[nruns * (trial - 1) + run] = true_dist_list
            Kp_array[nruns * (trial - 1) + run] = Kp
            avg_sampling_rate_per_run[nruns * (trial - 1) + run] = np.mean(sampled_frame_list)
            #avg_size_per_run[nruns * (trial - 1) + run] = np.mean(size_list)
            predicted_accuracy[nruns * (trial-1) + run] = compute_accuracy(col_ind[-1][0:num_targets],num_targets)
            trial_array[nruns * (trial-1) + run] = trial
            num_target_array[nruns * (trial-1) + run] = num_targets
            speed_array[nruns * (trial-1) + run] = s

    sf = tc.SFrame(tc.SArray(correspondence))
    sf['correspondence_full'] = tc.SArray(correspondence_array)
    sf['cost'] = tc.SArray(true_dist_array)
    sf['predicted_accuracy'] = tc.SArray(predicted_accuracy)
    sf['Kalman_gain'] = tc.SArray(Kp_array)
    sf['trial_Num_Original'] = tc.SArray(trial_array)
    sf['num_targets'] = tc.SArray(num_target_array)
    sf['speed'] = tc.SArray(speed_array)
    sf['avg_sampling_rate']=tc.SArray(avg_sampling_rate_per_run)
    #sf['size'] = tc.SArray(avg_size_per_run) # not present in zheng's data
    sf.export_json(destpath + 'correspondences_sub_' + str(rate)+'_Hz_'+str(file_list[sub_idx][4:7]) + '.json',
                   orient='records')
