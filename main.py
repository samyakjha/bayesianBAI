import numpy
from tqdm import tqdm
import numpy
import pandas
import StochasticBanditsModules
import StochasticBanditsPolicies
import matplotlib.pyplot

horizon_vals = [25, 50, 75, 100, 125, 150, 175, 200]
N = 5000
K = 8

no_of_rounds = numpy.ceil(numpy.log(K) / numpy.log(2))

arm_mean_vectorized = numpy.vectorize(StochasticBanditsModules.arm_mean)
arm_posterior_mean_vectorized = numpy.vectorize(StochasticBanditsModules.arm_posterior_mean)
arm_priori_mean_vectorized = numpy.vectorize(StochasticBanditsModules.arm_priori_mean)
arm_variance_square_vectorized = numpy.vectorize(StochasticBanditsModules.arm_variance_square)
arm_posterior_variance_square_vectorized = numpy.vectorize(StochasticBanditsModules.arm_posterior_variance_square)


# Main Body of Code: Simulation Work

probability_array = numpy.array([])
for horizon in horizon_vals:
    sum_val = 0
    for index in tqdm(range(N)):
        experiment = StochasticBanditsModules.Environment()
        # Careful!,  Import only priori arms otherwise priori_mean will be undefined
        for i in range(K):
            new_arm = StochasticBanditsModules.PrioriGaussianArm(experiment, 1 / (2 ** i))

        arms = experiment.listArms()
        arms_list = pandas.DataFrame(index=arms, data=arm_posterior_mean_vectorized(arms), columns=['posterior_mean'])
        print(arms_list)
        optimal_mean = numpy.amax(arm_mean_vectorized(arms))

        # BayesElim2 Code:
        for round_no in range(int(no_of_rounds)):

            sum_of_variance_square = numpy.sum(arm_variance_square_vectorized(arms_list.index))
            for arm in arms_list.index:
                no_of_pulls = (horizon/no_of_rounds*sum_of_variance_square)*arm.variance_square

                for i in range(int(no_of_pulls)):
                    arm.pullArm()

                no_of_samples = len(arm.history)

                arm.posterior_variance_square = 1 / (
                        (1 / arm.priori_variance_square) + (horizon /
                                                            (no_of_rounds * sum_of_variance_square)))

                arm.posterior_mean = arm.posterior_variance_square * \
                                     (arm.priori_mean / arm.priori_variance_square +
                                      numpy.sum(arm.history) / arm.variance_square)

            arms_list = arms_list.sort_values(by='posterior_mean', ascending=False)
            arms_list = arms_list.iloc[0:len(arms_list.index) // 2 + 1]

        if arms_list.index[0].mean == optimal_mean:
            sum_val += 1
    probability_array = numpy.append(probability_array, 1 - (sum_val / N))

# Plotting the results
matplotlib.pyplot.plot(horizon_vals, probability_array)
matplotlib.pyplot.xlabel('Horizon')
matplotlib.pyplot.ylabel('Probability of Misidentification of Optimal Arm')
matplotlib.pyplot.show()
