import StochasticBanditsModules
import numpy


class BayesAlumEnv:
    def __init__(self, priorArmMeans, variance_sq=None,  priorArmVarSq=None, optimalArm=None):
        """

        :type priorArmMeans: ndarray
        """
        self.priorArmMeans = priorArmMeans
        self.priorArmVarSq = priorArmVarSq
        self.K = numpy.array([])

        if len(variance_sq) != len(priorArmMeans):
            raise ValueError("Variance must be the same length as the priorArmMeans")

        if variance_sq is None:
            self.variance_sq = numpy.fill(len(priorArmMeans), 0.5)
        if optimalArm is None:
            self.optimalArm = numpy.where(numpy.isclose(priorArmMeans, max(priorArmMeans)))
        if priorArmVarSq is None:
            self.priorArmVarSq = numpy.fill(len(priorArmMeans), 0.5)
        else:
            self.priorArmVarSq = priorArmVarSq

        self.max_val = -numpy.inf

        for armIndex in range(len(self.K)):
            if self.K[armIndex][0] > self.max_val:
                self.max_val = self.K[armIndex]

    def getOptimalArm(self):
        return self.optimalArm


    def genDist(self):
        for i in range(len(self.priorArmMeans)):
            self.K = numpy.append(self.K,
                                  numpy.array(numpy.random.normal(self.priorArmMeans[i], self.priorArmVarSq[i])))

        max_arm = numpy.where(numpy.isclose(self.K, self.max_val))
        spliced_array = numpy.delete(self.K, max_arm)
        numpy.random.shuffle(spliced_array)
        spliced_arm_list_1, spliced_arm_list_2 = numpy.split(spliced_array, [self.optimalArm-1, len(spliced_array) - self.optimalArm])
        spliced_arm_list_1 = sorted(spliced_arm_list_1, key=lambda x: x[0])
        spliced_arm_list_2 = sorted(spliced_arm_list_2, key=lambda x: x[0])
        unimodality_mean_dist_inter = numpy.append(spliced_arm_list_1, self.max_val)
        unimodality_mean_dist = numpy.append(unimodality_mean_dist_inter, spliced_arm_list_2)
        return unimodality_mean_dist

    def alum(self, mean_dist, budget):

        arm_list = mean_dist
        no_of_phases = numpy.floor(numpy.log(len(self.K / 3)) / numpy.log(3 / 2))

        for i in range(no_of_phases):
            phase = i + 1
            arm_list_size = len(arm_list)
            f_list = numpy.array([arm_list[0], arm_list[numpy.ceil(arm_list_size / 3)],
                                  arm_list[numpy.floor(2 * arm_list_size / 3)], arm_list[arm_list_size - 1]])

            if phase == 1 or phase == 2:
                no_of_pulls = ((2 ** (no_of_phases - 2)) / (3 ** (no_of_phases - 1)) * budget)
            else:
                no_of_pulls = ((2 ** (no_of_phases - phase + 1)) / (3 ** (no_of_phases - phase + 2)) * budget)

            mean_vals = numpy.array([0, 0, 0, 0])
            drawn_vals = numpy.array([])

            for index, arm in f_list:
                for pulls in range(no_of_pulls / 4):
                    sample_val = numpy.random.normal(arm[0], arm[1])
                    drawn_vals[index] = drawn_vals[index] + sample_val

                mean_vals[index] = 4 * drawn_vals[index] / no_of_pulls

            max_index = numpy.argmax(mean_vals)

            if max_index == 0 or max_index == 1:
                arm_list = arm_list[0:numpy.floor(2 * arm_list_size / 3)]
            else:
                arm_list = arm_list[numpy.floor(arm_list_size / 3) - 1:arm_list_size]

        phase = no_of_phases + 1
        f_list = numpy.array(arm_list[0], arm_list[numpy.ceil(len(arm_list) / 3)], arm_list[len(arm_list) - 1])

        drawn_vals = numpy.array([])
        mean_vals = numpy.array([0, 0, 0])

        for index, arm in f_list:
            for pulls in range(budget / 9):
                sample_val = numpy.random.normal(arm[0], arm[1])
                drawn_vals[index] = drawn_vals[index] + sample_val
            mean_vals[index] = 9 * drawn_vals[index] / budget

        f_identified_arm = f_list[numpy.argmax(mean_vals)]

        return f_identified_arm

    def bayesAlum(self, mean_dist, budget):
        arm_list = mean_dist
        arm_list_expand = numpy.array([])
        for index, arm in arm_list:
            arm_list_expand[index] = numpy.array([index, arm, 0,0,0])
        index_list = numpy.arange(0, len(arm_list_expand))

        no_of_phases = numpy.floor(numpy.log(len(self.K / 3)) / numpy.log(3 / 2))

        for i in range(no_of_phases):
            phase = i + 1
            b_list = numpy.array([arm_list_expand[index_list[0]], arm_list_expand[index_list[numpy.ceil(len(index_list) / 3)]], arm_list_expand[index_list[numpy.floor(2 * len(index_list) / 3)]], arm_list_expand[index_list[len(index_list) - 1]]])
            for marker_arms in b_list:
                for pull in range(budget / 4):
                    sample_val = numpy.random.normal(marker_arms[1][0], marker_arms[1][1])
                    marker_arms[2] = marker_arms[2] + sample_val
                    marker_arms[3] = marker_arms[3] + 1
                marker_arms[4] = marker_arms[2] / marker_arms[3]

            max_index = max(b_list, key=lambda x: x[4])

            if max_index == 0 or max_index == 1:
                index_list = index_list[0:numpy.floor(2 * len(index_list) / 3)]
            else:
                index_list = index_list[numpy.floor(len(index_list) / 3) - 1:len(index_list)]

        phase = no_of_phases + 1
        b_list = numpy.array([arm_list_expand[index_list[0]], arm_list_expand[index_list[numpy.ceil(len(index_list) / 3)]], arm_list_expand[index_list[len(index_list) - 1]]])
        for marker_arms in b_list:
            for pull in range(budget / 9):
                sample_val = numpy.random.normal(marker_arms[1][0], marker_arms[1][1])
                marker_arms[2] = marker_arms[2] + sample_val
                marker_arms[3] = marker_arms[3] + 1
            marker_arms[4] = marker_arms[2] / marker_arms[3]

        phase = no_of_phases + 1
        b_list = numpy.array([arm_list_expand[index_list[0]], arm_list_expand[index_list[numpy.ceil(len(index_list) / 3)]], arm_list_expand[index_list[len(index_list) - 1]]])
        for marker_arms in b_list:
            for pull in range(budget / 9):
                sample_val = numpy.random.normal(marker_arms[1][0], marker_arms[1][1])
                marker_arms[2] = marker_arms[2] + sample_val
                marker_arms[3] = marker_arms[3] + 1
            marker_arms[4] = marker_arms[2] / marker_arms[3]

        b_identified_arm = b_list[numpy.argmax(b_list, key=lambda x: x[4])]
        return b_identified_arm












