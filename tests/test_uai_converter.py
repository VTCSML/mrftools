import unittest
from mrftools import *
import re
import os
import matplotlib.pyplot as plt

# class TestUAIConverter(unittest.TestCase):
#
#     def test_default(self):
#         file = "test/default.uai"
#         conv = UAIConverter(file, is_cuda=False)
#         mn = conv.convert()
#         bp = TorchMatrixBeliefPropagator(markov_net=mn, is_cuda=False)
#         bp.infer(display='full')
#         bp.load_beliefs()

    # def test_generate_graphs(self):
    #     return
	#
    #     length_vals = [[64, 256, 1024, 4096, 16384, 65536, 262144],
    #                    [64, 256, 1024, 4096, 16384, 65536],
    #                    [64, 256, 1024, 4096, 16384],
    #                    [64, 256, 1024, 4096]]
    #     k_vals = [8, 16, 32, 64]
	#
    #     open_gm_vals_780M = [[0.020272, 0.083722, 0.360800, 1.531692, 6.132734, 24.843525, 103.600243],
    #                          [0.096507, 0.358772, 1.150753, 5.384442, 21.453021, 84.609197],
    #                          [0.199723, 0.877941, 3.727052, 15.109935, 61.515682],
    #                          [0.740198, 3.198477, 13.549847, 53.679988]]
    #     torch_gpu_vals_780M = [[0.011237, 0.015032, 0.023691, 0.058706, 0.217165, 0.855206, 3.667111],
    #                            [0.015329, 0.016870, 0.040840, 0.135217, 0.561791, 2.277056],
    #                            [0.014366, 0.029045, 0.105524, 0.396690, 1.745220],
    #                            [0.022770, 0.079424, 0.342601, 1.383379]]
    #     torch_cpu_vals_780M = [[0.012888, 0.058068, 0.205534, 0.902016, 3.817908, 19.713295, 72.331408],
    #                            [0.031925, 0.135298, 0.601358, 2.322249, 10.471162, 43.121254],
    #                            [0.086149, 0.390077, 1.684750, 7.473420, 32.201635],
    #                            [0.276710, 1.272262, 6.001980, 24.984524]]
    #     sparse_cpu_vals_780M = [[0.008742, 0.037125, 0.122105, 0.525256, 2.282667, 13.101705, 51.483916],
    #                             [0.020133, 0.088272, 0.411977, 1.562178, 7.811223, 34.089050],
    #                             [0.060483, 0.285896, 1.276569, 5.603009, 28.427104],
    #                             [0.211892, 1.001056, 4.663128, 22.453724]]
    #     loopy_cpu_vals_780M = [[0.098578, 0.474721, 2.022244, 8.698764, 37.969736, 179.301628, 699.274341],
    #                            [0.460826, 0.486754, 2.137228, 9.173213, 39.972905, 162.973760],
    #                            [0.137518, 0.631804, 2.876917, 11.846864, 51.934883],
    #                            [0.283248, 1.320565, 5.978763, 24.563946]]
	#
    #     torch_gpu_vals_1080 = [[0.006733, 0.007871, 0.010978, 0.022142, 0.071725, 0.287736, 1.209909],
    #                            [0.007291, 0.008348, 0.015454, 0.048541, 0.193675, 0.764596],
    #                            [0.006640, 0.011980, 0.037853, 0.138474, 0.593276],
    #                            [0.009673, 0.028704, 0.119439, 0.480456]]
	#
    #     torch_gpu_vals_1080Ti = [[0.006051, 0.007495, 0.010052, 0.018141, 0.052596, 0.201496, 0.831604],
    #                              [0.006916, 0.007572, 0.012684, 0.035976, 0.137003, 0.524517],
    #                              [0.006339, 0.010025, 0.027862, 0.100321, 0.405351],
    #                              [0.008083, 0.021513, 0.085563, 0.335110]]
	#
    #     torch_gpu_vals_teslap40 = [[0.014558, 0.012896, 0.015374, 0.022354, 0.060849, 0.245069, 1.009314],
    #                                [0.011239, 0.011868, 0.016287, 0.043109, 0.156361, 0.646450],
    #                                [0.011685, 0.013591, 0.034786, 0.117012, 0.479337],
    #                                [0.010656, 0.025574, 0.101743, 0.388470]]
	#
    #     sparse_best_1080Ti = [[0.005374, 0.017588, 0.071773, 0.316002, 1.369214, 6.691314, 29.866451],
    #                           [0.013084, 0.054238, 0.240177, 1.015032, 5.091448, 22.067404],
    #                           [0.038213, 0.182602, 0.860372, 3.587058, 18.403050],
    #                           [0.135710, 0.680351, 3.189963, 14.196262]]
	#
    #     for x in range(len(length_vals)):
    #         plt.loglog(length_vals[x], loopy_cpu_vals_780M[x], '.-', label="Loopy-CPU")
    #         plt.loglog(length_vals[x], open_gm_vals_780M[x], '.-', label="OpenGM-CPU")
    #         plt.loglog(length_vals[x], torch_cpu_vals_780M[x], '.-', label="PyTorch-CPU")
    #         plt.loglog(length_vals[x], sparse_cpu_vals_780M[x], '.-', label="Sparse-CPU")
    #         plt.loglog(length_vals[x], torch_gpu_vals_780M[x], '.--', label="PyTorch-GPU", color="gray")
    #         plt.grid(True)
    #         plt.xlabel('# of variables in grid')
    #         plt.ylabel('time for inference (seconds)')
    #         title_string = "Belief Propagation Running Times (c = %d)" % k_vals[x]
    #         plt.title(title_string)
    #         plt.legend()
    #         plt.show()
	#
    #     for x in range(len(length_vals)):
    #         plt.loglog(length_vals[x], sparse_best_1080Ti[x], '.--', label="Sparse-CPU", color="gray")
    #         plt.loglog(length_vals[x], torch_gpu_vals_780M[x], '.-', label="GTX 780M")
    #         plt.loglog(length_vals[x], torch_gpu_vals_1080[x], '.-', label="GTX 1080")
    #         plt.loglog(length_vals[x], torch_gpu_vals_teslap40[x], '.-', label="Tesla P40")
    #         plt.loglog(length_vals[x], torch_gpu_vals_1080Ti[x], '.-', label="GTX 1080Ti")
    #         plt.grid(True)
    #         plt.xlabel('# of variables in grid')
    #         plt.ylabel('time for inference (seconds)')
    #         title_string = "Belief Propagation Running Times (c = %d)" % k_vals[x]
    #         plt.title(title_string)
    #         plt.legend()
    #         plt.show()


	#
    # # STILL WORKING ON THESE, DON'T TEST THESE YET, DOESN'T GET ANSWER YET
    # def test_grid(self):
    #     file = "test/grid10x10.f10.uai"
    #     conv = UAIConverter(file, is_cuda=False)
    #     mn = conv.convert()
    #     bp = TorchMatrixBeliefPropagator(markov_net=mn, is_cuda=False)
    #     bp.infer(display='full')
    #     bp.load_beliefs()
    #     # pr_file = open(file + ".PR", "r")
	#
    # def test_Grids(self):
    #     files = [f for f in os.listdir('test/Grids/prob')]
    #     for file in files:
    #         print(file)
    #         conv = UAIConverter('test/Grids/prob/' + file, is_cuda=False)
    #         mn = conv.convert()
    #         bp = TorchMatrixBeliefPropagator(markov_net=mn, is_cuda=False)
    #         bp.infer(display='full')
    #         bp.load_beliefs()
	#
    # def test_Segmentation(self):
    #     files = [f for f in os.listdir('test/Segmentation/prob')]
    #     for file in files:
    #         print(file)
    #         conv = UAIConverter('test/Segmentation/prob/' + file, is_cuda=False)
    #         mn = conv.convert()
    #         bp = TorchMatrixBeliefPropagator(markov_net=mn, is_cuda=False)
    #         bp.infer(display='full')
    #         bp.load_beliefs()