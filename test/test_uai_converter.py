import unittest
from mrftools import *
import re
import os

class TestUAIConverter(unittest.TestCase):

    def test_default(self):
        file = "test/default.uai"
        conv = UAIConverter(file, is_cuda=False)
        mn = conv.convert()
        bp = TorchMatrixBeliefPropagator(markov_net=mn, is_cuda=False)
        bp.infer(display='full')
        bp.load_beliefs()



    # STILL WORKING ON THESE, DON'T TEST THESE YET, DOESN'T GET ANSWER YET
    def test_grid(self):
        file = "test/grid10x10.f10.uai"
        conv = UAIConverter(file, is_cuda=False)
        mn = conv.convert()
        bp = TorchMatrixBeliefPropagator(markov_net=mn, is_cuda=False)
        bp.infer(display='full')
        bp.load_beliefs()
        # pr_file = open(file + ".PR", "r")

    def test_Grids(self):
        files = [f for f in os.listdir('test/Grids/prob')]
        for file in files:
            print file
            conv = UAIConverter('test/Grids/prob/' + file, is_cuda=False)
            mn = conv.convert()
            bp = TorchMatrixBeliefPropagator(markov_net=mn, is_cuda=False)
            bp.infer(display='full')
            bp.load_beliefs()