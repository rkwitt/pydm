"""dmfit.py

Example of how to use pydm to fit a Dirichlet mixture model
to a collection of probability vecotors.
"""


# Author: Roland Kwitt, Kitware Inc., 2013
# E-Mail: roland.kwitt@kitware.com


import os
import sys
import numpy as np
from optparse import OptionParser

import core.dirichlet as dirichlet
import core.dirmix as dirmix


def setup_parser():
    """Setup the CLI parsing.

    Returns
    -------
    parser : OptionParser object.
    """

    parser = OptionParser()
    parser.add_option("-f", 
                      "--dataFile", 
                      help="Input file with data samples.")
    parser.add_option("-c", 
                      "--components", 
                      help="Number of mixture components.",
                      default=1, 
                      type="int")
    parser.add_option("-s",
                      "--stepSize",
                      type="float",
                      help="Step size (<1) of NR update."),
    parser.add_option("-i", 
                      "--iterations", 
                      help="Maximum number of iterations.",
                      default=10,
                      type="int")
    return parser


def main(argv=None):
    if argv is None:
        argv = sys.argv
    
    parser = setup_parser()
    (options, args) = parser.parse_args()

    try:
        if options.dataFile is None:
            raise Exception("Data file not given.")
        if not os.path.exists(options.dataFile):
            raise Exception("File %s does not exist." % options.dataFile) 
    
        data = np.genfromtxt(options.dataFile)
        (w_hat,a_hat) = dirmix.estimate(data,
                                        options.components,
                                        options.iterations,
                                        options.stepSize)
        print "w_hat"
        print "-----"
        print w_hat
        print "a_hat"
        print "-----"
        print a_hat

    except Exception as e:
        print "OOps: %s" % e


if __name__ == "__main__":
    sys.exit( main() )
  
