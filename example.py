"""example.py

Example to use the Dirichlet mixture estimation module.
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
    parser.add_option("-f", "--file", help="Input file with data samples.")
    parser.add_option("-c", "--comp", help="Number of mixture components.",default=1, type="int")
    parser.add_option("-i", "--iter", help="Maximum number of iterations.",default=10,type="int")
    return parser


def main(argv=None):
    if argv is None:
        argv = sys.argv
    
    parser = setup_parser()
    (options, args) = parser.parse_args()

    try:
        if not os.path.exists(options.file):
            raise Exception("File %s does not exist." % options.file) 
        data = np.genfromtxt(options.file)
        (w_hat,a_hat) = dirmix.estimate(data,options.comp,options.iter)
        print "Weights:"
        print w_hat
        print "Alphas:"
        print a_hat

    except Exception as e:
        print "OOps: %s" % e

if __name__ == "__main__":
    sys.exit( main() )
  
