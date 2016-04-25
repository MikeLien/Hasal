import sys
import json
import numpy as np

if __name__ == '__main__':
    result_name = sys.argv[1]
    with open(result_name) as fh:
        result = json.load(fh)
    for test_name in result.iterkeys():
        print "Case name: " + test_name
        time_list = result[test_name]['time_list']
        print "Time list: " + str(time_list)
        print "Std_dev: %f" % np.std(time_list)
        print "Median time: %f" % result[test_name]['med_time']
        print "Mean time: %f" % result[test_name]['avg_time']

        print '\n'