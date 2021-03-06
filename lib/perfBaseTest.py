import unittest
import helper.desktopHelper as desktopHelper
import helper.resultHelper as resultHelper
import lib.helper.targetHelper as targetHelper
import lib.sikuli as sikuli
from common.environment import Environment
from helper.profilerHelper import Profilers


class PerfBaseTest(unittest.TestCase):

    profiler_list = [{"path": "lib.profiler.avconvProfiler", "name": "AvconvProfiler"}]

    def setUp(self):
        # Init environment variables
        self.env = Environment(self._testMethodName)

        # Init output dirs
        self.env.init_output_dir()

        # Init sikuli status
        self.sikuli_status = 0

        # get browser type
        self.browser_type = self.env.get_browser_type()

        # init target helper
        self.target_helper = targetHelper.TagetHelper(self.env)

        # Start video recordings
        self.profilers = Profilers(self.env)
        self.profilers.start_profiling(self.profiler_list)

        # init sikuli
        self.sikuli = sikuli.Sikuli()

        # minimize all windows
        desktopHelper.minimize_window()

        # launch browser
        desktopHelper.launch_browser(self.browser_type)

    def tearDown(self):

        # Stop profiler and save profile data
        self.profilers.stop_profiling()

        # Stop browser
        desktopHelper.stop_browser(self.browser_type, self.env)

        # Delete Url
        if self.test_url_id:
            self.target_helper.delete_target(self.test_url_id)

        # output result
        if self.sikuli_status == 0:
            resultHelper.result_calculation(self.env)
        else:
            print "[WARNING] This running result of sikuli execution is not successful, return code: " + str(self.sikuli_status)
