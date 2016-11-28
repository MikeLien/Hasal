
from lib.perfBaseTest import PerfBaseTest


class TestSikuli(PerfBaseTest):

    def setUp(self):
        super(TestSikuli, self).setUp()

    def test_firefox_topsites_www_bing_com_scroll(self):
        self.sikuli_status = self.sikuli.run_test(self.env.test_name, self.env.output_name, test_target="https://www.bing.com/search?q=flowers", script_dp=self.env.test_script_py_dp, args_list=[self.env.img_sample_dp, self.env.img_output_sample_1_fn])
