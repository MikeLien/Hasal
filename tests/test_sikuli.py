import os
import sys
import time
import lib.sikuli as sikuli
import lib.helper.captureHelper as captureHelper
from lib.perfBaseTest import PerfBaseTest


class TestSikuli(PerfBaseTest):

    def setUp(self):
        super(TestSikuli, self).setUp()
        time.sleep(5)
        captureHelper.capture_screen(self.env, self.env.video_output_sample_1_fp, self.env.img_sample_dp,
                                     self.env.img_output_sample_1_fn)
        self.sikuli = sikuli.Sikuli()

    def test_chrome_load_file(self):
        self.sikuli.run(self.env.sikuli_path, self.env.hasal_dir, self.env.test_method_name, self.env.test_method_name + "_" + self.env.time_stamp)
        assert(True)

    def test_chrome_gdoc_pagedown(self):
        self.sikuli.run(self.env.sikuli_path, self.env.hasal_dir, self.env.test_method_name, self.env.test_method_name + "_" + self.env.time_stamp)
        assert(True)

    def test_firefox_load_file(self):
        self.sikuli.run(self.env.sikuli_path, self.env.hasal_dir, self.env.test_method_name, self.env.test_method_name + "_" + self.env.time_stamp)
        assert(True)

    def test_firefox_gdoc_pagedown(self):
        self.sikuli.run(self.env.sikuli_path, self.env.hasal_dir, self.env.test_method_name, self.env.test_method_name + "_" + self.env.time_stamp)
        assert(True)

    def tearDown(self):
        time.sleep(5)
        captureHelper.capture_screen(self.env, self.env.video_output_sample_2_fp, self.env.img_sample_dp,
                                     self.env.img_output_sample_2_fn)

        if self.env.test_method_name.startswith("test_firefox"):
            self.sikuli.run(self.env.sikuli_path, self.env.hasal_dir, "test_firefox_timing", self.env.test_method_name + "_" + self.env.time_stamp)
        elif self.env.test_method_name.startswith("test_chrome"):
            self.sikuli.run(self.env.sikuli_path, self.env.hasal_dir, "test_chrome_timing",  self.env.test_method_name + "_" + self.env.time_stamp)
        time.sleep(3)
        self.sikuli.run(self.env.sikuli_path, self.env.hasal_dir, "test_firefox_gdoc_clean_all", self.env.test_method_name + "_" + self.env.time_stamp)
        time.sleep(3)

        super(TestSikuli, self).tearDown()
