__author__ = 'shako'
import os
import cv2
import json
import time
import argparse
import shutil
import numpy as np
from argparse import ArgumentDefaultsHelpFormatter
import re

DEFAULT_IMG_DIR_PATH = os.path.join(os.getcwd(), "images")
DEFAULT_SAMPLE_DIR_PATH = os.path.join(os.getcwd(), "sample")
DEFAULT_IMG_LIST_DATA_FN = "data.json"


class ImageTool(object):

    def __init__(self):
        self.image_list = []
        self.current_fps = 0
        self.search_range = [0, 0, 0, 0]

    def dump_result_to_json(self, data, output_fp):
        with open(output_fp, "wb") as fh:
            json.dump(data, fh)

    def convert_video_to_images(self, input_video_fp, output_image_dir_path, output_image_name=None, exec_timestamp_list=[], comp_mode=0):
        vidcap = cv2.VideoCapture(input_video_fp)
        if hasattr(cv2, 'CAP_PROP_FPS'):
            self.current_fps = vidcap.get(cv2.CAP_PROP_FPS)
        else:
            self.current_fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
        result, image = vidcap.read()
        if exec_timestamp_list:
            ref_start_point = exec_timestamp_list[1] - exec_timestamp_list[0]
            ref_end_point = exec_timestamp_list[2] - exec_timestamp_list[0]
            self.search_range = [
                int((ref_start_point - 10) * self.current_fps),
                int((ref_start_point + 10) * self.current_fps),
                int((ref_end_point - 10) * self.current_fps),
                int((ref_end_point + 10) * self.current_fps)]
        if output_image_name:
            if os.path.exists(output_image_dir_path) is False:
                os.mkdir(output_image_dir_path)
            str_image_fp = os.path.join(output_image_dir_path, output_image_name)
            cv2.imwrite(str_image_fp, image)
        else:
            img_cnt = 1
            if os.path.exists(output_image_dir_path):
                shutil.rmtree(output_image_dir_path)
            os.mkdir(output_image_dir_path)
            while result:
                str_image_fp = os.path.join(output_image_dir_path, "image_%d.jpg" % img_cnt)
                if comp_mode and img_cnt >= self.search_range[0]:
                    cv2.imwrite(str_image_fp, image)
                else:
                    if (img_cnt >= self.search_range[0] and img_cnt <= self.search_range[1]) or (img_cnt >= self.search_range[2] and img_cnt <= self.search_range[3]):
                        cv2.imwrite(str_image_fp, image)
                self.image_list.append({"time_seq": vidcap.get(0), "image_fp": str_image_fp})
                result, image = vidcap.read()
                img_cnt += 1
        if self.search_range[0] < 0:
            self.search_range[0] = 0
        if self.search_range[1] > len(self.image_list):
            self.search_range[1] = len(self.image_list)
        if self.search_range[2] < 0:
            self.search_range[2] = 0
        if self.search_range[3] > len(self.image_list):
            self.search_range[3] = len(self.image_list)
        return self.image_list

    def compare_with_sample_image(self, input_sample_dp):
        result_list = []
        print "Comparing sample file start %s" % time.strftime("%c")
        sample_fn_list = os.listdir(input_sample_dp)
        if len(sample_fn_list) != 2:
            return result_list
        sample_fn_list.sort()
        found_1 = False
        found_2 = False
        for sample_fn in sample_fn_list:
            breaking = False
            sample_fp = os.path.join(input_sample_dp, sample_fn)
            sample_dct = self.convert_to_dct(sample_fp)
            for img_index in range(self.search_range[1] - 1, self.search_range[0], -1):
                if found_1: break
                image_data = self.image_list[img_index]
                comparing_dct = self.convert_to_dct(image_data['image_fp'])
                if self.compare_two_images(sample_dct, comparing_dct):
                    print "Comparing sample file end %s" % time.strftime("%c")
                    result_list.append(image_data)
                    breaking = True
                    found_1 = True
                    break
            for img_index in range(self.search_range[2] - 1, self.search_range[3]):
                if breaking: break
                if found_2: break
                image_data = self.image_list[img_index]
                comparing_dct = self.convert_to_dct(image_data['image_fp'])
                if self.compare_two_images(sample_dct, comparing_dct):
                    print "Comparing sample file end %s" % time.strftime("%c")
                    result_list.append(image_data)
                    breaking = True
                    found_2 = True
                    break
        print result_list
        return result_list

    def compare_two_images(self, dct_obj_1, dct_obj_2):
        match = False
        row1, cols1 = dct_obj_1.shape
        row2, cols2 = dct_obj_2.shape
        if (row1 != row2) or (cols1 != cols2):
            return match
        else:
            threshold = 0.0001
            mismatch_rate = np.sum(np.absolute(np.subtract(dct_obj_1,dct_obj_2)))/(row1*cols1)
            if mismatch_rate > threshold:
                return False
            else:
                return True

    def convert_to_dct(self, image_fp):
        img_obj = cv2.imread(image_fp)
        img_gray = cv2.cvtColor(img_obj, cv2.COLOR_BGR2GRAY)
        img_dct = np.float32(img_gray)/255.0
        dct_obj = cv2.dct(img_dct)
        return dct_obj

    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        return [self.atoi(c) for c in re.split('(\d+)', text)]

    def compare_with_sample_object(self, input_sample_dp):
        result_list = []
        m_start_index = 0
        print "Comparing sample file start %s" % time.strftime("%c")
        sample_fn_list = os.listdir(input_sample_dp)
        if len(sample_fn_list) <= 2:
            return result_list
        sample_fn_list.sort(key=self.natural_keys)
        for sample_index in range(0, len(sample_fn_list)):
            sample_fp = os.path.join(input_sample_dp, sample_fn_list[sample_index])
            if sample_index == 1:
                print "Template matching will skip sample 2 comparison"
            elif sample_index == 0:
                sample_dct = self.convert_to_dct(sample_fp)
                for img_index in range(self.search_range[1] - 1, self.search_range[0], -1):
                    image_data = self.image_list[img_index]
                    comparing_dct = self.convert_to_dct(image_data['image_fp'])
                    if self.compare_two_images(sample_dct, comparing_dct):
                        print "Comparing sample file end %s" % time.strftime("%c")
                        result_list.append(image_data)
                        m_start_index = img_index
                        print m_start_index
                        break
            else:
                min_val = 1.0
                result_list.append(self.image_list[m_start_index])
                print m_start_index
                for img_index in range(m_start_index, self.search_range[3]):
                    image_data = self.image_list[img_index]
                    match_val = self.template_match(sample_fp, image_data['image_fp'])
                    print "Comparing sample file end %s" % time.strftime("%c")
                    result_list.append(image_data)
                    if match_val < min_val:
                        min_val = match_val
                        result_list[-1] = image_data
        print result_list
        return result_list

    def template_match(self, input_sample_fp, template_fp):
        img = cv2.imread(input_sample_fp)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        template = cv2.imread(template_fp, 0)
        # template = cv2.imread('BlankPage.jpg',0)
        # template = cv2.imread('MainIcon.jpg',0)
        w, h = template.shape[::-1]

        # All the 6 methods for comparison in a list
        # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
        #            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
        methods = 'cv2.TM_SQDIFF_NORMED'
        method_eval = eval(methods)
        # Apply template Matching
        res = cv2.matchTemplate(img_gray, template, method_eval)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        print np.abs(min_val) * 100
        '''
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method_eval in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)


        cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)
        cv2.imshow('original', img)
        cv2.waitKey(0)
        '''
        return min_val

def main():
    arg_parser = argparse.ArgumentParser(description='Image tool',
                                         formatter_class=ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument('--convertvideo', action='store_true', dest='convert_video_flag', default=False,
                            help='convert video to images.', required=False)
    arg_parser.add_argument('--compareimg', action='store_true', dest='compare_img_flag', default=False,
                            help='compare images.', required=False)
    arg_parser.add_argument('-i', '--input', action='store', dest='input_video_fp', default=None,
                            help='Specify the video file path.', required=False)
    arg_parser.add_argument('-o', '--outputdir', action='store', dest='output_img_dp', default=None,
                            help='Specify output image dir path.', required=False)
    arg_parser.add_argument('-n', '--outputimgname', action='store', dest='output_img_name', default=None,
                            help='Specify output image name.', required=False)
    arg_parser.add_argument('-s', '--sample', action='store', dest='sample_img_dp', default=None,
                            help='Specify sample image dir path.', required=False)
    arg_parser.add_argument('-r', '--resultfp', action='store', dest='result_fp', default=None,
                            help='Specify result file path.', required=False)
    args = arg_parser.parse_args()

    img_tool_obj = ImageTool()
    input_video_fp = args.input_video_fp
    output_img_dp = args.output_img_dp
    output_img_name = args.output_img_name
    sample_img_dp = args.sample_img_dp
    result_fp = args.result_fp

    if args.convert_video_flag is False and args.compare_img_flag is False:
        # default is compare images
        if input_video_fp and output_img_dp and sample_img_dp and result_fp:
            img_tool_obj.convert_video_to_images(input_video_fp, output_img_dp, output_img_name)
            img_tool_obj.dump_result_to_json(img_tool_obj.compare_with_sample_image(sample_img_dp), result_fp)
        else:
            print "Please specify the input video dir path, output image dir path, output image name, sample image dir path and result file path."
    elif args.convert_video_flag:
        # convert video to images
        if input_video_fp and output_img_dp:
            img_tool_obj.convert_video_to_images(input_video_fp, output_img_dp, output_img_name)
        else:
            print "Please specify the input video dir path, output image dir path and output image name."
    else:
        # compare images
        if input_video_fp and output_img_dp and sample_img_dp and result_fp:
            img_tool_obj.convert_video_to_images(input_video_fp, output_img_dp, output_img_name)
            img_tool_obj.dump_result_to_json(img_tool_obj.compare_with_sample_image(sample_img_dp), result_fp)
        else:
            print "Please specify the input video dir path, output image dir path, output image name, sample image dir path and result file path."

if __name__ == '__main__':
    main()