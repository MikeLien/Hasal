import os
import cv2
import time
import json
import argparse
import peakutils
import numpy as np
from imageTool import ImageTool
from commonUtil import CommonUtil
from argparse import ArgumentDefaultsHelpFormatter
from logConfig import get_logger
from environment import Environment


logger = get_logger(__name__)


def get_frame_difference(input_image_dir_path):
    """
    Calculate the average dct differences between each two frames from image folder
    @param input_image_dir_path: input image folder path
    @return (difference_norm, img_list): normalized dct difference array, list of image file path
    """
    difference = []
    img_list = os.listdir(input_image_dir_path)
    img_list.sort(key=CommonUtil.natural_keys)
    img_list = [os.path.join(input_image_dir_path, item) for item in img_list if
                os.path.splitext(item)[1] in Environment.IMG_FILE_EXTENSION]
    img_list_dct = [ImageTool().convert_to_dct(item) for item in img_list]
    for img_index in range(1, len(img_list)):
        pre_dct = img_list_dct[img_index - 1]
        cur_dct = img_list_dct[img_index]
        height, width = pre_dct.shape
        mismatch_rate = np.sum(np.absolute(np.subtract(pre_dct, cur_dct))) / (height * width)
        difference.append(mismatch_rate)
    difference_norm = [item / max(difference) for item in difference]
    return difference_norm, img_list


def moving_average(data, n_period, type='simple'):
    """
    Calculate an n period moving average through convolution
    @param data: input array for calculation
    @param n_period: number of period
    @param type: 'simple' type or 'exponential' type
    @return: convolution result
    """
    data = np.asarray(data)
    if type == 'simple':
        weights = np.ones(n_period)
    else:
        weights = np.exp(np.linspace(-1., 0., n_period))
    weights /= weights.sum()
    convolution = np.convolve(data, weights, mode='full')[:len(data)]
    convolution[:n_period] = convolution[n_period]
    return convolution


def get_similarity_score(first_data, second_data):
    """
    Calculate a similarity through correlation coefficient
    @param first_data: first input data sequence
    @param second_data: second input data sequence
    @return: similarity score between two data sequence
    """
    if len(first_data) < len(second_data):
        similarity = np.corrcoef(np.append(first_data, [0] * (len(second_data) - len(first_data))), second_data)[0][1]
    elif len(first_data) > len(second_data):
        similarity = np.corrcoef(first_data, np.append(second_data, [0] * (len(first_data) - len(second_data))))[0][1]
    else:
        similarity = np.corrcoef(first_data, second_data)[0][1]
    logger.debug('Similarity: %d' % similarity)
    return similarity


def quantizer(data, level=16):
    """
    Transform data to a quantized level
    @param data: input data sequence to transform
    @param level: quantization interval
    @return: quantized data sequence
    """
    step = (max(data) - min(data)) / level
    data_norm = [round(float(val) / step) * step for val in data]
    for i in range(len(data_norm)):
        if data_norm[i] < 0.5:
            data_norm[i] = 0
    return data_norm


def dynamic_time_warping(first_data, second_data):
    """
    Compute optimal alignment between two time series sequences
    @param first_data: first input data sequence
    @param second_data: second input data sequence
    @return (dtw, distance): matrix which contains accumulated distances between two input sequences, pure distance matrix between two input sequences
    """
    distance = np.zeros((len(first_data), len(second_data)))
    dtw = np.zeros((len(first_data), len(second_data)))
    dtw[0][0] = np.abs(first_data[0] - second_data[0])
    for i in range(len(first_data)):
        for j in range(len(second_data)):
            cost = np.abs(first_data[i] - second_data[j])
            distance[i][j] = cost
            min_val = 0
            if i and j:
                min_val = min(dtw[i - 1][j], dtw[i][j - 1], dtw[i - 1][j - 1])
            elif not i and j:
                min_val = dtw[i][j - 1]
            elif not j and i:
                min_val = dtw[i - 1][j]
            dtw[i][j] = cost + min_val
    return dtw, distance


def get_warp_path(distance):
    """
    Calculate warp path with minimum cost from a distance matrix
    @param distance: input distance matrix
    @return: warp path which contains pair of coordinates from input matrix
    """
    i = distance.shape[0] - 1
    j = distance.shape[1] - 1
    path = [[i, j]]
    while i > 0 and j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            min_val = min(distance[i - 1][j - 1], distance[i - 1][j], distance[i][j - 1])
            if distance[i - 1][j - 1] == min_val:
                i -= 1
                j -= 1
            elif distance[i - 1][j] == min_val:
                i -= 1
            elif distance[i][j - 1] == min_val:
                j -= 1
        path.append([i, j])
    path.append([0, 0])
    return path


def get_path_cost(path, distance):
    """
    Calculate the cost of path from a distance matrix
    @param path: input warp path based on the result of dynamic time warping
    @param distance: input matrix from the result of dynamic time warping
    @return: the cost  value of warp path
    """
    cost = 0
    for [map_s, map_f] in path:
        cost = cost + distance[map_s, map_f]
    return cost


def find_peaks(data):
    """
    Find index of peaks from data sequence
    @param data:  a data sequence to calculate
    @return: index list of peaks
    """
    threshold = 0.3
    min_dist = 2
    indexes = peakutils.indexes(np.asarray(data), threshold, min_dist)
    return indexes


def target_peaks(path, first_indexes, second_indexes):
    """
    Find target peaks and matching based on two peaks location list
    @param path: path from Dynamic Time Warping
    @param first_indexes: index list of peaks from first data sequence
    @param second_indexes: index list of peaks from second data sequence
    @return: merged target index list of peaks
    """
    distance_threshold = 10
    first_peaks = []
    for i in range(len(path)):
        for j in range(len(first_indexes)):
            if (abs((path[i][0] - path[i][1])) > distance_threshold):
                first_peaks.append(path[i])

    second_peaks = []
    for i in range(len(path)):
        for j in range(len(second_indexes)):
            if (abs((path[i][0] - path[i][1])) > distance_threshold):
                second_peaks.append(path[i])

    merge_peak = []
    num_peaks = max(len(first_peaks), len(second_peaks))
    for i in range(num_peaks):
        if num_peaks == len(second_peaks):
            merge_peak.append(second_peaks[i])
            if i < len(first_peaks) and not first_peaks[i] in second_peaks:
                merge_peak.append(first_peaks[i])
        else:
            merge_peak.append(first_peaks[i])
            if i < len(second_peaks) and not second_peaks[i] in first_peaks:
                merge_peak.append(second_peaks[i])
    merge_peak.sort()
    return merge_peak


def get_distance(peaks):
    """
    Get distances from each two peaks
    @param peaks: index list of peaks
    @return: distance list of each pair of peaks
    """
    return [abs((val[0] - val[1])) for val in peaks]


def get_average_latency(distance):
    """
    Get the average distance from a distance list
    @param distance: distance list
    @return: average value of distance
    """
    return 0.0 if not distance else np.mean(np.array(distance))


def clustering(peaks):
    """
    Clustering matching distance of time between two data sequences
    @param peaks: index list of peaks
    @return: cluster list which contains peak list in each cluster
    """
    sequence = []
    cluster_list = []
    if peaks:
        distance_list = get_distance(peaks)
        logger.debug('distance list: %s' % distance_list)
        step = 200
        logger.debug('quantization step of cluster: %d' % step)
        distance_list_norm = [round(float(val) / step) * step for val in distance_list]
        logger.debug('quantized distance list: %s' % distance_list_norm)
        mark_distance = [0] * len(distance_list_norm)
        cur_label = 1
        cur_level = distance_list_norm[0]
        sequence.append(peaks[0])
        mark_distance[0] = cur_label
        for i in range(1, len(mark_distance)):
            if cur_level != distance_list_norm[i]:
                if cur_level:
                    cluster_list.append(sequence)
                cur_level = distance_list_norm[i]
                cur_label += 1
                sequence = []
            sequence.append(peaks[i])
            mark_distance[i] = cur_label
        cluster_list.append(sequence)
    return cluster_list


def cluster_duration(cluster_list):
    """
    Calculate each cluster's start and end points from two data sequences, separately
    @param cluster_list: cluster list which contains peak list in each cluster
    @return: duration list which contains each data sequence's start and end points in every cluster
    """
    duration_list = []
    for cluster_index in range(len(cluster_list)):
        if cluster_index == 0:
            start_point_level = min(cluster_list[cluster_index][0]) / 200
            if not start_point_level:
                start = 0
            else:
                start = min(cluster_list[cluster_index][0]) / 2
            first_seq_start = start
            second_seq_start = start
            if len(cluster_list) > 1:
                if (cluster_list[cluster_index][len(cluster_list[cluster_index]) - 1][0] + 100) >= \
                        cluster_list[cluster_index + 1][0][0]:
                    first_seq_end = cluster_list[cluster_index][len(cluster_list[cluster_index]) - 1][0] + 1
                else:
                    first_seq_end = cluster_list[cluster_index][len(cluster_list[cluster_index]) - 1][0] + 200
                if (cluster_list[cluster_index][len(cluster_list[cluster_index]) - 1][1] + 100) >= \
                        cluster_list[cluster_index + 1][0][1]:
                    second_seq_end = cluster_list[cluster_index][len(cluster_list[cluster_index]) - 1][1] + 1
                else:
                    second_seq_end = cluster_list[cluster_index][len(cluster_list[cluster_index]) - 1][1] + 200
            else:
                first_seq_end = cluster_list[cluster_index][len(cluster_list[cluster_index]) - 1][0] + 200
                second_seq_end = cluster_list[cluster_index][len(cluster_list[cluster_index]) - 1][1] + 200
        elif cluster_index == (len(cluster_list) - 1):
            max_seq_length = \
                max(cluster_list[cluster_index][len(cluster_list[cluster_index]) - 1][0] -
                    cluster_list[cluster_index][0][0],
                    cluster_list[cluster_index][len(cluster_list[cluster_index]) - 1][1] -
                    cluster_list[cluster_index][0][1]) + 100
            first_seq_start = cluster_list[cluster_index][0][0] - max_seq_length
            second_seq_start = cluster_list[cluster_index][0][1] - max_seq_length
            if first_seq_start < 0 or second_seq_start < 0:
                compensation = max((0 - first_seq_start), (0 - second_seq_start))
                first_seq_start += compensation
                second_seq_start += compensation
            first_seq_end = cluster_list[cluster_index][len(cluster_list[cluster_index]) - 1][0] + 200
            second_seq_end = cluster_list[cluster_index][len(cluster_list[cluster_index]) - 1][1] + 200
        else:
            if (cluster_list[cluster_index - 1][len(cluster_list[cluster_index - 1]) - 1][0] + 100) >= \
                    cluster_list[cluster_index][0][0]:
                first_seq_start = cluster_list[cluster_index - 1][len(cluster_list[cluster_index - 1]) - 1][0] + 1
            else:
                first_seq_start = cluster_list[cluster_index - 1][len(cluster_list[cluster_index - 1]) - 1][0] + 100
            if (cluster_list[cluster_index - 1][len(cluster_list[cluster_index - 1]) - 1][1] + 100) >= \
                    cluster_list[cluster_index][0][1]:
                second_seq_start = cluster_list[cluster_index - 1][len(cluster_list[cluster_index - 1]) - 1][1] + 1
            else:
                second_seq_start = cluster_list[cluster_index - 1][len(cluster_list[cluster_index - 1]) - 1][1] + 100

            if (cluster_list[cluster_index][len(cluster_list[cluster_index]) - 1][0] + 100) >= \
                    cluster_list[cluster_index + 1][0][0]:
                first_seq_end = cluster_list[cluster_index + 1][0][0] - 1
            else:
                first_seq_end = cluster_list[cluster_index][len(cluster_list[cluster_index]) - 1][0] + 100
            if (cluster_list[cluster_index][len(cluster_list[cluster_index]) - 1][1] + 100) >= \
                    cluster_list[cluster_index + 1][0][1]:
                second_seq_end = cluster_list[cluster_index + 1][0][1] - 1
            else:
                second_seq_end = cluster_list[cluster_index][len(cluster_list[cluster_index]) - 1][1] + 100
        first_seq = (first_seq_start, first_seq_end)
        second_seq = (second_seq_start, second_seq_end)
        duration_list.append([first_seq, second_seq])
    return duration_list


def cluster_video_out(first_img_list, second_img_list, duration_list, output_video_dp):
    """
    Generate video based on each cluster information from two data sequences
    @param first_img_list: first input image file path list
    @param second_img_list: second input image file path list
    @param duration_list: an input list which contains each data sequence's start and end points in every cluster
    @param output_video_dp: the target directory path of output video
    @return: output list of video file paths
    """
    video = cv2.VideoWriter()
    if hasattr(cv2, 'VideoWriter_fourcc'):
        fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    else:
        fourcc = cv2.cv.CV_FOURCC(*'XVID')
    video_list = []
    video_dp = os.path.join(output_video_dp, 'video_fluency_measurement_' + str(int(time.time())))
    if not os.path.exists(video_dp):
        os.mkdir(video_dp)
    for i in range(len(duration_list)):
        video_fp = os.path.join(video_dp, 'video_fluency_measurement_' + str(i + 1) + '.avi')
        video_list.append(video_fp)
        video.open(video_fp, fourcc, 60, (2048, 768), True)
        first_seq_start = duration_list[i][0][0]
        second_seq_start = duration_list[i][1][0]
        if duration_list[i][0][1] > (len(first_img_list) - 1):
            first_seq_end = len(first_img_list) - 1
        else:
            first_seq_end = duration_list[i][0][1]
        if duration_list[i][1][1] > (len(second_img_list) - 1):
            second_seq_end = len(second_img_list) - 1
        else:
            second_seq_end = duration_list[i][1][1]
        first_seq_length = duration_list[i][0][1] - duration_list[i][0][0] + 1
        second_seq_length = duration_list[i][1][1] - duration_list[i][1][0] + 1
        video_length = max(first_seq_length, second_seq_length)
        for j in range(video_length):
            if j >= first_seq_length or (first_seq_start + j) >= len(first_img_list):
                img1 = cv2.imread(first_img_list[first_seq_end])
            elif j < first_seq_length:
                img1 = cv2.imread(first_img_list[first_seq_start + j])
            if j >= second_seq_length or (second_seq_start + j) >= len(second_img_list):
                img2 = cv2.imread(second_img_list[second_seq_end])
            elif j < second_seq_length:
                img2 = cv2.imread(second_img_list[second_seq_start + j])
            video.write(np.concatenate((img1, img2), axis=1))
        video.release()
    return video_list


def video_generation(img_dp, video_fp):
    """
    Output  video from a given image directory path
    @param img_dp: input image directory path
    @param video_fp: input target video file path
    @return:
    """
    if os.path.exists(video_fp):
        os.remove(video_fp)
    img_list = os.listdir(img_dp)
    img_list.sort(key=CommonUtil.natural_keys)
    img_list = [os.path.join(img_dp, item) for item in img_list if
                os.path.splitext(item)[1] in Environment.IMG_FILE_EXTENSION]
    img = cv2.imread(os.path.join(img_dp, img_list[0]))
    height, width, channels = img.shape
    video = cv2.VideoWriter()
    if hasattr(cv2, 'VideoWriter_fourcc'):
        fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    else:
        fourcc = cv2.cv.CV_FOURCC(*'XVID')
    video.open(video_fp, fourcc, 15, (width, height), True)

    for img in img_list:
        video.write(cv2.imread(os.path.join(img_dp, img)))
    video.release()


def main():

    arg_parser = argparse.ArgumentParser(description='Video Fluency Measurement',
                                         formatter_class=ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument('-d', '--detection', action='store_true', dest='detection_flag', default=False,
                            help='convert video to images.', required=False)
    arg_parser.add_argument('-g', '--golden', action='store', dest='golden_waveform_fp', default=None,
                            help='Specify waveform information file of golden sample.', required=False)
    arg_parser.add_argument('-i', '--input', action='store', dest='input_waveform_dp', default=None,
                            help='Specify the folder of waveform information files.', required=False)
    arg_parser.add_argument('-o', '--output_dir', action='store', dest='output_dp', default=None,
                            help='Specify output dir path.', required=False)
    arg_parser.add_argument('-v', '--video_enable', action='store_true', dest='video_out_flag', default=False,
                            help='Output video flag to decide generate video clips or not after analysis.', required=False)
    arg_parser.add_argument('--advance', action='store_true', dest='advance', default=False,
                            help='advance mode for debug log', required=False)
    args = arg_parser.parse_args()
    logger = get_logger(__file__, args.advance)

    if args.output_dp:
        output_dp = args.output_dp
        output_fp = os.path.join(output_dp, 'video_fluency_measurement.json')
    else:
        output_fp = os.path.join(os.getcwd(), 'video_fluency_measurement.json')

    if os.path.exists(output_fp):
        with open(output_fp) as fh:
            result = json.load(fh)
    else:
        result = {}

    if args.golden_waveform_fp:
        golden_waveform_fp = args.golden_waveform_fp
        with open(golden_waveform_fp) as fh:
            golden = json.load(fh)
            golden_data = golden['data']
        current_test = os.path.basename(golden_waveform_fp)
        result[current_test] = {}
        result[current_test]['golden sample'] = golden_waveform_fp
        result[current_test]['comparison'] = []
        comparison_count = 0
        if args.input_waveform_dp:
            input_waveform_dp = args.input_waveform_dp
            waveform_fp_list = os.listdir(input_waveform_dp)
            waveform_fp_list = [os.path.join(input_waveform_dp, item) for item in waveform_fp_list]
            for waveform_fp in waveform_fp_list:
                with open(waveform_fp) as fh:
                    json_input = json.load(fh)
                    input_data = json_input['data']
                comparison_data = {'compared data': waveform_fp,
                                   'result clip': None,
                                   'latency': 0.0}
                result[current_test]['comparison'].append(comparison_data)
                if args.detection_flag:
                    if args.output_video_dp:
                        if not os.path.exists(args.output_video_dp):
                            os.mkdir(args.output_video_dp)
                        threshold = 0.9
                        sim_score = get_similarity_score(input_data, golden_data)
                        if sim_score < threshold:
                            d1norm = quantizer(golden_data)
                            d2norm = quantizer(input_data)
                            dtw, distance = dynamic_time_warping(d1norm, d2norm)
                            path = get_warp_path(dtw)
                            peak1 = find_peaks(d1norm)
                            peak2 = find_peaks(d2norm)
                            m_peak = target_peaks(path, peak1, peak2)
                            cluster_list = clustering(m_peak)
                            distance_list = get_distance(m_peak)
                            result[current_test]['comparison'][-1]['latency'] = get_average_latency(distance_list)
                            duration_list = cluster_duration(cluster_list)
                            # video_list = cluster_video_out(golden_img_list, input_img_list, v_duration, output_video_dp)
                            # result[current_test]['comparison'][-1]['result clip'] = video_list
                            comparison_count += 1
                        else:
                            logger.info("Similarity score of input data sequence greater than or equal to threshold")
                    else:
                        logger.error("Please specify output video dir path.")
        else:
            comparison_data = {'compared data': golden_waveform_fp,
                               'result clip': None,
                               'latency': 0.0}
            result[current_test]['comparison'].append(comparison_data)
            comparison_count += 1
            d1norm = quantizer(golden_data)
            last_idx = 0
            count = 0
            times = 0
            latency = []
            latency_times = [[]]
            for i in range(len(d1norm)):
                if d1norm[i] != 0 and (i - last_idx) > 0:
                    latency.append(i - last_idx)
                    last_idx = i
            logger.debug("latency: %s" % latency)
            target_latency = 0.0
            for j in range(len(latency)):
                if latency[j] < (Environment.DEFAULT_VIDEO_RECORDING_FPS / 5):
                    latency_times[times].append(latency[j])
                else:
                    times += 1
                    latency_times.append([])
            valid_times = 0
            for i in range(len(latency_times)):
                if len(latency_times[i]) == 20:
                    valid_times += 1
                    logger.debug("%s, %d" % (latency_times[i], len(latency_times[i])))
                    for item in latency_times[i]:
                        target_latency += item
                        count += 1
                else:
                    logger.debug("%s, %d" % (latency_times[i], len(latency_times[i])))
            target_latency /= count
            logger.debug("valid times: %d" % valid_times)
            logger.debug("latency of each frame: %f" % (1000.0 / Environment.DEFAULT_VIDEO_RECORDING_FPS))
        latency_list = [item['latency'] for item in result[current_test]['comparison']]
        latency_list.sort()
        non_zero_list = [item for item in latency_list if item != 0]
        if not non_zero_list:
            result[current_test]['average latency'] = 0.0
            result[current_test]['median latency'] = 0.0
        else:
            result[current_test]['average latency'] = sum(non_zero_list) / len(non_zero_list)
            result[current_test]['median latency'] = np.median(non_zero_list)
        result[current_test]['occurrence probability'] = float(len(non_zero_list)) / len(latency_list)
        result[current_test]['total comparison'] = comparison_count
        with open(output_fp, "wb") as fh:
            json.dump(result, fh, indent=2)

if __name__ == '__main__':
    main()
