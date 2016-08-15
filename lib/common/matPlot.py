import matplotlib.pyplot as plt
from videoFluency import VideoFluency


class MatPlot(object):
    def __init__(self, f_data, s_data):
        self.f_data = f_data
        self.s_data = s_data
        self.video_obj = VideoFluency()
        self.indexes = []
        self.data_norm = []
        self.indexes.append(self.video_obj.find_peaks(self.f_data))
        self.indexes.append(self.video_obj.find_peaks(self.s_data))
        self.data_norm.append(self.video_obj.quantization(self.f_data))
        self.data_norm.append(self.video_obj.quantization(self.s_data))

    def subplot_data_peaks(self):
        plt.subplot(211)
        plt.plot(self.data_norm[0], 'bo', label='First Result')
        for i in range(len(self.indexes[0])):
            plt.plot(self.indexes[0][i], self.data_norm[0][self.indexes[0][i]], 'o', color='r')
        plt.legend()

        plt.subplot(212)
        plt.plot(self.data_norm[1], 'go', label='Second Result')
        for i in range(len(self.indexes[1])):
            plt.plot(self.indexes[1][i], self.data_norm[1][self.indexes[1][i]], 'o', color='r')
        plt.legend()
        plt.show()

    def plot_peak_mapping(self):
        dtw, distance = self.video_obj.dtw(self.data_norm[0], self.data_norm[1])
        path = self.video_obj.warp_path(dtw)
        peak1 = self.video_obj.find_peaks(self.data_norm[0])
        peak2 = self.video_obj.find_peaks(self.data_norm[1])
        m_peak = self.video_obj.target_peaks(path, peak1, peak2)
        for [map_f, map_s] in m_peak:
            plt.plot([map_f, map_s], [self.data_norm[0][map_f], self.data_norm[1][map_s]], 'r')

        plt.plot(self.data_norm[0], 'bo', label='First Result')
        plt.plot(self.data_norm[1], 'g^', label='Second Result')
        plt.axis([-100, max(len(self.data_norm[0]), len(self.data_norm[1]) + 100),
                  min(min(self.data_norm[0]), min(self.data_norm[1])),
                  max(max(self.data_norm[0]), max(self.data_norm[1])) * 1.2])
        plt.legend()
        plt.show()

    def plot_dtw_mapping(self):
        dtw, distance = self.video_obj.dtw(self.data_norm[0], self.data_norm[1])
        path = self.video_obj.warp_path(dtw)
        for [map_f, map_s] in path:
            plt.plot([map_f, map_s], [self.data_norm[0][map_f], self.data_norm[1][map_s]], 'r')
        plt.plot(self.data_norm[0], 'bo', label='First Result')
        plt.plot(self.data_norm[1], 'g^', label='Second Result')
        plt.axis([-100, max(len(self.data_norm[0]), len(self.data_norm[1]) + 100),
                  min(min(self.data_norm[0]), min(self.data_norm[1])),
                  max(max(self.data_norm[0]), max(self.data_norm[1])) * 1.2])
        plt.legend()
        plt.show()

    def plot_dtw_path(self):
        dtw, distance = self.video_obj.dtw(self.data_norm[0], self.data_norm[1])
        path = self.video_obj.warp_path(dtw)
        plt.imshow(dtw, interpolation='nearest', cmap='Greens')
        plt.gca().invert_yaxis()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid()
        plt.colorbar()
        if path:
            path_x = [point[0] for point in path]
            path_y = [point[1] for point in path]
            plt.plot(path_y, path_x, 'r')
        plt.show()

    @staticmethod
    def plot_waveform(waveform):
        plt.plot(waveform)
        plt.axis([-100, len(waveform) + 100, min(waveform) * 1.2, max(waveform) * 1.2])
        plt.show()

    @staticmethod
    def plot_two_waveform(f_waveform, s_waveform):
        plt.subplot(211)
        plt.plot(f_waveform, 'bo', label='First Result')
        plt.axis([-100, max(len(f_waveform), len(s_waveform) + 100),
                  min(min(f_waveform), min(s_waveform)),
                  max(max(f_waveform), max(s_waveform)) * 1.2])
        plt.legend()

        plt.subplot(212)
        plt.plot(s_waveform, 'g^', label='Second Result')
        plt.axis([-100, max(len(f_waveform), len(s_waveform) + 100),
                  min(min(f_waveform), min(s_waveform)),
                  max(max(f_waveform), max(s_waveform)) * 1.2])
        plt.legend()
        plt.show()
