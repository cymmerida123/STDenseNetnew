from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np
from models.MinMaxNorm import MinMaxNorm01
from os import listdir
from os.path import isfile,join
import rasterio
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class MilanSlidingWindowDataset(Dataset):
    def __init__(self,
                 filename,
                 gisdirpath,
                 traffic_type,
                 len_test,
                 crop, rows=[0, 100], cols=[0, 100],
                 mode:str = 'train',
                 window_size: int = 10,
                 stride: int = 24,
                 input_len: int = 24,
                 pred_len: int = 24,
                 flatten: bool = False,):
        hf = h5py.File(filename, 'r')
        milan_traffic = np.array(hf.get('data'))
        [T,WH,C] = milan_traffic.shape
        H = 100
        self.milan_traffic = milan_traffic.reshape(T,H,WH//H,C)
        self.traffic_type = traffic_type
        self.len_test = len_test
        self.milan_traffic_chosen = self._trafficloader(crop, rows, cols)
        self.milan_traffic_chosen_mmn, self.mmn = self.mmn()
        self.gisdirpath = gisdirpath
        self.milan_gis_mmn, self.gisnum = self._gisloader()
        self.mode = mode
        self.window_size = window_size
        self.input_len = input_len
        self.flatten = flatten
        self.pred_len = pred_len
        if H % window_size != 0:
            pad_size = (window_size - (H % window_size)) // 2
        else:
            pad_size = 0
        self.milan_traffic_chosen_mmn_pad = np.pad(self.milan_traffic_chosen_mmn,
                                     ((0, 0), (pad_size, pad_size),
                                      (pad_size, pad_size)),
                                     'constant', constant_values=0)
        self.milan_gis_mmn_pad = np.pad(self.milan_gis_mmn,
                                     ((0, 0), (pad_size, pad_size),
                                      (pad_size, pad_size)),
                                     'constant', constant_values=0)
        self.Hsize = self.milan_traffic_chosen_mmn_pad.shape[1]//window_size
        self.Wsize = self.milan_traffic_chosen_mmn_pad.shape[2]//window_size
        self.stride = stride
        
    def _trafficloader(self, crop, rows=[0, 100], cols=[0, 100]):
        if self.traffic_type == 'sms':
            data = self.milan_traffic[:, :, :, 0]
        elif self.traffic_type == 'call':
            data = self.milan_traffic[:, :, :, 1]
        elif self.traffic_type == 'internet':
            data = self.milan_traffic[:, :, :, 2]
        else:
            raise IOError("Unknown traffic type")

        # result = data.transpose(0,3,1,2)

        if crop:
            data = data[:, rows[0]:rows[1], cols[0]:cols[1]]
        return data

    def mmn(self):
        mmn = StandardScaler()
        T,H,W = self.milan_traffic_chosen.shape
        data_mmn = mmn.fit_transform(self.milan_traffic_chosen.reshape(T,H*W))

        return data_mmn.reshape(T,H,W), mmn
        # mmn = MinMaxNorm01()
        # data_train = self.milan_traffic_chosen[:-self.len_test]
        # mmn.fit(data_train)

        # return mmn.transform(self.milan_traffic_chosen), mmn

    def _gisloader(self):
        onlyfiles = [f for f in listdir(self.gisdirpath) if isfile(join(self.gisdirpath,f))]
        x_gis = []
        for f in onlyfiles:
            filepath = self.gisdirpath + '/' + f
            gisras = rasterio.open(filepath)
            gisinfo = gisras.read(1)
            H,W = gisinfo.shape
            x_gis.append(gisinfo.reshape(1,H,W))

        gisall = np.vstack(x_gis)
        C,H,W = gisall.shape
        mmn = StandardScaler()
        gisall_mmn = mmn.fit_transform(gisall.transpose(1,2,0).reshape(H*W,C))
        gisall_mmn = gisall_mmn.reshape(H,W,C).transpose(2,0,1)
        return gisall_mmn, C

    def __len__(self):
        if self.mode == 'train':
            return ((self.milan_traffic_chosen_mmn_pad.shape[0]-self.input_len-self.pred_len-self.len_test)//self.stride + 1) * self.Hsize * self.Wsize 
        elif self.mode == 'test':
            return ((self.len_test-self.pred_len) // self.stride +1)* self.Hsize * self.Wsize 
        else:
            raise IOError("Unknown mode")

    def __getitem__(self, index):
        n_slice = index // (self.Hsize * self.Wsize) * self.stride
        n_row = (index % (
            self.Hsize * self.Wsize)) // self.Wsize
        n_col = (index % (
            self.Hsize * self.Wsize)) % self.Wsize
        if self.mode == 'train':
            n_slice = n_slice
        elif self.mode == 'test':
            n_slice = n_slice + self.milan_traffic_chosen_mmn_pad.shape[0]-self.input_len-self.len_test
        else:
            raise IOError("Unknown mode")

        XC = self.milan_traffic_chosen_mmn_pad[n_slice:n_slice+self.input_len,
                                              n_row*self.window_size:(n_row+1)*self.window_size,
                                              n_col*self.window_size:(n_col+1)*self.window_size]
        XG = self.milan_gis_mmn_pad[:,
                                    n_row*self.window_size:(n_row+1)*self.window_size,
                                    n_col*self.window_size:(n_col+1)*self.window_size]
        y = self.milan_traffic_chosen_mmn_pad[n_slice+self.input_len:n_slice+self.input_len+self.pred_len,
                                              n_row*self.window_size:(n_row+1)*self.window_size,
                                              n_col*self.window_size:(n_col+1)*self.window_size]
        if self.flatten:
            XC = XC.reshape((self.input_len, self.window_size * self.window_size))
            XG = XG.reshape((self.gisnum, self.window_size * self.window_size))
            y = y.reshape((self.pred_len, self.window_size * self.window_size))

        return (XG, XC, y) 
