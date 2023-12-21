WORKDIR = '../workdir'
DEVICE = 'cpu'

AREA_RNG = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
AREA_RNG_LBL = ['all', 'small', 'medium', 'large']
IOU_THRS = [0.5,  0.55, 0.6,  0.65, 0.7,  0.75, 0.8,  0.85, 0.9,  0.95]
M = [1, 10, 100]
MAX_LEN_SCORES = 5000
NUM_IOU_THRESHOLDS = 10
IOU_50_THRESHOLD_IDX = 0
IOU_75_THRESHOLD_IDX = 5
EPS = 1e-7
