import os
import configparser
import numpy as np
from ._base_dataset import _BaseDataset
from .. import utils
from ..utils import TrackEvalException

class AirMot(_BaseDataset):
    id_to_class = {0: 'airplane', 1: 'person', 2: 'baggage_tug', 3: 'follow_me_vehicle'}
    class_to_id = {v: k for k, v in id_to_class.items()}

    @staticmethod
    def get_default_dataset_config():
        code_path = utils.get_code_path()
        default_config = {'GT_FOLDER': os.path.join(code_path, 'data/gt/airmot/'), 'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/airmot/'), 'OUTPUT_FOLDER': None, 'TRACKERS_TO_EVAL': None, 'CLASSES_TO_EVAL': list(AirMot.id_to_class.values()), 'SPLIT_TO_EVAL': 'val', 'PRINT_CONFIG': True, 'TRACKER_SUB_FOLDER': '', 'OUTPUT_SUB_FOLDER': 'output', 'SEQMAP_FOLDER': None, 'SEQMAP_FILE': None, 'SEQ_INFO': None, 'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt'}
        return default_config

    def __init__(self, config=None):
        super().__init__()
        self.config = utils.init_config(config, self.get_default_dataset_config(), self.get_name())
        self.gt_fol = os.path.join(self.config['GT_FOLDER'], self.config['SPLIT_TO_EVAL'])
        self.tracker_fol = self.config['TRACKERS_FOLDER']
        self.output_fol = self.config['OUTPUT_FOLDER']
        if self.output_fol is None:
            self.output_fol = self.config['TRACKERS_FOLDER']
        self.output_sub_fol = self.config['OUTPUT_SUB_FOLDER']
        self.class_list = [cls.lower() for cls in self.config['CLASSES_TO_EVAL'] if cls.lower() in self.class_to_id]
        if len(self.class_list) != len(self.config['CLASSES_TO_EVAL']):
            print('警告: CLASSES_TO_EVAL 中包含无效类别, 已被忽略。')
        self.seq_list, self.seq_lengths = self._get_seq_info()
        if not self.seq_list:
            raise TrackEvalException('没有选择任何序列进行评估。')
        for seq in self.seq_list:
            curr_file = self.config['GT_LOC_FORMAT'].format(gt_folder=self.gt_fol, seq=seq)
            if not os.path.isfile(curr_file):
                raise TrackEvalException(f'GT 文件未找到: {curr_file}')
        if self.config['TRACKERS_TO_EVAL'] is None:
            self.tracker_list = [t for t in os.listdir(self.tracker_fol) if os.path.isdir(os.path.join(self.tracker_fol, t))]
        else:
            self.tracker_list = self.config['TRACKERS_TO_EVAL']
        for tracker in self.tracker_list:
            for seq in self.seq_list:
                curr_file = os.path.join(self.tracker_fol, tracker, self.config['TRACKER_SUB_FOLDER'], seq + '.txt')
                if not os.path.isfile(curr_file):
                    raise TrackEvalException(f'跟踪器文件未找到: {curr_file}')

    def _get_seq_info(self):
        if self.config['SEQ_INFO']:
            return (list(self.config['SEQ_INFO'].keys()), self.config['SEQ_INFO'])
        seqmap_file = None
        if self.config['SEQMAP_FILE']:
            if isinstance(self.config['SEQMAP_FILE'], list):
                seqmap_file = self.config['SEQMAP_FILE'][0]
            else:
                seqmap_file = self.config['SEQMAP_FILE']
        if seqmap_file is None:
            seqmap_folder = self.config['SEQMAP_FOLDER'] or os.path.join(self.config['GT_FOLDER'], 'seqmaps')
            seqmap_file = os.path.join(seqmap_folder, f"airmot-{self.config['SPLIT_TO_EVAL']}.txt")
        if not os.path.isfile(seqmap_file):
            print(f'警告: seqmap 文件未找到: {seqmap_file}。将尝试从目录中自动发现序列。')
            seq_list = [f for f in os.listdir(self.gt_fol) if os.path.isdir(os.path.join(self.gt_fol, f))]
        else:
            with open(seqmap_file) as f:
                f.readline()
                seq_list = [row.strip() for row in f if row.strip()]
        seq_lengths = {}
        for seq in seq_list:
            ini_file = os.path.join(self.gt_fol, seq, 'seqinfo.ini')
            if not os.path.isfile(ini_file):
                raise TrackEvalException(f'seqinfo.ini 文件未找到: {ini_file}')
            ini_data = configparser.ConfigParser()
            ini_data.read(ini_file)
            seq_lengths[seq] = int(ini_data['Sequence']['seqLength'])
        return (seq_list, seq_lengths)

    def _load_raw_file(self, tracker, seq, is_gt):
        if is_gt:
            file_path = self.config['GT_LOC_FORMAT'].format(gt_folder=self.gt_fol, seq=seq)
        else:
            file_path = os.path.join(self.tracker_fol, tracker, self.config['TRACKER_SUB_FOLDER'], seq + '.txt')
        with open(file_path, 'r') as f:
            read_data = [line.strip().split() for line in f.readlines() if line.strip()]
        num_timesteps = self.seq_lengths[seq]
        data_keys = ['ids', 'classes', 'dets']
        if not is_gt:
            data_keys.append('tracker_confidences')
        raw_data = {}
        for k in data_keys:
            if k == 'dets':
                raw_data[k] = [np.empty((0, 4), dtype=np.float64) for _ in range(num_timesteps)]
            elif k in ['ids', 'gt_ids', 'tracker_ids']:
                raw_data[k] = [np.empty(0, dtype=np.int64) for _ in range(num_timesteps)]
            else:
                raw_data[k] = [np.empty(0) for _ in range(num_timesteps)]
        for row in read_data:
            try:
                frame = int(row[0]) - 1
                if frame < 0 or frame >= num_timesteps:
                    continue
                track_id = int(row[1])
                class_id = int(row[2])
                class_name = self.id_to_class.get(class_id)
                if class_name not in self.class_list:
                    continue
                det = np.array([float(row[3]), float(row[4]), float(row[5]), float(row[6])])
                raw_data['ids'][frame] = np.append(raw_data['ids'][frame], track_id)
                raw_data['classes'][frame] = np.append(raw_data['classes'][frame], class_name)
                raw_data['dets'][frame] = np.vstack([raw_data['dets'][frame], det])
                if not is_gt:
                    score = float(row[7])
                    raw_data['tracker_confidences'][frame] = np.append(raw_data['tracker_confidences'][frame], score)
            except (IndexError, ValueError):
                raise TrackEvalException(f'文件 {file_path} 中包含无效行: {row}')
        if is_gt:
            key_map = {'ids': 'gt_ids', 'classes': 'gt_classes', 'dets': 'gt_dets'}
        else:
            key_map = {'ids': 'tracker_ids', 'classes': 'tracker_classes', 'dets': 'tracker_dets'}
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)
        raw_data['num_timesteps'] = num_timesteps
        raw_data['seq'] = seq
        return raw_data

    def get_preprocessed_seq_data(self, raw_data, cls):
        self._check_unique_ids(raw_data)
        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'tracker_confidences', 'similarity_scores']
        data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
        unique_gt_ids = set()
        unique_tracker_ids = set()
        num_gt_dets = 0
        num_tracker_dets = 0
        for t in range(raw_data['num_timesteps']):
            gt_mask = raw_data['gt_classes'][t] == cls
            data['gt_ids'][t] = raw_data['gt_ids'][t][gt_mask]
            data['gt_dets'][t] = raw_data['gt_dets'][t][gt_mask]
            tracker_mask = raw_data['tracker_classes'][t] == cls
            data['tracker_ids'][t] = raw_data['tracker_ids'][t][tracker_mask]
            data['tracker_dets'][t] = raw_data['tracker_dets'][t][tracker_mask]
            data['tracker_confidences'][t] = raw_data['tracker_confidences'][t][tracker_mask]
            data['similarity_scores'][t] = self._calculate_similarities(data['gt_dets'][t], data['tracker_dets'][t])
            unique_gt_ids.update(data['gt_ids'][t])
            unique_tracker_ids.update(data['tracker_ids'][t])
            num_gt_dets += len(data['gt_ids'][t])
            num_tracker_dets += len(data['tracker_ids'][t])
        if unique_gt_ids:
            gt_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(list(unique_gt_ids)))}
            for t in range(raw_data['num_timesteps']):
                if len(data['gt_ids'][t]) > 0:
                    data['gt_ids'][t] = np.array([gt_id_map[i] for i in data['gt_ids'][t]], dtype=np.int64)
        if unique_tracker_ids:
            tracker_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(list(unique_tracker_ids)))}
            for t in range(raw_data['num_timesteps']):
                if len(data['tracker_ids'][t]) > 0:
                    data['tracker_ids'][t] = np.array([tracker_id_map[i] for i in data['tracker_ids'][t]], dtype=np.int64)
        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_dets'] = num_gt_dets
        data['num_tracker_ids'] = len(unique_tracker_ids)
        data['num_gt_ids'] = len(unique_gt_ids)
        data['num_timesteps'] = raw_data['num_timesteps']
        data['seq'] = raw_data['seq']
        self._check_unique_ids(data, after_preproc=True)
        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        return self._calculate_box_ious(gt_dets_t, tracker_dets_t, box_format='xywh')
