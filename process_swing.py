import cv2 #opencv
import pandas as pd
import os
import numpy as np
pd.options.mode.chained_assignment = None 
import matplotlib.pyplot as plt

class DataProcessor:
    """
    Handles the loading and processing of the dataset.
    """
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.data = None

    def load_data(self):
        # Keep only the golfer from detected peaple
        all_csvs= [file for file in os.listdir(self.folder_path) if file.endswith('.csv')] # video and csv has to be in same file (one folder for each video and csv file)
        # It is usually the first person but to be sure we pick the person standing the widest
        filename=all_csvs[0].split('.')[0]
        if len(all_csvs)>1: 
                ankle_width=[]
                for i in range(len(all_csvs)):
                    data_current=pd.read_csv(self.folder_path + all_csvs[i])
                    ankle_width.append(data_current["left_ankle X"].iloc[0]-data_current["right_ankle X"].iloc[0])
                golfer_id=pd.Series(ankle_width).idxmax()
                if golfer_id>0:
                    filename=filename+'_'+str(golfer_id+1)

        self.data = pd.read_csv(self.folder_path+'/'+filename+'.csv')

    def split_swing(self):
        """
        Keeps only 3 key frame with the swing parts we are analyzing.
        """
        # Consider only time between backswing and finish
        halfway_back_ind=self.data['right_wrist_x'].idxmin()
        halfway_front_ind=self.data.right_wrist_x[self.data.index>halfway_back_ind].idxmax()
        middle_data=self.data[(self.data.index>halfway_back_ind)&(self.data.index<halfway_front_ind)]

        if middle_data.empty:
            halfway_back_ind=self.data.right_wrist_x[self.data.index<halfway_back_ind].idxmin()
            halfway_front_ind=self.data.right_wrist_x[self.data.index>halfway_back_ind].idxmax()
            middle_data=self.data[(self.data.index>halfway_back_ind)&(self.data.index<halfway_front_ind)]     
        # Find moment of ball contact as the lowest wrist point on y
        contact_frame=middle_data['right_wrist_y'].idxmax()

        # Isolate only backswing data
        back_data=self.data[(self.data.index<int(contact_frame))]
        # Find moment of top of backswing as the highest wrist point on y
        top_backswing_frame=back_data['right_wrist_y'].idxmin()
        # Find moment before start of the swing as the lowest wrist point on y before going halfway back
        halfway_back_data=self.data[self.data.index<halfway_back_ind]
        address_frame=halfway_back_data['right_wrist_y'].idxmax()

        self.data=self.data.iloc[[max([int(address_frame)-4,0]), int(top_backswing_frame), int(contact_frame)]]

        self.data['index']=self.data.index
        self.data=self.data.reset_index(drop=True)

         
    
    def preprocess_data(self):
        self.load_data()
        # This can be also adjusted when getting the data
        self.data=self.data.reset_index()
        self.data.columns = [col.replace(' X', '_x').replace(' Y', '_y').lower() for col in self.data.columns]
        self.data=self.data.drop(['video_timestamp'],axis=1)

        # Reduce the dataset to only include the 3 phases we are looking for
        self.split_swing()
        


class Evaluator:
    """
    Evaluating the correctness of positions and angles.
    Input dataset has three rows, each for a different swing part.
    """
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.swing_part_indices = {'address': 0, 'top': 1, 'contact': 2}  # Indices for each swing part
        self.head_point_address=[self.data_processor.data.nose_y.iloc[self.swing_part_indices['address']],self.data_processor.data.nose_x.iloc[self.swing_part_indices['address']]]
        self.results={}
    def evaluate_all_swing_parts(self):
        """
        Runs the evaluation for all swing parts automatically and returns a structured result.
        """
        for swing_part, index in self.swing_part_indices.items():
            self.results[swing_part] = self.evaluate_correctness(swing_part, index)
        return self.results

    def evaluate_correctness(self, swing_part, swing_part_id):
        """
        Evaluates the correctness based on the provided swing part and its ID.
        """
        data = self.data_processor.data

        if swing_part == 'address':
            return self.evaluate_address(data, swing_part_id)
        elif swing_part == 'top':
            return self.evaluate_top(data, swing_part_id)
        elif swing_part == 'contact':
            return self.evaluate_contact(data, swing_part_id)


    def evaluate_address(self, data, swing_part_id):
        results = {
            "correct_midpoint": int(data['midpoint_x'].iloc[swing_part_id] - data['left_wrist_x'].iloc[swing_part_id] < 0), # Wrist should be on the left from center.
            "correct_arm_angle": int(165 <= data['arm_angle'].iloc[swing_part_id] <= 180)   # Left arm should be straight.
        }
        return results

    def evaluate_top(self, data, swing_part_id):
        head_point_current = np.array([data['nose_y'].iloc[swing_part_id], data['nose_x'].iloc[swing_part_id]])
        results = {
            "correct_pelvis": int(150 <= data['pelvis_angle'].iloc[swing_part_id] <= 180), # Leg, hip and shoulder should make straight line.
            # "correct_arm_angle": int(150 <= data['arm_angle'].iloc[swing_part_id] <= 180), # Left out, usually wrong detection in both models
            "correct_head": int(self.calculate_head_distance(head_point_current, self.head_point_address) <= 30) # Little to no head movement from address
        }
        return results

    def evaluate_contact(self, data, swing_part_id):
        head_point_current = np.array([data['nose_y'].iloc[swing_part_id], data['nose_x'].iloc[swing_part_id]])
        results = {
            "correct_shoulder_ankle": int(data['left_ankle_x'].iloc[swing_part_id] - data['left_shoulder_x'].iloc[swing_part_id] >= 0), # Shoulder shoud not go beyond ankle.
            "correct_head": int(self.calculate_head_distance(head_point_current, self.head_point_address) <= 30), # Little to no head movement from address.
            "correct_knee_angle": int(165 <= data['knee_angle'].loc[swing_part_id] <= 180), # Straight leg
            "correct_arm_angle": int(160 <= data['arm_angle'].iloc[swing_part_id] <= 180) # Straight arm
        }
        return results

    @staticmethod
    def calculate_head_distance(head_point_current, head_point_address):
        return np.linalg.norm(np.array(head_point_current).flatten()- np.array(head_point_address).flatten()) # Calculates distance of the head from the address point.



class SwingScorer:
    """
    Golf-style swing score: LOWER is BETTER (0 == physically perfect swing).

    For every key joint angle the ideal is a perfectly STRAIGHT line = 180 deg
    (straight lead arm, straight lead leg, straight ankle-hip-shoulder turn line).
    The score is the DISTANCE FROM THAT IDEAL, measured continuously and always on:
    a swing 4 deg from straight scores 4, one 20 deg from straight scores 20. This
    rewards precision everywhere, so swings never cluster at 0 the way a pass/fail
    band does - which is what a ranked leaderboard needs.

    (The pass/fail bands still live in `Evaluator` and drive the red/green overlay;
    the score no longer uses them.)

    Units
    -----
    * Angle checks       -> penalty is DEGREES away from the ideal `target` (180).
    * Head steadiness     -> ideal is zero drift; penalty is head travel from its
      address spot, divided by the golfer's body height (scale-invariant, so
      distance-to-camera and player height do not skew it) and put on the same
      ~degree scale via POS_GAIN.
    * Directional checks (hands/shoulder) -> these are 'must not cross' constraints,
      not precision targets, so they only score when on the wrong side of the line.

    Each check's penalty is capped (CAP) so one blown MediaPipe detection cannot
    produce an absurd leaderboard number. The total is the sum of all checks.

    Extensibility: to score against a real professional instead of golf theory,
    set a check's `target` to the pro's measured angle - nothing else changes.
    """

    POS_GAIN = 100.0   # a positional error of one full body-height == 100 strokes
    CAP = 25.0         # max strokes any single check may contribute
    HEAD_TOL = 0.0     # ideal head drift is zero; any movement from address is scored

    # Declarative check table.
    #   kind='angle'  -> penalise |target - col|  (target = 180 = perfectly straight)
    #   kind='pos_ge' -> `a_col` should be >= `b_col`; only penalise how far it falls short
    #   kind='head'   -> penalise nose drift from its address spot (ideal = 0)
    CHECKS = [
        dict(phase='address', name='correct_arm_angle',      kind='angle',  col='arm_angle',    target=180,
             label='Lead arm straight at address'),
        dict(phase='address', name='correct_midpoint',       kind='pos_ge', a_col='left_wrist_x', b_col='midpoint_x',
             label='Hands ahead of stance centre at address'),
        dict(phase='top',     name='correct_pelvis',         kind='angle',  col='pelvis_angle', target=180,
             label='Full turn: ankle-hip-shoulder in line at top'),
        dict(phase='top',     name='correct_head',           kind='head',
             label='Head steady at top of backswing'),
        dict(phase='contact', name='correct_knee_angle',     kind='angle',  col='knee_angle',   target=180,
             label='Lead leg straight at contact'),
        dict(phase='contact', name='correct_arm_angle',      kind='angle',  col='arm_angle',    target=180,
             label='Lead arm straight at contact'),
        dict(phase='contact', name='correct_shoulder_ankle', kind='pos_ge', a_col='left_ankle_x', b_col='left_shoulder_x',
             label='Lead shoulder not past front foot at contact'),
        dict(phase='contact', name='correct_head',           kind='head',
             label='Head steady at contact'),
    ]

    def __init__(self, data_processor):
        self.data = data_processor.data
        self.phase_index = {'address': 0, 'top': 1, 'contact': 2}
        # Body height (px) at address used to make positional penalties scale-invariant.
        addr = self.phase_index['address']
        self.body_scale = abs(self.data['left_ankle_y'].iloc[addr] - self.data['nose_y'].iloc[addr])
        if self.body_scale < 1:               # degenerate detection -> avoid divide-by-zero
            self.body_scale = 1.0
        # Address nose position, ordered [y, x] to match Evaluator.calculate_head_distance.
        self.head_address = np.array([self.data['nose_y'].iloc[addr], self.data['nose_x'].iloc[addr]])

    def _angle_penalty(self, col, idx, target):
        value = float(self.data[col].iloc[idx])
        return value, min(self.CAP, abs(target - value))         # degrees away from ideal (straight = 180)

    def _pos_ge_penalty(self, a_col, b_col, idx):
        a = float(self.data[a_col].iloc[idx])
        b = float(self.data[b_col].iloc[idx])
        shortfall_px = max(0.0, b - a)                            # a should be >= b
        penalty = shortfall_px / self.body_scale * self.POS_GAIN
        return (a - b), min(self.CAP, penalty)

    def _head_penalty(self, idx):
        nose = np.array([self.data['nose_y'].iloc[idx], self.data['nose_x'].iloc[idx]])
        dist_px = float(np.linalg.norm(nose - self.head_address))
        tol_px = self.HEAD_TOL * self.body_scale
        penalty = max(0.0, dist_px - tol_px) / self.body_scale * self.POS_GAIN
        return dist_px, min(self.CAP, penalty)

    def score(self):
        """
        Returns a dict:
          total       -> single golf-style score (float, lower is better, 0 == ideal)
          phase       -> {'address':.., 'top':.., 'contact':..} penalty subtotals
          checks      -> per-check breakdown [{phase,name,label,actual,penalty,passed}, ...]
        """
        checks, phase_totals = [], {'address': 0.0, 'top': 0.0, 'contact': 0.0}
        for c in self.CHECKS:
            idx = self.phase_index[c['phase']]
            if c['kind'] == 'angle':
                actual, penalty = self._angle_penalty(c['col'], idx, c['target'])
            elif c['kind'] == 'pos_ge':
                actual, penalty = self._pos_ge_penalty(c['a_col'], c['b_col'], idx)
            else:  # head
                actual, penalty = self._head_penalty(idx)
            penalty = round(penalty, 2)
            phase_totals[c['phase']] += penalty
            checks.append({'phase': c['phase'], 'name': c['name'], 'label': c['label'],
                           'actual': round(actual, 1), 'penalty': penalty, 'passed': penalty == 0.0})
        total = round(sum(phase_totals.values()), 1)
        phase_totals = {k: round(v, 1) for k, v in phase_totals.items()}
        return {'total': total, 'phase': phase_totals, 'checks': checks}

    @staticmethod
    def _severity(penalty):
        """Coarse tier for the scorecard (the score itself is continuous)."""
        if penalty == 0:  return 'ideal'
        if penalty < 5:   return 'good '
        if penalty < 12:  return 'off  '
        return 'BIG  '

    def scorecard_text(self, player_name='Player'):
        """Booth-friendly scorecard string."""
        r = self.score()
        lines = [f"==== GOLF SWING SCORE : {player_name} ====",
                 f"  FINAL SCORE: {r['total']:.1f}  (lower is better, 0 = perfectly straight/still)",
                 f"  By phase -> address {r['phase']['address']:.1f} | "
                 f"top {r['phase']['top']:.1f} | contact {r['phase']['contact']:.1f}",
                 "  ------------------------------------------------------------"]
        for c in r['checks']:
            lines.append(f"  [{self._severity(c['penalty'])}] {c['phase']:<7} {c['label']:<48} +{c['penalty']:.1f}")
        return "\n".join(lines)


class VideoProcessor:
    """
    Handles frame extraction, writing results into frame and exporting frame
    """
    def __init__(self, folder_path,data_processor,evaluator):
        self.folder_path = folder_path
        self.data=data_processor.data
        self.correct=evaluator.evaluate_all_swing_parts()
        self.swing_part_indices = {'address': 0, 'top': 1, 'contact': 2}  # Indices for each swing part
    
    def print_swing_analysis(self):
        """
        Prints the messages based on evaluations.
        """
        messages = {
            'correct_midpoint': 'WRONG: Arms should be positioned more to the left side of the center of feet.',
            'correct_arm_angle': 'WRONG: Left arm should be straight at this point.',
            'correct_pelvis': 'WRONG: Left ankle, left hip and right shoulder angle should form a straight line. Try turning more into the backswing.',
            'correct_head':  'WRONG: Head should remain relatively still until contact.',
            'correct_shoulder_ankle': 'WRONG: Left shoulder should not go beyond the front foot at this point.',
            'correct_knee_angle': 'WRONG: Knee should not bend as much.'
        }
        #Iterate over each swing part and create the analysis message
        analysis = []
        for swing_part, checks in self.correct.items():
            part_analysis = [f"Swing part {swing_part.upper()}: "]
            messages_list = []
            for check, value in checks.items():
                if value == 0:
                    messages_list.append("-> " + messages[check])

            if messages_list:
                analysis.append("\n".join([part_analysis[0]] + messages_list))

        print("\n\n".join(analysis))

    def save_frame(self):
        """
        Captures the 3 main frames, plots the sought after angles and points based (colored based on evaluatuon) and saves them.
        """
        video_files = [file for file in os.listdir(self.folder_path) if file.endswith('.mp4')]
        video_path=video_files[0]

        for swing_part, index in self.swing_part_indices.items():
            cap = cv2.VideoCapture(self.folder_path+'/'+video_path)

            cap.set(cv2.CAP_PROP_POS_FRAMES, self.data['index'].iloc[index])
            ret, frame = cap.read()

            frame=self.plot_points(frame, swing_part)

            frame_path =self.folder_path+'/'+ video_path.split(".")[0]+ f'_frame_{swing_part}.jpg'
            cv2.imwrite(frame_path, frame)
            cap.release()

    def plot_line(self, frame, start_point, end_point, is_correct):
        color = (0, 255, 0) if is_correct else (0, 0, 255)
        cv2.line(frame, start_point, end_point, color, 2)

    def plot_circle(self, frame, center, is_correct, radius=6, thickness=-1):
        color = (0, 255, 0) if is_correct else (0, 0, 255)
        cv2.circle(frame, center, radius, color, thickness)

    def plot_text(self, frame, text, position, is_correct):
        color = (0, 255, 0) if is_correct else (0, 0, 255)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def plot_points(self, frame, swing_part):
        swing_part_id = self.swing_part_indices[swing_part]
        evaluation_results = self.correct[swing_part]
        if swing_part == 'address':
            self.plot_circle(frame, (int(self.data['midpoint_x'].iloc[swing_part_id]), int(self.data['midpoint_y'].iloc[swing_part_id])), evaluation_results['correct_midpoint'])
            self.plot_line(frame, (int(self.data['right_ankle_x'].iloc[swing_part_id]), int(self.data['right_ankle_y'].iloc[swing_part_id])), (int(self.data['left_ankle_x'].iloc[swing_part_id]), int(self.data['left_ankle_y'].iloc[swing_part_id])), evaluation_results['correct_midpoint'])
            self.plot_line(frame, (int(self.data['midpoint_x'].iloc[swing_part_id]), int(self.data['midpoint_y'].iloc[swing_part_id])), (int(self.data['midpoint_x'].iloc[swing_part_id]), int(self.data['midpoint_y'].iloc[swing_part_id]) - 150), evaluation_results['correct_midpoint'])

            arm_angle_text = f'Arm Angle: {self.data["arm_angle"].iloc[swing_part_id]:.2f}'
            self.plot_text(frame, arm_angle_text, (10, 120), evaluation_results['correct_arm_angle'])
            self.plot_line(frame, (self.data['left_shoulder_x'].iloc[swing_part_id], self.data['left_shoulder_y'].iloc[swing_part_id]), (self.data['left_elbow_x'].iloc[swing_part_id], self.data['left_elbow_y'].iloc[swing_part_id]), evaluation_results['correct_arm_angle'])
            self.plot_line(frame, (self.data['left_elbow_x'].iloc[swing_part_id], self.data['left_elbow_y'].iloc[swing_part_id]), (self.data['left_wrist_x'].iloc[swing_part_id], self.data['left_wrist_y'].iloc[swing_part_id]), evaluation_results['correct_arm_angle'])
        elif swing_part == 'contact':
            shoulders_inclination_text = f'Shoulders inclination: {self.data["shoulders_inclination"].iloc[swing_part_id]:.2f}'
            self.plot_text(frame, shoulders_inclination_text, (10, 20), True)  # True for always yellow
            self.plot_line(frame, (self.data['left_shoulder_x'].iloc[swing_part_id], self.data['left_shoulder_y'].iloc[swing_part_id]), (self.data['left_shoulder_x'].iloc[swing_part_id], self.data['left_shoulder_y'].iloc[swing_part_id]), True)


            hips_inclination_text = f'Hips inclination: {self.data["hips_inclination"].iloc[swing_part_id]:.2f}'
            self.plot_text(frame, hips_inclination_text, (10, 45), True)  # True for always yellow
            self.plot_line(frame, (self.data['left_hip_x'].iloc[swing_part_id], self.data['left_hip_y'].iloc[swing_part_id]), (self.data['left_hip_x'].iloc[swing_part_id], self.data['left_hip_y'].iloc[swing_part_id]), True)


            knee_angle_text = f'Knee Angle: {self.data["knee_angle"].iloc[swing_part_id]:.2f}'
            self.plot_text(frame, knee_angle_text, (10, 70), evaluation_results['correct_knee_angle'])
            self.plot_line(frame, (self.data['left_hip_x'].iloc[swing_part_id], self.data['left_hip_y'].iloc[swing_part_id]), (self.data['left_knee_x'].iloc[swing_part_id], self.data['left_knee_y'].iloc[swing_part_id]), evaluation_results['correct_knee_angle'])
            self.plot_line(frame, (self.data['left_knee_x'].iloc[swing_part_id], self.data['left_knee_y'].iloc[swing_part_id]), (self.data['left_ankle_x'].iloc[swing_part_id], self.data['left_ankle_y'].iloc[swing_part_id]), evaluation_results['correct_knee_angle'])

            self.plot_circle(frame, (self.data['left_ankle_x'].iloc[swing_part_id], self.data['left_ankle_y'].iloc[swing_part_id]), evaluation_results['correct_shoulder_ankle'])
            self.plot_line(frame, (self.data['left_ankle_x'].iloc[swing_part_id], self.data['left_ankle_y'].iloc[swing_part_id]), (self.data['left_ankle_x'].iloc[swing_part_id], self.data['left_ankle_y'].iloc[swing_part_id] - 200), evaluation_results['correct_shoulder_ankle'])
            
            if self.data['nose_x'].iloc[swing_part_id]!=0 and self.data['nose_y'].iloc[swing_part_id]!=0:
                self.plot_line(frame, (self.data['nose_x'].iloc[0], self.data['nose_y'].iloc[0]), (self.data['nose_x'].iloc[swing_part_id], self.data['nose_y'].iloc[swing_part_id]), evaluation_results['correct_head'])
                self.plot_circle(frame, (self.data['nose_x'].iloc[swing_part_id], self.data['nose_y'].iloc[swing_part_id]), evaluation_results['correct_head'])
                self.plot_circle(frame, (self.data['nose_x'].iloc[0], self.data['nose_y'].iloc[0]), evaluation_results['correct_head'], 30, 2)
            else:
                self.plot_circle(frame, (self.data['nose_x'].iloc[0], self.data['nose_y'].iloc[0]), evaluation_results['correct_head'], 30, 2)

            arm_angle_text = f'Arm Angle: {self.data["arm_angle"].iloc[swing_part_id]:.2f}'
            self.plot_text(frame, arm_angle_text, (10, 120), evaluation_results['correct_arm_angle'])
            self.plot_line(frame, (self.data['left_shoulder_x'].iloc[swing_part_id], self.data['left_shoulder_y'].iloc[swing_part_id]), (self.data['left_elbow_x'].iloc[swing_part_id], self.data['left_elbow_y'].iloc[swing_part_id]), evaluation_results['correct_arm_angle'])
            self.plot_line(frame, (self.data['left_elbow_x'].iloc[swing_part_id], self.data['left_elbow_y'].iloc[swing_part_id]), (self.data['left_wrist_x'].iloc[swing_part_id], self.data['left_wrist_y'].iloc[swing_part_id]), evaluation_results['correct_arm_angle'])
        
        elif swing_part == 'top':

            shoulders_inclination_text = f'Shoulders inclination: {self.data["shoulders_inclination"].iloc[swing_part_id]:.2f}'
            self.plot_text(frame, shoulders_inclination_text, (10, 20), True) 
            self.plot_line(frame, (self.data['left_shoulder_x'].iloc[swing_part_id], self.data['left_shoulder_y'].iloc[swing_part_id]), (self.data['left_shoulder_x'].iloc[swing_part_id], self.data['left_shoulder_y'].iloc[swing_part_id]), True)

            hips_inclination_text = f'Hips inclination: {self.data["hips_inclination"].iloc[swing_part_id]:.2f}'
            self.plot_text(frame, hips_inclination_text, (10, 45), True)  
            self.plot_line(frame, (self.data['left_hip_x'].iloc[swing_part_id], self.data['left_hip_y'].iloc[swing_part_id]), (self.data['left_hip_x'].iloc[swing_part_id], self.data['left_hip_y'].iloc[swing_part_id]), True)


            pelvis_angle_text = f'Pelvis Angle: {self.data["pelvis_angle"].iloc[swing_part_id]:.2f}'
            self.plot_text(frame, pelvis_angle_text, (10, 95), evaluation_results['correct_pelvis'])
            self.plot_line(frame, (self.data['left_hip_x'].iloc[swing_part_id], self.data['left_hip_y'].iloc[swing_part_id]), (self.data['left_ankle_x'].iloc[swing_part_id], self.data['left_ankle_y'].iloc[swing_part_id]), evaluation_results['correct_pelvis'])
            self.plot_line(frame, (self.data['left_hip_x'].iloc[swing_part_id], self.data['left_hip_y'].iloc[swing_part_id]), (self.data['right_shoulder_x'].iloc[swing_part_id], self.data['right_shoulder_y'].iloc[swing_part_id]), evaluation_results['correct_pelvis'])


            if self.data['nose_x'].iloc[swing_part_id]!=0 and self.data['nose_y'].iloc[swing_part_id]!=0:
                self.plot_line(frame, (self.data['nose_x'].iloc[0], self.data['nose_y'].iloc[0]), (self.data['nose_x'].iloc[swing_part_id], self.data['nose_y'].iloc[swing_part_id]), evaluation_results['correct_head'])
                self.plot_circle(frame, (self.data['nose_x'].iloc[swing_part_id], self.data['nose_y'].iloc[swing_part_id]), evaluation_results['correct_head'])
                self.plot_circle(frame, (self.data['nose_x'].iloc[0], self.data['nose_y'].iloc[0]), evaluation_results['correct_head'], 30, 2)
            else:
                self.plot_circle(frame, (self.data['nose_x'].iloc[0], self.data['nose_y'].iloc[0]), evaluation_results['correct_head'], 30, 2)
            # arm_angle_text = f'Arm Angle: {self.data["arm_angle"].iloc[swing_part_id]:.2f}'
            # self.plot_text(frame, arm_angle_text, (10, 120), evaluation_results['correct_arm_angle'])
            # self.plot_line(frame, (self.data['left_shoulder_x'].iloc[swing_part_id], self.data['left_shoulder_y'].iloc[swing_part_id]), (self.data['left_elbow_x'].iloc[swing_part_id], self.data['left_elbow_y'].iloc[swing_part_id]), evaluation_results['correct_arm_angle'])
            # self.plot_line(frame, (self.data['left_elbow_x'].iloc[swing_part_id], self.data['left_elbow_y'].iloc[swing_part_id]), (self.data['left_wrist_x'].iloc[swing_part_id], self.data['left_wrist_y'].iloc[swing_part_id]), evaluation_results['correct_arm_angle'])

        return frame
   

        
    