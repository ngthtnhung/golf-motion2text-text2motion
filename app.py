# app.py - Golf Swing Analyzer Web App (FULL, chạy được ngay)

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import torch
import os

# ================== CONFIG ==================
MODEL_PATH = "golf_phase_model.pth"  # Đường dẫn file model đã save

# ================== LOAD MEDIA PIPE ==================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

# ================== LOAD MODEL LSTM ==================
class GolfPhaseLSTM(torch.nn.Module):
    def __init__(self, input_size=99 + 5, hidden_size=128, num_classes=8):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.fc_phase = torch.nn.Linear(hidden_size * 2, num_classes)
        self.fc_class = torch.nn.Linear(hidden_size * 2, 1)  # giữ để load state_dict khớp

    def forward(self, x):
        out, (h, c) = self.lstm(x)  # h shape: (4, batch, 128) nếu bidirectional + 2 layers
        
        # Phase prediction: dùng toàn bộ sequence output
        phase_out = self.fc_phase(out)  # (batch, seq_len, 8)
        
        # Class prediction: concat hidden state cuối của forward và backward (layer 2)
        # h[-2] = forward last layer, h[-1] = backward last layer
        last_hidden = torch.cat((h[-2], h[-1]), dim=-1)  # shape: (batch, 128 + 128) = (batch, 256)
        class_out = torch.sigmoid(self.fc_class(last_hidden))  # (batch, 1)
        
        return phase_out, class_out.squeeze(-1)  # squeeze để thành (batch,)
    
model = GolfPhaseLSTM()
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    st.success("Model loaded thành công!")
else:
    st.error(f"Không tìm thấy file model: {MODEL_PATH}. Hãy save model từ notebook trước.")

# ================== HÀM EXTRACT KEYPOINTS (từ Cell 3) ==================
def extract_keypoints(frames):
    keypoints_seq = []
    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
            keypoints_seq.append(keypoints)
        else:
            keypoints_seq.append(np.zeros(99))  # Padding
    return np.array(keypoints_seq)

# ================== HÀM BIO FEATURES (từ Cell 4) ==================
def extract_biomechanical_features(keypoints_seq):
    if len(keypoints_seq) == 0:
        return {}
    
    kp = keypoints_seq.reshape(-1, 33, 3)
    
    features = {}
    hip_left = kp[:, 23, :2]
    hip_right = kp[:, 24, :2]
    shoulder_left = kp[:, 11, :2]
    shoulder_right = kp[:, 12, :2]
    
    hip_vector = hip_left - hip_right
    shoulder_vector = shoulder_left - shoulder_right
    
    hip_angle = np.arctan2(hip_vector[:,1], hip_vector[:,0]) * 180 / np.pi
    shoulder_angle = np.arctan2(shoulder_vector[:,1], shoulder_vector[:,0]) * 180 / np.pi
    features['x_factor'] = np.abs(shoulder_angle - hip_angle)
    
    elbow_left = kp[:, 13, :2]
    wrist_left = kp[:, 15, :2]
    arm_vector = wrist_left - elbow_left
    arm_angle = np.arctan2(arm_vector[:,1], arm_vector[:,0]) * 180 / np.pi
    features['lead_arm_angle'] = arm_angle
    
    wrist_right = kp[:, 16, :2]
    wrist_vel = np.linalg.norm(np.diff(wrist_right, axis=0), axis=1)
    features['wrist_velocity'] = np.concatenate([[0], wrist_vel])
    
    hip_center = (hip_left + hip_right) / 2
    features['avg_sway'] = np.std(hip_center[:,0])
    
    features['max_x_factor'] = np.max(features['x_factor'])
    features['max_wrist_vel'] = np.max(features['wrist_velocity'])
    
    return features

# ================== PRO REFERENCE (từ Cell 6, bạn có thể chỉnh) ==================
pro_reference = {
    'max_x_factor': 48.0,
    'max_wrist_vel': 0.18,
    'avg_sway': 0.015,
    'impact_ratio': 0.02,
    'backswing_ratio': 0.45,
}

# ================== HÀM SIMILARITY & FEEDBACK (từ Cell 6) ==================
def compute_bio_similarity(player_feat, player_phases, pro_ref):
    scores = {}
    diff_x = abs(player_feat['max_x_factor'] - pro_ref['max_x_factor'])
    scores['x_factor'] = max(0, 1 - diff_x / 60.0)
    
    scores['wrist_vel'] = min(1.0, player_feat['max_wrist_vel'] / pro_ref['max_wrist_vel'])
    
    scores['sway'] = max(0, 1 - player_feat['avg_sway'] / 0.08)
    
    phase_counts = np.bincount(player_phases, minlength=8)
    total = len(player_phases) if len(player_phases) > 0 else 1
    impact_ratio = phase_counts[5] / total
    scores['impact'] = max(0, 1 - abs(impact_ratio - pro_ref['impact_ratio']) / 0.05)
    
    backswing_frames = phase_counts[1] + phase_counts[2] + phase_counts[3]
    backswing_ratio = backswing_frames / total
    scores['backswing'] = max(0, 1 - abs(backswing_ratio - pro_ref['backswing_ratio']) / 0.3)
    
    weights = {'x_factor': 0.30, 'wrist_vel': 0.25, 'sway': 0.15, 'impact': 0.15, 'backswing': 0.15}
    overall_sim = sum(scores[k] * weights[k] for k in weights)
    return overall_sim, scores

def generate_detailed_feedback(player_feat, player_phases, sim_score, original_score):
    issues = []
    praises = []
    
    if player_feat['max_x_factor'] < 35:
        issues.append("Rotation thấp → thiếu power.")
    elif player_feat['max_x_factor'] > 60:
        issues.append("Rotation quá mức → dễ slice.")
    else:
        praises.append("Rotation ổn.")
    
    if player_feat['avg_sway'] > 0.04:
        issues.append("Hip sway cao → mất ổn định.")
    else:
        praises.append("Ổn định tốt.")
    
    if player_feat['max_wrist_vel'] < 0.10:
        issues.append("Tốc độ wrist thấp → thiếu acceleration.")
    else:
        praises.append("Tốc độ ổn.")
    
    phase_counts = np.bincount(player_phases, minlength=8)
    total = len(player_phases)
    impact_ratio = phase_counts[5] / total if total > 0 else 0
    if impact_ratio > 0.05:
        issues.append("Impact kéo → early release.")
    elif impact_ratio < 0.005:
        issues.append("Impact khó xác định.")
    
    backswing_ratio = (phase_counts[1] + phase_counts[2] + phase_counts[3]) / total
    if backswing_ratio < 0.35:
        issues.append("Backswing ngắn → transition vội.")
    
    overall = "Swing trung bình, tiềm năng cải thiện." if sim_score <= 0.70 else "Swing khá tốt!"
    feedback_text = f"{overall}\n\n"
    if praises:
        feedback_text += f"Điểm mạnh: {'; '.join(praises)}.\n"
    if issues:
        feedback_text += f"Cần cải thiện: {'; '.join(issues)}.\n"
    feedback_text += f"\nSimilarity AI: {sim_score:.3f}"
    
    return feedback_text

# ================== APP INTERFACE ==================
st.title("Golf Swing Analyzer - AI Feedback")
st.write("Upload video swing (.MOV hoặc .MP4) để nhận phân tích biomechanics.")

uploaded_file = st.file_uploader("Chọn video", type=["mov", "mp4"])

if uploaded_file:
    tfile = "temp_video.mov"
    with open(tfile, "wb") as f:
        f.write(uploaded_file.read())
    
    st.video(tfile)
    
    if st.button("Phân tích video"):
        with st.spinner("Đang xử lý..."):
            cap = cv2.VideoCapture(tfile)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            
            if len(frames) == 0:
                st.error("Không đọc được video.")
            else:
                keypoints_seq = extract_keypoints(frames)
                feat = extract_biomechanical_features(keypoints_seq)
                
                extra = np.stack([
                    feat['x_factor'], feat['lead_arm_angle'], feat['wrist_velocity'],
                    feat['x_factor']*0.5, feat['lead_arm_angle']*0.5
                ], axis=1)
                seq = np.concatenate([keypoints_seq, extra], axis=1)
                seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    phase_out, _ = model(seq_tensor)  # bỏ class_out vì app không dùng
                    phases_pred = torch.argmax(phase_out, dim=-1).squeeze(0).cpu().numpy()
                
                sim_score, detail_scores = compute_bio_similarity(feat, phases_pred, pro_reference)
                feedback = generate_detailed_feedback(feat, phases_pred, sim_score, "N/A")
                
                st.subheader("Kết quả")
                st.metric("Similarity với pro", f"{sim_score:.3f}")
                st.markdown(feedback)
                
                with st.expander("Chi tiết metrics"):
                    for k, v in detail_scores.items():
                        st.write(f"{k}: {v:.3f}")