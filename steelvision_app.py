# steelvision_app.py
import os
import tempfile
import base64
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from fpdf import FPDF
import plotly.graph_objects as go

from models.basic_cnn import BasicSteelCNN
from train_advanced_cnn import GradCAM


class SteelVisionApp:
    def __init__(self):
        self.setup_page()
        self.initialize_session_state()
        self.load_model()

    def setup_page(self):
        st.set_page_config(
            page_title="SteelVision - Crack Detection",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.markdown(
            """
            <style>
            .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
            .sub-header { font-size: 1.2rem; color: #ff7f0e; margin-bottom: 0.5rem; }
            .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 10px; border-left: 5px solid #1f77b4; }
            .crack-detected { background-color: #ffcccc; padding: 1rem; border-radius: 10px; border-left: 5px solid #ff0000; }
            .no-crack { background-color: #ccffcc; padding: 1rem; border-radius: 10px; border-left: 5px solid #00aa00; }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<h1 class="main-header">SteelVision</h1>', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Advanced Steel Surface Crack Detection System</h3>', unsafe_allow_html=True)

    def initialize_session_state(self):
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'device' not in st.session_state:
            st.session_state.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'gradcam_layer' not in st.session_state:
            st.session_state.gradcam_layer = None
        if 'history' not in st.session_state:
            st.session_state.history = []

    def load_model(self):
        if st.session_state.model_loaded:
            return

        device = st.session_state.device
        advanced_checkpoint = 'checkpoints/best_advanced_cnn.pth'
        basic_checkpoint = 'checkpoints/best_basic_cnn.pth'

        model = BasicSteelCNN(num_classes=2)

        try:
            if os.path.exists(advanced_checkpoint):
                checkpoint = torch.load(advanced_checkpoint, map_location=device)
                model_type = "Advanced"
            elif os.path.exists(basic_checkpoint):
                checkpoint = torch.load(basic_checkpoint, map_location=device)
                model_type = "Basic"
            else:
                st.sidebar.warning("No trained model found. Please train a model first.")
                st.sidebar.info("Run: python train_basic_cnn.py")
                return

            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()

            st.session_state.model = model
            st.session_state.model_loaded = True
            st.session_state.gradcam_layer = model.conv4

            if 'val_acc' in checkpoint:
                st.sidebar.info(f"Model Accuracy: {checkpoint['val_acc']:.2f}%")
            if 'val_f1' in checkpoint:
                st.sidebar.info(f"Model F1 Score: {checkpoint['val_f1']:.4f}")
            st.sidebar.success(f"{model_type} model loaded")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.info("Please train a model first using: python train_basic_cnn.py")

    def get_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image):
        transform = self.get_transform()
        return transform(image).unsqueeze(0)

    def apply_clahe(self, image):
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)

    def denoise_image(self, image):
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    def sharpen_image(self, image):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)

    def estimate_crack_diameter(self, heatmap, original_shape):
        _, binary_heatmap = cv2.threshold((heatmap * 255).astype(np.uint8), 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, None, None

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        diameter_pixels = (w + h) / 2

        scale_x = original_shape[1] / heatmap.shape[1]
        scale_y = original_shape[0] / heatmap.shape[0]
        diameter_original = diameter_pixels * ((scale_x + scale_y) / 2)

        # Demonstration conversion: 100 pixels = 1 mm
        diameter_mm = diameter_original * 0.01
        return diameter_mm, diameter_original, (x, y, w, h)

    def generate_pdf_report(self, original_image_pil, overlay_image_pil, prediction, probability, diameter_mm, severity):
        pdf = FPDF()
        pdf.add_page()

        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'SteelVision - Crack Detection Report', 0, 1, 'C')
        pdf.ln(6)

        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 8, f'Report generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1)
        pdf.ln(4)

        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 8, 'Detection Results:', 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 8, f'Prediction: {"CRACK DETECTED" if prediction == 1 else "NO CRACK"}', 0, 1)
        pdf.cell(0, 8, f'Confidence: {probability:.2%}', 0, 1)
        pdf.cell(0, 8, f'Severity: {severity}', 0, 1)
        if prediction == 1:
            pdf.cell(0, 8, f'Estimated Crack Diameter: {diameter_mm:.2f} mm', 0, 1)
        pdf.ln(6)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp1:
            original_image_pil.save(tmp1.name)
            original_path = tmp1.name

        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp2:
            overlay_image_pil.save(tmp2.name)
            overlay_path = tmp2.name

        pdf.cell(0, 8, 'Original Image:', 0, 1)
        pdf.image(original_path, x=10, y=pdf.get_y(), w=80)
        pdf.ln(85)

        pdf.cell(0, 8, 'Analysis Overlay:', 0, 1)
        pdf.image(overlay_path, x=10, y=pdf.get_y(), w=80)

        os.unlink(original_path)
        os.unlink(overlay_path)

        return pdf

    def run(self):
        st.sidebar.title("Configuration")

        image_source = st.sidebar.radio("Select Image Source:", ["Upload Image", "Camera Capture", "Sample Image"])

        st.sidebar.subheader("Image Enhancement")
        enhance_clahe = st.sidebar.checkbox("CLAHE Enhancement", value=True)
        enhance_denoise = st.sidebar.checkbox("Denoise Image", value=False)
        enhance_sharpen = st.sidebar.checkbox("Sharpen Image", value=False)

        st.sidebar.subheader("Analysis Options")
        show_heatmap = st.sidebar.checkbox("Show Heatmap", value=True)
        show_overlay = st.sidebar.checkbox("Show Overlay", value=True)
        estimate_diameter = st.sidebar.checkbox("Estimate Crack Diameter", value=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Input Image")

            image = None
            # Ensure these variables are always defined (avoids "possibly unbound" errors
            # when later referenced in the results panel even if no image is selected).
            predicted_class = 0
            crack_probability = 0.0
            severity = "Unknown"
            diameter_mm = 0.0
            heatmap_colored = None
            overlay = None
            cv_image = None

            if image_source == "Upload Image":
                uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg', 'bmp'])
                if uploaded_file is not None:
                    image = Image.open(uploaded_file).convert('RGB')

            elif image_source == "Camera Capture":
                camera_image = st.camera_input("Take a picture")
                if camera_image is not None:
                    image = Image.open(camera_image).convert('RGB')

            else:  # Sample Image
                if st.button("Generate Sample Crack Image"):
                    sample_img = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
                    cv2.line(sample_img, (50, 50), (150, 150), (0, 0, 0), 3)
                    cv2.line(sample_img, (150, 150), (180, 100), (0, 0, 0), 2)
                    image = Image.fromarray(sample_img)
                if st.button("Generate Sample No-Crack Image"):
                    sample_img = np.random.randint(150, 230, (224, 224, 3), dtype=np.uint8)
                    image = Image.fromarray(sample_img)

            if image is not None:
                st.image(image, caption="Original Image", use_column_width=True)

                cv_image = np.array(image)
                original_shape = cv_image.shape

                enhanced_image = cv_image.copy()
                enhancement_steps = []

                if enhance_clahe:
                    enhanced_image = self.apply_clahe(enhanced_image)
                    enhancement_steps.append("CLAHE")
                if enhance_denoise:
                    enhanced_image = self.denoise_image(enhanced_image)
                    enhancement_steps.append("Denoise")
                if enhance_sharpen:
                    enhanced_image = self.sharpen_image(enhanced_image)
                    enhancement_steps.append("Sharpen")
                if enhancement_steps:
                    st.image(enhanced_image, caption=f"Enhanced Image ({', '.join(enhancement_steps)})", use_column_width=True)

                # Re-initialize with enhanced image overlay
                overlay = Image.fromarray(enhanced_image)

                # Initialize variables before conditional block
                predicted_class = 0
                crack_probability = 0.0
                severity = "Unknown"
                diameter_mm = 0.0
                heatmap_colored = None

                if st.session_state.model_loaded:
                    with st.spinner("Analyzing image..."):
                        pil_image = Image.fromarray(enhanced_image)
                        input_tensor = self.preprocess_image(pil_image).to(st.session_state.device)

                        with torch.no_grad():
                            outputs = st.session_state.model(input_tensor)
                            probabilities = F.softmax(outputs, dim=1)
                            predicted_class = int(torch.argmax(outputs, dim=1).item())
                            crack_probability = float(probabilities[0, 1].item())

                        if st.session_state.gradcam_layer:
                            gradcam = GradCAM(st.session_state.model, st.session_state.gradcam_layer)
                            heatmap, _ = gradcam.generate_heatmap(input_tensor[0], target_class=predicted_class)

                            heatmap_resized = cv2.resize(heatmap, (enhanced_image.shape[1], enhanced_image.shape[0]))
                            heatmap_uint8 = np.uint8(255 * heatmap_resized)
                            heatmap_uint8 = np.ascontiguousarray(heatmap_uint8, dtype=np.uint8)
                            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                            overlay_np = cv2.addWeighted(enhanced_image, 0.7, heatmap_colored, 0.3, 0)

                            if predicted_class == 1 and estimate_diameter:
                                diameter_mm, diameter_pixels, bbox = self.estimate_crack_diameter(heatmap_resized, original_shape)
                                if bbox:
                                    x, y, w, h = bbox
                                    cv2.rectangle(overlay_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                    cv2.putText(overlay_np, f"Diameter: {diameter_mm:.2f}mm", (x, max(0, y - 10)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                            overlay = Image.fromarray(overlay_np)

                            mean_activation = float(np.mean(heatmap_resized))
                            if mean_activation < 0.3:
                                severity = "Low"
                            elif mean_activation < 0.6:
                                severity = "Medium"
                            else:
                                severity = "High"
                else:
                    st.warning("Model not loaded. Please train a model first: python train_basic_cnn.py")

                # Save last analysis to history
                st.session_state.history.append({
                    'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'prediction': 'crack' if predicted_class == 1 else 'no_crack',
                    'confidence': crack_probability
                })

        with col2:
            st.subheader("Analysis Results")

            if image is None:
                st.info("No image selected.")
            else:
                # Display detection card
                if predicted_class == 1:
                    st.markdown(f"""
                        <div class="crack-detected">
                            <h3>CRACK DETECTED</h3>
                            <p>Confidence: <b>{crack_probability:.2%}</b></p>
                            <p>Severity: <b>{severity}</b></p>
                            {f'<p>Estimated Diameter: <b>{diameter_mm:.2f} mm</b></p>' if diameter_mm > 0 else ''}
                        </div>
                        """, unsafe_allow_html=True)
                    st.error("Consult a structural engineer for further inspection.")
                else:
                    st.markdown(f"""
                        <div class="no-crack">
                            <h3>NO CRACK DETECTED</h3>
                            <p>Confidence: <b>{(1 - crack_probability):.2%}</b></p>
                            <p>Surface condition: <b>Good</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                    st.success("Surface appears to be in good condition.")

                # Probability bar
                st.subheader("Probability Distribution")
                fig_prob = go.Figure(data=[go.Bar(
                    x=['No Crack', 'Crack'],
                    y=[1 - crack_probability, crack_probability]
                )])
                fig_prob.update_layout(title="Classification Probabilities", yaxis_title="Probability", yaxis_range=[0, 1])
                st.plotly_chart(fig_prob, use_container_width=True)

                # Confidence gauge
                st.subheader("Confidence Gauge")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=crack_probability * 100,
                    title={'text': "Crack Confidence (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "red" if predicted_class == 1 else "green"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgray"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "orange"}
                        ]
                    }
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Visualizations
                if show_heatmap and 'heatmap_colored' in locals() and heatmap_colored is not None:
                    st.subheader("Grad-CAM Heatmap")
                    st.image(heatmap_colored, caption="Grad-CAM Heatmap", use_column_width=True)

                if show_overlay and overlay is not None:
                    st.subheader("Overlay Visualization")
                    st.image(overlay, caption="Heatmap Overlay", use_column_width=True)

                # Report generation
                st.subheader("Report Generation")
                if st.button("Generate PDF Report"):
                    # Safely construct PIL images: avoid passing None into Image.fromarray
                    if isinstance(image, Image.Image):
                        orig_pil = image
                    elif cv_image is not None:
                        orig_pil = Image.fromarray(cv_image)
                    else:
                        # fallback blank image when neither image nor cv_image are available
                        orig_pil = Image.new("RGB", (224, 224), (255, 255, 255))

                    if isinstance(overlay, Image.Image):
                        overlay_pil = overlay
                    elif overlay is not None and isinstance(overlay, np.ndarray):
                        # Convert BGR (OpenCV) to RGB if necessary
                        if overlay.ndim == 3 and overlay.shape[2] == 3:
                            overlay_conv = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                        else:
                            overlay_conv = overlay
                        overlay_pil = Image.fromarray(overlay_conv)
                    else:
                        # fallback to original image for overlay if none available
                        overlay_pil = orig_pil.copy()
                    pdf = self.generate_pdf_report(orig_pil, overlay_pil, predicted_class, crack_probability, diameter_mm, severity)
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                        pdf.output(tmp.name)
                        with open(tmp.name, 'rb') as f:
                            pdf_bytes = f.read()
                        os.unlink(tmp.name)

                    b64_pdf = base64.b64encode(pdf_bytes).decode()
                    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="steelvision_report.pdf">Download PDF Report</a>'
                    st.markdown(href, unsafe_allow_html=True)

        # Sidebar: history and controls
        st.sidebar.subheader("Analysis History")
        if st.sidebar.button("Clear History"):
            st.session_state.history = []

        for entry in reversed(st.session_state.history[-10:]):
            st.sidebar.write(f"{entry['time']} — {entry['prediction']} ({entry['confidence']:.2%})")

        st.sidebar.subheader("Real-time Monitoring")
        if st.sidebar.button("Start Live Camera"):
            st.warning("Live camera feature not implemented in this version.")

        st.sidebar.subheader("Advanced Options")
        _ = st.sidebar.checkbox("Ensemble Mode", value=False)
        _ = st.sidebar.checkbox("Test-Time Augmentation", value=False)


def main():
    app = SteelVisionApp()
    app.run()


if __name__ == "__main__":
    main()
