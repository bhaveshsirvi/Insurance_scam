import gradio as gr
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image, ImageChops, ImageEnhance
from skimage.util import view_as_windows
from skimage.transform import rescale
from scipy.fftpack import fft2
import tempfile
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Constants
TARGET_SIZE = (256, 256)

# Feature Extraction Functions (keeping all your original functions)
def convert_to_ela_image(image_input, quality=90):
    """Convert image to ELA (Error Level Analysis) representation"""
    try:
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert('RGB')
        else:
            image = image_input.convert('RGB')
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_filename = temp_file.name
        
        image.save(temp_filename, 'JPEG', quality=quality)
        temp_image = Image.open(temp_filename)
        
        ela_image = ImageChops.difference(image, temp_image)
        
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff
        
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        ela_image = ela_image.resize(TARGET_SIZE)
        
        os.unlink(temp_filename)
        
        return np.array(ela_image.convert('L'))
    
    except Exception as e:
        print(f"Error in ELA computation: {e}")
        return np.zeros(TARGET_SIZE, dtype=np.uint8)

def compute_fft(image):
    """Compute FFT-based features"""
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        f = np.abs(fft2(gray))
        fshift = np.fft.fftshift(f)
        f_log = np.log1p(fshift)
        
        norm = cv2.normalize(f_log, None, 0, 255, cv2.NORM_MINMAX)
        return norm.astype(np.uint8)
    
    except Exception as e:
        print(f"Error in FFT computation: {e}")
        return np.zeros(TARGET_SIZE, dtype=np.uint8)

def compute_prnu(image):
    """Compute PRNU (Photo Response Non-Uniformity) features"""
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        gray = gray.astype(np.float32)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        residual = gray - blur
        
        norm = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX)
        return norm.astype(np.uint8)
    
    except Exception as e:
        print(f"Error in PRNU computation: {e}")
        return np.zeros(TARGET_SIZE, dtype=np.uint8)

def shuffle_patches(image, patch_size=2):
    """Shuffle image patches for MLEP computation"""
    np.random.seed(42)
    H, W = image.shape[:2]
    if len(image.shape) == 3:
        C = image.shape[2]
    else:
        C = 1
        image = image[:, :, np.newaxis]
    
    H = (H // patch_size) * patch_size
    W = (W // patch_size) * patch_size
    image = image[:H, :W]
    
    reshaped = image.reshape(H // patch_size, patch_size, W // patch_size, patch_size, C)
    patches = reshaped.transpose(0, 2, 1, 3, 4).reshape(-1, patch_size, patch_size, C)
    np.random.shuffle(patches)
    
    shuffled = patches.reshape(H // patch_size, W // patch_size, patch_size, patch_size, C)
    shuffled = shuffled.transpose(0, 2, 1, 3, 4).reshape(H, W, C)
    
    if C == 1:
        shuffled = shuffled[:, :, 0]
    
    return shuffled

def multi_scale_resample(image, scales=(1.0, 0.5, 0.25)):
    """Resample image at multiple scales"""
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
    
    resampled = []
    for s in scales:
        if s != 1.0:
            down = rescale(image, (s, s, 1), mode='reflect', anti_aliasing=True)
            up = cv2.resize((down * 255).astype(np.uint8), (image.shape[1], image.shape[0]))
            if len(up.shape) == 2:
                up = up[:, :, np.newaxis]
        else:
            up = image
        resampled.append(up)
    
    return np.concatenate(resampled, axis=2)

def compute_lep_vectorized(gray_img):
    """Compute Local Entropy Pattern"""
    if len(gray_img.shape) != 2:
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_RGB2GRAY)
    
    padded = np.pad(gray_img, ((0, 1), (0, 1)), mode='reflect')
    windows = view_as_windows(padded, (2, 2))
    reshaped = windows.reshape(-1, 4)
    
    entropies = np.zeros(reshaped.shape[0])
    for val in range(256):
        p = np.mean(reshaped == val, axis=1)
        mask = p > 0
        entropies[mask] += -p[mask] * np.log2(p[mask] + 1e-10)
    
    return entropies.reshape(windows.shape[0], windows.shape[1])

def compute_mlep(image, output_size=TARGET_SIZE):
    """Compute Multi-scale Local Entropy Pattern"""
    try:
        image = cv2.resize(image, (224, 224))
        shuffled = shuffle_patches(image, patch_size=2)
        multi_scale = multi_scale_resample(shuffled)
        
        lep_maps = []
        for c in range(multi_scale.shape[2]):
            gray = multi_scale[:, :, c]
            lep = compute_lep_vectorized(gray)
            lep_maps.append(lep)
        
        if len(lep_maps) > 0:
            stacked = np.stack(lep_maps, axis=-1)
            avg = np.mean(stacked, axis=2)
        else:
            avg = np.zeros((112, 112))
        
        avg_resized = cv2.resize(avg, output_size)
        norm = cv2.normalize(avg_resized, None, 0, 255, cv2.NORM_MINMAX)
        return norm.astype(np.uint8)
    
    except Exception as e:
        print(f"Error in MLEP computation: {e}")
        return np.zeros(TARGET_SIZE, dtype=np.uint8)

# Model Definition (keeping your original model)
class MultiStreamForensicsNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, fusion_method='concat'):
        super(MultiStreamForensicsNet, self).__init__()
        self.fusion_method = fusion_method

        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.ela_branch = models.resnet18(weights=weights)
        self.ela_branch.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.fft_branch = models.resnet18(weights=weights)
        self.fft_branch.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.prnu_branch = models.resnet18(weights=weights)
        self.prnu_branch.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.mlep_branch = models.resnet18(weights=weights)
        self.mlep_branch.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        for branch in [self.ela_branch, self.fft_branch, self.prnu_branch, self.mlep_branch]:
            branch.fc = nn.Identity()

        if pretrained:
            with torch.no_grad():
                pretrained_weight = models.resnet18(weights=weights).conv1.weight
                avg_weight = pretrained_weight.mean(dim=1, keepdim=True)

                for branch in [self.ela_branch, self.fft_branch, self.prnu_branch, self.mlep_branch]:
                    branch.conv1.weight.copy_(avg_weight)

        feature_dim = 512
        if fusion_method == 'concat':
            self.fusion_fc = nn.Linear(feature_dim * 4, 512)
        elif fusion_method == 'add':
            self.fusion_fc = nn.Linear(feature_dim, 512)
        elif fusion_method == 'attention':
            self.attention_weights = nn.Linear(feature_dim * 4, 4)
            self.fusion_fc = nn.Linear(feature_dim, 512)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        ela_input = x[:, 0:1, :, :]
        fft_input = x[:, 1:2, :, :]
        prnu_input = x[:, 2:3, :, :]
        mlep_input = x[:, 3:4, :, :]

        ela_features = self.ela_branch(ela_input)
        fft_features = self.fft_branch(fft_input)
        prnu_features = self.prnu_branch(prnu_input)
        mlep_features = self.mlep_branch(mlep_input)

        if self.fusion_method == 'concat':
            fused_features = torch.cat([ela_features, fft_features, prnu_features, mlep_features], dim=1)
            fused_features = self.fusion_fc(fused_features)
        elif self.fusion_method == 'add':
            fused_features = ela_features + fft_features + prnu_features + mlep_features
            fused_features = self.fusion_fc(fused_features)
        elif self.fusion_method == 'attention':
            all_features = torch.cat([ela_features, fft_features, prnu_features, mlep_features], dim=1)
            attention_scores = F.softmax(self.attention_weights(all_features), dim=1)
            weighted_ela = attention_scores[:, 0:1] * ela_features
            weighted_fft = attention_scores[:, 1:2] * fft_features
            weighted_prnu = attention_scores[:, 2:3] * prnu_features
            weighted_mlep = attention_scores[:, 3:4] * mlep_features
            fused_features = weighted_ela + weighted_fft + weighted_prnu + weighted_mlep
            fused_features = self.fusion_fc(fused_features)

        output = self.classifier(fused_features)
        return output

# Inference Functions
def extract_forensics_features(image_input):
    """Extract all 4 forensics features from an image"""
    try:
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Could not load image from {image_input}")
        else:
            # Handle PIL Image from Gradio
            if hasattr(image_input, 'convert'):
                image_input = image_input.convert('RGB')
                image = np.array(image_input)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image = image_input
        
        image = cv2.resize(image, TARGET_SIZE)
        
        ela = convert_to_ela_image(image)
        fft_map = compute_fft(image)
        prnu = compute_prnu(image)
        mlep = compute_mlep(image)
        
        features = np.stack([ela, fft_map, prnu, mlep], axis=0).astype(np.float32) / 255.0
        
        return torch.tensor(features).unsqueeze(0)
    
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def load_model():
    """Load the model - you'll need to upload your model file to the space"""
    try:
        # Try to load from Hugging Face Hub or local file
        model_path = 'multi-input-cnn-model.pth'  # Make sure to upload this file
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found. Please upload it to your space.")
        
        state_dict = torch.load(model_path, map_location=DEVICE)
        
        # Detect fusion method
        if 'attention_weights.weight' in state_dict:
            fusion_method = 'attention'
        elif 'fusion_fc.weight' in state_dict:
            fusion_fc_shape = state_dict['fusion_fc.weight'].shape
            if fusion_fc_shape[1] == 2048:
                fusion_method = 'concat'
            elif fusion_fc_shape[1] == 512:
                fusion_method = 'add'
            else:
                fusion_method = 'concat'
        else:
            fusion_method = 'concat'
        
        model = MultiStreamForensicsNet(num_classes=2, pretrained=False, fusion_method=fusion_method)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def create_feature_visualization(image_input):
    """Create visualization of forensics features"""
    try:
        # Convert PIL to numpy array
        if hasattr(image_input, 'convert'):
            image_input = image_input.convert('RGB')
            original = np.array(image_input)
        else:
            original = image_input
        
        original = cv2.resize(original, TARGET_SIZE)
        
        # Extract features
        ela = convert_to_ela_image(original)
        fft_map = compute_fft(original)
        prnu = compute_prnu(original)
        mlep = compute_mlep(original)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Original Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(ela, cmap='hot')
        axes[0, 1].set_title('ELA (Error Level Analysis)', fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(fft_map, cmap='viridis')
        axes[0, 2].set_title('FFT Features', fontweight='bold')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(prnu, cmap='plasma')
        axes[1, 0].set_title('PRNU (Noise Pattern)', fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(mlep, cmap='inferno')
        axes[1, 1].set_title('MLEP (Multi-scale Local Entropy)', fontweight='bold')
        axes[1, 1].axis('off')
        
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Error in visualization: {e}")
        return None

# Load model at startup
print("Loading model...")
model = load_model()

def predict_image(image):
    np.random.seed(42)
    """Main prediction function for Gradio"""
    if model is None:
        return "Model not loaded. Please check if the model file exists.", None, None
    
    if image is None:
        return "Please upload an image.", None, None
    
    try:
        # Extract features
        features = extract_forensics_features(image)
        
        if features is None:
            return "Could not extract features from the image.", None, None
        
        # Make prediction
        with torch.no_grad():
            features = features.to(DEVICE)
            outputs = model(features)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        class_names = ['AI Generated/Fake', 'Real/Authentic']
        prediction_text = f"""
        **Prediction**: {class_names[predicted_class]}
        **Confidence**: {confidence:.3f} ({confidence*100:.1f}%)
        
        **Detailed Probabilities**:
        - AI Generated/Fake: {probabilities[0][0].item():.3f} ({probabilities[0][0].item()*100:.1f}%)
        - Real/Authentic: {probabilities[0][1].item():.3f} ({probabilities[0][1].item()*100:.1f}%)
        """
        
        # Create feature visualization
        viz_fig = create_feature_visualization(image)
        
        # Create probability chart
        prob_fig, ax = plt.subplots(figsize=(8, 6))
        classes = class_names
        probs = [probabilities[0][0].item(), probabilities[0][1].item()]
        colors = ['#ff6b6b', '#4ecdc4']
        
        bars = ax.bar(classes, probs, color=colors, alpha=0.7)
        ax.set_ylabel('Probability')
        ax.set_title('Prediction Probabilities')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        return prediction_text, viz_fig, prob_fig
        
    except Exception as e:
        return f"Error during prediction: {str(e)}", None, None

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="AI vs Real Image Detector", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🔍 AI vs Real Image Detector
        
        This tool uses advanced forensics analysis to detect whether an image is AI-generated or real.
        It analyzes multiple forensic features including:
        - **ELA (Error Level Analysis)**: Detects compression artifacts
        - **FFT Features**: Analyzes frequency domain patterns
        - **PRNU (Photo Response Non-Uniformity)**: Examines sensor noise patterns  
        - **MLEP (Multi-scale Local Entropy Pattern)**: Studies local texture patterns
        
        Upload an image to get started!
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload Image",
                    height=400
                )
                
                predict_btn = gr.Button(
                    "🔍 Analyze Image", 
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                prediction_output = gr.Markdown(
                    label="Prediction Results",
                    value="Upload an image and click 'Analyze Image' to see results."
                )
        
        with gr.Row():
            with gr.Column():
                feature_plot = gr.Plot(
                    label="Forensics Features Visualization",
                    show_label=True
                )
            
            with gr.Column():
                probability_plot = gr.Plot(
                    label="Prediction Probabilities",
                    show_label=True
                )
        
        # Event handlers
        predict_btn.click(
            fn=predict_image,
            inputs=[image_input],
            outputs=[prediction_output, feature_plot, probability_plot]
        )
        
        # Example section
        gr.Markdown("""
        ## ℹ️ How it works:
        
        1. **Upload** an image (JPG, PNG, etc.)
        2. **Click** the "Analyze Image" button
        3. **View** the prediction results and confidence score
        4. **Examine** the forensics features visualization to understand the analysis
        
        **Note**: This model analyzes technical artifacts and patterns that may indicate AI generation. 
        Results should be interpreted as indicators rather than definitive proof.
        """)
    
    return demo

# Launch the app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False  # Set to True if you want a public link
    )
