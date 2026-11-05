import io
import torch
import numpy as np
from PIL import Image
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64

from models.full_model import BR_UA_MDASGTNet
from preprocessing.pfwf_filter import ProbabilisticWienerFilter

app = FastAPI(title="Liver Lesion Segmentation & Epistemic Uncertainty Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BR_UA_MDASGTNet(in_channels=1, num_classes=1)
checkpoint_path = "checkpoints/best_model.pth"

if torch.cuda.is_available() and os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()

denoiser = ProbabilisticWienerFilter()

def array_to_base64_png(arr: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", arr)
    if not success:
        return ""
    return base64.b64encode(encoded.tobytes()).decode("utf-8")

@app.post("/analyze_slice")
async def analyze_slice(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("L")
    img_np = np.array(image.resize((224, 224)), dtype=np.float32) / 255.0

    # Step 1: Preprocess with PFWF
    denoised_np = denoiser(img_np)

    # Step 2: Inference with Monte Carlo Uncertainty (T=15 passes)
    tensor_in = torch.from_numpy(denoised_np).unsqueeze(0).unsqueeze(0).to(DEVICE)
    mean_prob, entropy = model.predict_with_uncertainty(tensor_in, num_samples=15)

    prob_map = mean_prob.squeeze().cpu().numpy()
    entropy_map = entropy.squeeze().cpu().numpy()

    # Binary mask
    binary_mask = (prob_map > 0.5).astype(np.uint8) * 255

    # Normalize entropy to heatmap [0, 255]
    norm_entropy = (entropy_map / (entropy_map.max() + 1e-8) * 255).astype(np.uint8)
    entropy_color = cv2.applyColorMap(norm_entropy, cv2.COLORMAP_JET)

    return JSONResponse({
        "raw_image": array_to_base64_png((img_np * 255).astype(np.uint8)),
        "preprocessed_image": array_to_base64_png((denoised_np * 255).astype(np.uint8)),
        "predicted_mask": array_to_base64_png(binary_mask),
        "uncertainty_heatmap": array_to_base64_png(entropy_color),
        "mean_confidence": float(np.mean(prob_map[prob_map > 0.5])) if np.any(prob_map > 0.5) else 0.0
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)