from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import torch
import torch.nn as nn

app = FastAPI()

# Define improved custom CNN architecture
class CustomCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(CustomCNN, self).__init__()
        
        # Define the layers of the custom CNN
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(1024)
        self.conv6 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(2048)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(2048 * 3 * 3, 4096)  # Adjusted input size
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, num_classes)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = self.pool(torch.relu(self.bn5(self.conv5(x))))
        x = self.pool(torch.relu(self.bn6(self.conv6(x))))
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x

# Load the model
model = CustomCNN(num_classes=4)

# Define the manual transformation
def transform_image(image_bytes):
    # Load the image
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('RGB')
    img = img.resize((224, 224))
    
    # Convert the image to a tensor
    img_tensor = torch.tensor(np.array(img)).float()
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # Change shape from HWC to CHW format and add a batch dimension
    img_tensor = img_tensor.to(device)  # Move tensor to the same device as the model
    return img_tensor

# Load your model here
model.load_state_dict(torch.load('plant_disease_model.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()
class_names = {0: "Bacterial Leaf Disease", 1: "Dried Leaf", 2: "Fungal Brown Spot Disease", 3: "Healthy Leaf"}

# Endpoint to handle image uploads and predictions
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    tensor = transform_image(image_data)
    
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class_name = class_names[int(predicted.item())]
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()
    
    return JSONResponse(content={"predicted_class": predicted_class_name, "confidence": confidence})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8080)
