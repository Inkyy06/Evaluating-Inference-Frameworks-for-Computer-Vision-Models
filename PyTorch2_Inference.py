import torch
import time
from torchvision.models import resnet50, ResNet50_Weights


def synchronize(device):
    # Wait for pending CUDA work; do nothing on CPU and other devices.
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# Choose a device
device = torch.device("cpu")
print("Device: ", device)

batch_size = 1
warmup_runs = 10
measurement_runs = 50

# Load resnet50 onto the selected device using float32 with pretrained(IMAGENET1K_V2) weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2
                 ).to(device = device, dtype = torch.float32)

# Switch the model to evaluation mode
model.eval()

# Set seed for fixed input
torch.manual_seed(31)

# Input for model
x = torch.randn(batch_size, 3, 224, 224,
                device = device,
                dtype = torch.float32)

# Compile the model
compiled_model = torch.compile(model)



with torch.inference_mode():

    # Warmup Runs
    for i in range(warmup_runs):

        output = compiled_model(x)

    #synchronize the processes for cuda
    synchronize(device)

    # Store the starting time
    start_time = time.perf_counter()

    # Measurement runs
    for i in range(measurement_runs):

        output = compiled_model(x)

    #synchronize the processes for cuda
    synchronize(device)

    # Calculate total measurement time
    total_time_seconds = time.perf_counter() - start_time



# Average time for one batch in miliseconds
average_batch_time_ms = (total_time_seconds / measurement_runs) * 1000

# Number of images processed per second
throughput = (measurement_runs * batch_size) / total_time_seconds

print("Output Size: ", output.shape)
print(f"Compiled Average Inference Time: {average_batch_time_ms:.2f} ms")
print(f"Throughput: {throughput:.2f} image/s")