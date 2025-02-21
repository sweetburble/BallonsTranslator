import torch
import paddle
import paddleocr

def check_versions_and_gpu():
    # Check PyTorch version
    pytorch_version = torch.__version__
    print(f"PyTorch version: {pytorch_version}")

    # Check if GPU is available for PyTorch
    pytorch_gpu_available = torch.cuda.is_available()
    print(f"PyTorch GPU available: {pytorch_gpu_available}")

    # Check Paddle version
    paddle_version = paddle.__version__
    print(f"Paddle version: {paddle_version}")

    # Check PaddleOCR version
    paddleocr_version = paddleocr.__version__
    print(f"PaddleOCR version: {paddleocr_version}")

if __name__ == "__main__":
    check_versions_and_gpu()