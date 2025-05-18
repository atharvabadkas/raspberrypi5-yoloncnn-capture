import os
import sys
import argparse
import subprocess
from pathlib import Path

def check_requirements():
    try:
        # Check for ultralytics (YOLO) package
        try:
            import ultralytics
            print(f"✅ Ultralytics package installed: version {ultralytics.__version__}")
        except ImportError:
            print("❌ Ultralytics package not found. Please install it with:")
            print("   pip install ultralytics")
            return False
        
        # Check for onnx2ncnn executable
        import shutil
        onnx2ncnn_path = shutil.which('onnx2ncnn')
        if not onnx2ncnn_path:
            print("❌ onnx2ncnn executable not found in PATH")
            print("   Please ensure NCNN is properly installed and onnx2ncnn is in your PATH")
            return False
        else:
            print(f"✅ onnx2ncnn found at: {onnx2ncnn_path}")
            
        return True
    
    except Exception as e:
        print(f"❌ Error checking requirements: {e}")
        return False

def convert_to_onnx(model_path, output_dir):
    try:
        from ultralytics import YOLO
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the model
        print(f"Loading YOLOv8n model from {model_path}...")
        model = YOLO(model_path)
        
        # Export to ONNX format with correct parameters
        print(f"Converting {model_path} to ONNX format...")
        
        # Export with simplify but without dynamic or half
        onnx_path = model.export(format="onnx", simplify=True, dynamic=False, half=False)
        
        # Verify the ONNX model was created
        if onnx_path and os.path.exists(onnx_path):
            print(f"✅ Model successfully exported to ONNX: {onnx_path}")
            return onnx_path
        else:
            print(f"❌ Failed to convert model to ONNX")
            return None
                
    except Exception as e:
        print(f"❌ Error converting to ONNX: {e}")
        return None

def convert_onnx_to_ncnn(onnx_path, output_dir):
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename without extension
        base_name = Path(onnx_path).stem
        
        # Define output paths
        param_path = Path(output_dir) / f"{base_name}.param"
        bin_path = Path(output_dir) / f"{base_name}.bin"
        
        # Run onnx2ncnn command
        print(f"Converting {onnx_path} to NCNN format...")
        cmd = [
            "onnx2ncnn",
            str(onnx_path),
            str(param_path),
            str(bin_path)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ onnx2ncnn failed: {result.stderr}")
            if result.stdout:
                print(f"Command output: {result.stdout}")
            return None
            
        # Verify the files were created
        if param_path.exists() and bin_path.exists():
            print(f"✅ NCNN model created: {param_path} and {bin_path}")
            return (param_path, bin_path)
        else:
            print("❌ NCNN conversion failed - output files not found")
            return None
        
    except Exception as e:
        print(f"❌ Error converting to NCNN: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Convert YOLOv8n model to NCNN format")
    parser.add_argument("--model", type=str, default="models/yolov8n.pt", 
                        help="Path to the YOLOv8n model file")
    parser.add_argument("--output-dir", type=str, default="models", 
                        help="Directory to save the converted model")
    parser.add_argument("--format", type=str, default="ncnn", 
                        choices=["ncnn", "onnx"], 
                        help="Format to convert to")
    parser.add_argument("--use-existing-onnx", action="store_true",
                        help="Use existing ONNX file if available")
    
    args = parser.parse_args()
    
    # Check requirements first
    if not check_requirements():
        print("❌ Missing required dependencies. Please install them and try again.")
        return 1
    
    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        print("Please specify a valid model path or run download_model.py first to download the model.")
        return 1
    
    print(f"Converting model: {model_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target format: {args.format}")
    
    # For ONNX format, do a direct conversion
    if args.format == "onnx":
        onnx_path = convert_to_onnx(model_path, args.output_dir)
        if not onnx_path:
            print("❌ ONNX conversion failed.")
            return 1
        
        print("\n==== Conversion Summary ====")
        print(f"✅ Original model: {model_path}")
        print(f"✅ ONNX model: {onnx_path}")
        return 0
        
    # For NCNN format, first convert to ONNX, then to NCNN
    if args.format == "ncnn":
        # Check for existing ONNX file
        onnx_path = Path(args.output_dir) / f"{model_path.stem}.onnx"
        
        if args.use_existing_onnx and onnx_path.exists():
            print(f"Using existing ONNX file: {onnx_path}")
        else:
            # Convert to ONNX first
            onnx_path = convert_to_onnx(model_path, args.output_dir)
            if not onnx_path:
                print("❌ ONNX conversion failed, cannot proceed to NCNN conversion.")
                return 1
        
        # Then convert ONNX to NCNN
        ncnn_paths = convert_onnx_to_ncnn(onnx_path, args.output_dir)
        if not ncnn_paths:
            print("❌ NCNN conversion failed.")
            return 1
            
        param_path, bin_path = ncnn_paths
        
        print("\n==== Conversion Summary ====")
        print(f"✅ Original model: {model_path}")
        print(f"✅ ONNX model: {onnx_path}")
        print(f"✅ NCNN model: {param_path} and {bin_path}")
        print("\nThe model has been successfully converted to NCNN format.")
        print("You can now use these files with your application for better performance.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())