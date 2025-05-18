#!/usr/bin/env python3
import re
import os

def update_model_file():
    # Check if model.py exists
    if not os.path.exists('model.py'):
        print("❌ model.py not found. Please run in the correct directory.")
        return False

    with open('model.py', 'r') as f:
        content = f.read()
    
    # Make a backup
    with open('model.py.backup', 'w') as f:
        f.write(content)
    
    # Add better error handling for NCNN inference
    improved_ncnn_error_handling = """
            # If inference fails, try to fall back to PyTorch
            print(f"Error in NCNN inference: {e}")
            if hasattr(self, 'model_fallback') and self.model_fallback:
                print("Falling back to PyTorch model for this frame")
                try:
                    return self.model_fallback(frame)
                except Exception as e2:
                    print(f"PyTorch fallback also failed: {e2}")
            return []
    """
    
    # Find the NCNN inference method's exception handler and improve it
    pattern = r"(    except Exception as e:\s+print\(f\"Error in NCNN inference: \{e\}\"\).*?return \[\])"
    content = re.sub(pattern, 
                    "    except Exception as e:" + improved_ncnn_error_handling, 
                    content, 
                    flags=re.DOTALL)
    
    # Add capability to store a fallback PyTorch model
    load_model_update = """
        # Store PyTorch model as fallback
        if os.path.exists(self.model_path) and not self.using_ncnn:
            self.model_fallback = self.model
        else:
            self.model_fallback = None
    """
    
    # Add the fallback storage after PyTorch model loading
    pattern = r"(            print\(\"PyTorch model loaded successfully\"\)\s+self\.using_ncnn = False\s+return True)"
    content = re.sub(pattern, r"\1\n" + load_model_update, content)
    
    # Write the updated content back
    with open('model.py', 'w') as f:
        f.write(content)
    
    print("✅ model.py updated with improved NCNN error handling and fallback mechanism")
    return True

if __name__ == "__main__":
    update_model_file() 