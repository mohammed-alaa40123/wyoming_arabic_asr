# Wyoming Arabic ASR Add-on

This add-on provides Arabic speech recognition for Home Assistant using a FastConformer CTC model.

## Installation

1. Copy your model files to `/config/share/arabic_asr/`:
   - `arabic_ctc_model.onnx` - The ONNX model file
   - `tokens.txt` - The vocabulary file

2. Install and start the add-on

3. Add to your `configuration.yaml`:
   ```yaml
   wyoming:
     - uri: tcp://localhost:10305
       protocol: wyoming  
       name: "Arabic ASR"
       language: "ar"
   ```

## Configuration

- **uri**: Wyoming protocol URI (default: tcp://0.0.0.0:10305)
- **device**: Device for inference (cpu/cuda)
- **debug**: Enable debug logging
- **model_path**: Path to ONNX model file
- **tokens_path**: Path to tokens file

## Model Files

You need to place these files in `/config/share/arabic_asr/`:
- `arabic_ctc_model.onnx` - Your exported ONNX model
- `tokens.txt` - Vocabulary/tokens file

The add-on expects these files to be in the Home Assistant shared directory.