# Wyoming Arabic ASR for Home Assistant

[![Open your Home Assistant instance and show the add add-on repository dialog with a specific repository URL pre-filled.](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A//github.com/mohammed-alaa40123/wyoming-arabic-asr)

This repository contains Home Assistant add-ons for Arabic Speech Recognition using Wyoming protocol.

## Add-ons

### Wyoming Arabic ASR

Arabic speech-to-text using NVIDIA NeMo FastConformer CTC model with automatic model downloading from Hugging Face.

**Features:**
- 🎤 **Arabic Speech Recognition** - Optimized for Arabic language
- 🤖 **Auto-Download** - Models downloaded automatically from Hugging Face
- 🏠 **Home Assistant Integration** - Full Wyoming protocol support
- 🔊 **Multiple Dialects** - Supports various Arabic dialects
- ⚡ **GPU Support** - Optional CUDA acceleration
- 📦 **No Manual Setup** - Zero configuration required

## Installation

### Method 1: One-Click Install (Recommended)

Click this button to add the repository directly to your Home Assistant:

[![Open your Home Assistant instance and show the add add-on repository dialog with a specific repository URL pre-filled.](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A//github.com/mohammed-alaa40123/wyoming-arabic-asr)

### Method 2: Manual Installation

1. **Add Repository**:
   - Go to **Supervisor** → **Add-on Store** → **⋮** → **Repositories**
   - Add: `https://github.com/mohammed-alaa40123/wyoming-arabic-asr`

2. **Install Add-on**:
   - Find "Wyoming Arabic ASR" in the add-on store
   - Click **Install**

3. **Configure Home Assistant**:
   Add to your `configuration.yaml`:
   ```yaml
   wyoming:
     - uri: tcp://localhost:10305
       protocol: wyoming
       name: "Arabic ASR"
       language: "ar"
   ```

4. **Restart Home Assistant** and configure your voice assistant

## Configuration

The add-on works out of the box with default settings, but you can customize:

- **Model Repository**: Change Hugging Face model repository
- **Device**: Use CPU or CUDA for inference
- **Debug**: Enable detailed logging
- **URI**: Change the Wyoming protocol endpoint

## Voice Assistant Setup

1. Go to **Settings** → **Voice assistants**
2. Create a new assistant or edit existing one
3. Set **Speech-to-text** to "Arabic ASR"
4. Set **Language** to Arabic (ar)
5. Configure **Conversation** and **Text-to-speech** as desired

## Model Information

Uses the Arabic FastConformer CTC model from:
- **Repository**: [Mo-alaa/arabic-fastconformer-ctc-onnx](https://huggingface.co/Mo-alaa/arabic-fastconformer-ctc-onnx)
- **Model Type**: NVIDIA NeMo FastConformer CTC
- **Languages**: Arabic (ar), Egyptian Arabic (ar-EG), Saudi Arabic (ar-SA), UAE Arabic (ar-AE)
- **Sample Rate**: 16kHz
- **Format**: ONNX for optimized inference

## Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/mohammed-alaa40123/wyoming-arabic-asr/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/mohammed-alaa40123/wyoming-arabic-asr/discussions)
- 📖 **Documentation**: See individual add-on README files

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits

- **NVIDIA NeMo**: For the FastConformer CTC model architecture
- **Wyoming Protocol**: For the voice assistant integration framework
- **Home Assistant**: For the amazing smart home platform