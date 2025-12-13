# Draw Things gRPC Image Generator CLI

A Python command-line tool for generating images using the Draw Things gRPC API.

## Features

- 🎨 Generate images from text prompts
- ⚙️ Configuration file support (YAML)
- 🔧 Command-line parameter overrides
- 📊 Real-time progress display
- 💾 Automatic image saving
- 🔒 Shared secret authentication support
- 📦 Handles chunked responses for large images
- 🎯 Support for advanced settings (LoRA, ControlNet, hi-res fix, etc.)

## Installation

### Prerequisites

- Python 3.8 or higher
- Draw Things gRPC server running (see main repository README)

### Install Dependencies

```bash
cd tools/grpc-image-generator
pip install -r requirements.txt
```

## Quick Start

### 1. Create a Default Configuration File

```bash
./draw_things_cli.py init-config
```

This creates a `config.yaml` file with default settings that you can customize.

### 2. Generate Your First Image

```bash
./draw_things_cli.py generate "a beautiful sunset over mountains"
```

### 3. Use Custom Configuration

```bash
./draw_things_cli.py generate "cyberpunk city at night" -c config.yaml
```

### 4. Override Configuration Settings

```bash
./draw_things_cli.py generate "portrait of a woman" \
    -c config.yaml \
    --steps 30 \
    --width 768 \
    --height 768 \
    --cfg 8.5
```

## Usage

### Commands

#### `init-config` - Create Default Configuration

```bash
./draw_things_cli.py init-config [-o OUTPUT_PATH]
```

Options:
- `-o, --output`: Output path for config file (default: `config.yaml`)

Example:
```bash
./draw_things_cli.py init-config -o my-config.yaml
```

#### `generate` - Generate Images

```bash
./draw_things_cli.py generate PROMPT [OPTIONS]
```

**Required Arguments:**
- `PROMPT`: Text prompt for image generation

**Optional Arguments:**

*Server Connection:*
- `-c, --config PATH`: Path to configuration file
- `--host HOST`: Server hostname (default: localhost)
- `--port PORT`: Server port (default: 50051)
- `--secret SECRET`: Shared secret for authentication

*Generation Parameters:*
- `-n, --negative-prompt TEXT`: Negative prompt
- `--model NAME`: Model file name
- `--steps N`: Number of generation steps
- `--width N`: Image width in pixels
- `--height N`: Image height in pixels
- `--cfg, --guidance-scale N`: Guidance scale (CFG)
- `--sampler TYPE`: Sampler type (see Samplers section)
- `--seed N`: Random seed for reproducibility
- `--batch-count N`: Number of batches to generate
- `--batch-size N`: Images per batch

*Output:*
- `-o, --output DIR`: Output directory for generated images

### Configuration File

The configuration file uses YAML format and supports all generation parameters.

#### Example Configuration

```yaml
server:
  host: localhost
  port: 50051
  shared_secret: null  # Set if server requires authentication

generation:
  model: v1-5-pruned-emaonly.ckpt
  start_width: 512
  start_height: 512
  steps: 20
  guidance_scale: 7.5
  sampler: dpmpp_2m_karras
  seed: 42
  batch_count: 1
  batch_size: 1
  clip_skip: 1
  strength: 0.75

  # Advanced settings (uncomment to use)
  # hires_fix: true
  # hires_fix_start_width: 512
  # hires_fix_start_height: 512
  # hires_fix_strength: 0.7
  # upscaler: RealESRGAN_x4plus_anime_6B.pth
  # face_restoration: GFPGANv1.4.pth
  # refiner_model: sd_xl_refiner_1.0.ckpt
  # refiner_start: 0.7
  # mask_blur: 8.0
  # shift: 1.0

  # LoRA weights
  # loras:
  #   - file: add_detail.safetensors
  #     weight: 0.8
  #   - file: anime_style.safetensors
  #     weight: 0.6

  # ControlNet
  # controls:
  #   - file: control_v11p_sd15_canny.ckpt
  #     weight: 1.0
  #     guidance_start: 0.0
  #     guidance_end: 1.0

output:
  directory: output
  show_progress: true
```

### Samplers

Available sampler types:
- `dpmpp_2m_karras` (default, recommended)
- `euler_a`
- `ddim`
- `plms`
- `dpmpp_sde_karras`
- `unipc`
- `lcm` (for LCM models)

## Examples

### Basic Generation

```bash
# Simple generation with defaults
./draw_things_cli.py generate "a red rose"

# With negative prompt
./draw_things_cli.py generate "beautiful landscape" \
    -n "blurry, low quality, distorted"

# Specify output directory
./draw_things_cli.py generate "portrait" -o ./my_images/
```

### Using Configuration Files

```bash
# Generate using config file
./draw_things_cli.py generate "cyberpunk city" -c config.yaml

# Override specific settings from config
./draw_things_cli.py generate "anime character" \
    -c anime_config.yaml \
    --steps 40 \
    --seed 12345
```

### Advanced Generation

```bash
# High resolution with more steps
./draw_things_cli.py generate "detailed fantasy castle" \
    --width 1024 \
    --height 1024 \
    --steps 50 \
    --cfg 9.0

# Reproducible generation with seed
./draw_things_cli.py generate "sunset" \
    --seed 42 \
    --steps 30

# Using a specific model
./draw_things_cli.py generate "realistic portrait" \
    --model realisticVisionV51.ckpt \
    --steps 35 \
    --cfg 7.0
```

### Remote Server

```bash
# Connect to remote server
./draw_things_cli.py generate "mountain landscape" \
    --host 192.168.1.100 \
    --port 50051

# With authentication
./draw_things_cli.py generate "portrait" \
    --host remote.server.com \
    --port 50051 \
    --secret "my-secret-key"
```

### Batch Generation

```bash
# Generate 4 variations
./draw_things_cli.py generate "fantasy landscape" \
    --batch-size 4

# Generate 3 batches of 2 images each (total: 6 images)
./draw_things_cli.py generate "character design" \
    --batch-count 3 \
    --batch-size 2
```

## Output

Generated images are saved to the output directory (default: `./output/`) with filenames based on:
- The prompt (truncated and sanitized)
- Unix timestamp
- Image index

Example filename: `beautiful_sunset_over_mountains_1702512345_0.png`

## Progress Display

During generation, the CLI displays:
- ⏳ Current generation stage (text encoding, sampling, etc.)
- 👁 Preview availability
- 📦 Expected download size
- ⬇ Model download progress (if server needs to download models)
- ✓ Completion status and timing

Example output:
```
Connecting to localhost:50051...
✓ Connected: HELLO draw-things-cli

🎨 Generating image...
   Prompt: a beautiful sunset over mountains
   Negative: blurry, low quality
   Model: v1-5-pruned-emaonly.ckpt
   Steps: 20
   Size: 512x512

  ⏳ Text encoded
  ⏳ Image encoded
  ⏳ Sampling step 1
  ⏳ Sampling step 5
  ⏳ Sampling step 10
  ⏳ Sampling step 15
  ⏳ Sampling step 20
  ⏳ Image decoded
  📦 Expecting 1.23 MB of image data

✓ Image saved: output/beautiful_sunset_over_mountains_1702512345_0.png
  Time: 12.3s
```

## Configuration Reference

### Server Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | string | localhost | gRPC server hostname |
| `port` | integer | 50051 | gRPC server port |
| `shared_secret` | string | null | Authentication secret (if required) |

### Generation Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | v1-5-pruned-emaonly.ckpt | Model checkpoint file |
| `start_width` | integer | 512 | Image width in pixels |
| `start_height` | integer | 512 | Image height in pixels |
| `steps` | integer | 20 | Number of diffusion steps |
| `guidance_scale` | float | 7.5 | CFG scale for prompt adherence |
| `sampler` | string | dpmpp_2m_karras | Sampling algorithm |
| `seed` | integer | random | Random seed for reproducibility |
| `batch_count` | integer | 1 | Number of batches |
| `batch_size` | integer | 1 | Images per batch |
| `clip_skip` | integer | 1 | CLIP layers to skip |
| `strength` | float | 0.75 | Denoising strength (for img2img) |

### Advanced Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hires_fix` | boolean | false | Enable high-res fix (2-pass) |
| `hires_fix_start_width` | integer | - | First pass width |
| `hires_fix_start_height` | integer | - | First pass height |
| `hires_fix_strength` | float | 0.7 | Second pass denoising strength |
| `upscaler` | string | - | Upscaler model for hi-res fix |
| `face_restoration` | string | - | Face restoration model |
| `refiner_model` | string | - | SDXL refiner model |
| `refiner_start` | float | 0.7 | When to switch to refiner (0-1) |
| `mask_blur` | float | 0 | Mask blur radius |
| `shift` | float | 1.0 | Timestep shift parameter |
| `loras` | array | [] | LoRA weights to apply |
| `controls` | array | [] | ControlNet configurations |

### Output Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `directory` | string | output | Output directory for images |
| `show_progress` | boolean | true | Display progress during generation |

## Troubleshooting

### Connection Issues

**Problem:** `Connection failed: failed to connect to all addresses`

**Solutions:**
- Verify the gRPC server is running
- Check the host and port are correct
- Ensure firewall allows the connection
- Try connecting to `127.0.0.1` instead of `localhost`

### Authentication Issues

**Problem:** `Server requires authentication (shared secret)`

**Solution:** Add the shared secret to your config file or use `--secret` flag:
```bash
./draw_things_cli.py generate "test" --secret "your-secret-here"
```

### Model Not Found

**Problem:** Generated images are empty or server returns errors

**Solutions:**
- Verify the model file exists on the server
- Check the model name in your configuration
- Use the `Echo` endpoint to list available models (when browsing is enabled)

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'grpc'`

**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

## Architecture

The CLI tool consists of:

1. **draw_things_cli.py** - Main CLI application with argparse interface
2. **config_builder.py** - FlatBuffer configuration builder
3. **generated/** - Auto-generated gRPC client code from proto files
4. **requirements.txt** - Python dependencies

### Data Flow

1. CLI parses arguments and loads config file
2. Configuration is built into FlatBuffer format
3. gRPC request is created and sent to server
4. Server streams back progress updates and image data
5. CLI displays progress and saves completed images

## Development

### Regenerating gRPC Code

If the proto files are updated, regenerate the Python code:

```bash
python3 -m grpc_tools.protoc \
    -I../../Libraries/GRPC/Models/Sources/imageService \
    --python_out=./generated \
    --grpc_python_out=./generated \
    ../../Libraries/GRPC/Models/Sources/imageService/imageService.proto
```

## License

This tool is part of the Draw Things Community repository. See the main repository LICENSE for details.

## See Also

- [Draw Things gRPC API Documentation](../../GRPC_API.md)
- [Main Repository README](../../README.md)
- [Draw Things App](https://apps.apple.com/us/app/draw-things-ai-generation/id6444050820)
