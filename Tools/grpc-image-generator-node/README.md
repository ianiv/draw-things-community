# Draw Things gRPC Image Generator CLI (Node.js/TypeScript)

A modern TypeScript/Node.js command-line tool for generating images using the Draw Things gRPC API.

## Features

- 🎨 Generate images from text prompts
- 📋 List available models, LoRAs, and ControlNets on the server
- ⚙️ YAML configuration file support
- 🔧 Command-line parameter overrides
- 📊 Real-time progress display
- 💾 Automatic image saving
- 🔒 Shared secret authentication support
- 📦 Handles chunked responses for large images
- 🎯 Support for advanced settings (LoRA, ControlNet, hi-res fix, etc.)
- 🔧 Full TypeScript support with type safety

## Prerequisites

- Node.js 16+ and npm/yarn
- Draw Things gRPC server running (see main repository README)

## Installation

### Quick Install

```bash
cd tools/grpc-image-generator-node
npm install
npm run build
npm install -g .
```

After installation, you can use `draw-things` from anywhere on your system:

```bash
draw-things init-config
draw-things list-models
draw-things generate "your prompt"
```

For detailed installation options, see [INSTALL.md](./INSTALL.md).

## Quick Start

### 1. Create a Default Configuration File

```bash
draw-things init-config
```

This creates a `config.yaml` file with default settings you can customize.

### 2. Generate Your First Image

```bash
draw-things generate "a beautiful sunset over mountains"
```

### 3. List Available Models

```bash
draw-things list-models
```

### 4. Use Custom Configuration

```bash
draw-things generate "cyberpunk city at night" -c config.yaml
```

### 5. Override Configuration Settings

```bash
draw-things generate "portrait of a woman" \
  -c config.yaml \
  --steps 30 \
  --width 768 \
  --height 768 \
  --cfg 8.5
```

## Upgrade from Development

If you previously used `npm start`, you can now just use `draw-things` directly:

```bash
# Old way (during development)
npm start -- generate "test"

# New way (after installation)
draw-things generate "test"
```

## Usage

### Commands

#### `init-config` - Create Default Configuration

```bash
draw-things init-config [-o OUTPUT_PATH]
```

**Options:**
- `-o, --output <path>`: Output path for config file (default: `config.yaml`)

**Example:**
```bash
draw-things init-config -o my-config.yaml
```

#### `list-models` - List Available Models

Query the server to discover available models, LoRAs, ControlNets, and other resources.

```bash
draw-things list-models [OPTIONS]
```

**Options:**
- `-c, --config <path>`: Path to configuration file
- `--host <host>`: Server hostname (default: localhost)
- `--port <port>`: Server port (default: 50051)
- `--secret <secret>`: Shared secret for authentication
- `-t, --type <type>`: Filter by model type
  - `models` - Stable Diffusion models
  - `loras` - LoRA weights
  - `controlnets` - ControlNet models
  - `textual_inversions` - Textual inversions/embeddings
  - `upscalers` - Upscaler models
  - `files` - Raw .ckpt files
- `-v, --verbose`: Show detailed information

**Examples:**
```bash
# List all available resources
draw-things list-models

# List only LoRAs
draw-things list-models --type loras

# List models with detailed information
draw-things list-models --type models --verbose

# Use config file for server settings
draw-things list-models -c config.yaml
```

#### `generate` - Generate Images

```bash
draw-things generate <prompt> [OPTIONS]
```

**Arguments:**
- `<prompt>`: Text prompt for image generation (required)

**Options:**
- `-n, --negative-prompt <text>`: Negative prompt
- `-c, --config <path>`: Path to configuration file
- `-o, --output <dir>`: Output directory for generated images
- `--host <host>`: Server hostname (default: localhost)
- `--port <port>`: Server port (default: 50051)
- `--secret <secret>`: Shared secret for authentication
- `--model <name>`: Model file name
- `--steps <num>`: Number of generation steps
- `--width <num>`: Image width in pixels
- `--height <num>`: Image height in pixels
- `--cfg <num>`: Guidance scale (CFG)
- `--sampler <type>`: Sampler type
- `--seed <num>`: Random seed
- `--batch-count <num>`: Number of batches
- `--batch-size <num>`: Images per batch

**Examples:**
```bash
# Simple generation with defaults
draw-things generate "a red rose"

# With negative prompt
draw-things generate "beautiful landscape" \
  -n "blurry, low quality, distorted"

# Specify output directory
draw-things generate "portrait" -o ./my_images/

# High resolution with more steps
draw-things generate "detailed fantasy castle" \
  --width 1024 \
  --height 1024 \
  --steps 50 \
  --cfg 9.0

# Reproducible generation with seed
draw-things generate "sunset" \
  --seed 42 \
  --steps 30

# Using a specific model
draw-things generate "realistic portrait" \
  --model realisticVisionV51.ckpt \
  --steps 35 \
  --cfg 7.0
```

## Configuration File

The configuration file uses YAML format and supports all generation parameters.

### Example Configuration

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

output:
  directory: output
  show_progress: true
```

## Output

Generated images are saved to the output directory (default: `./output/`) with filenames based on:
- The prompt (truncated and sanitized)
- Unix timestamp
- Image index

Example filename: `beautiful_sunset_over_mountains_1702512345_0.png`

## Model Discovery

The `list-models` command queries the server to discover available resources:

```
Connecting to localhost:50051...
✓ Connected: HELLO draw-things-cli-list

📦 Models:
================================================================================
  • v1-5-pruned-emaonly.ckpt
  • sd_xl_base_1.0.ckpt (Stable Diffusion XL 1.0 Base)
  • dreamshaper_8.safetensors (DreamShaper 8)

🎨 LoRAs:
================================================================================
  • add_detail.safetensors
  • anime_style_v2.safetensors

🎮 ControlNets:
================================================================================
  • control_v11p_sd15_canny.ckpt
  • control_v11f1p_sd15_depth.pth

⬆️  Upscalers:
================================================================================
  • RealESRGAN_x4plus.pth
  • RealESRGAN_x4plus_anime_6B.pth
```

With `--verbose` flag:

```
📦 Models:
================================================================================

  📄 sd_xl_base_1.0.ckpt
    Name: Stable Diffusion XL 1.0 Base
    Model Version: SDXL
    Resolution: 1024
    Description: Base model for SDXL
```

## Progress Display

During generation, the CLI displays:

```
🎨 Generating image...
   Prompt: a beautiful sunset over mountains
   Model: v1-5-pruned-emaonly.ckpt
   Steps: 20
   Size: 512x512

  ⏳ Text encoded
  ⏳ Image encoded
  ⏳ Sampling step 1
  ⏳ Sampling step 10
  ⏳ Sampling step 20
  ⏳ Image decoded

✓ Image saved: output/beautiful_sunset_over_mountains_1702512345_0.png
  Time: 12.3s
```

## Samplers

Available sampler types:
- `dpmpp_2m_karras` (default, recommended)
- `euler_a`
- `ddim`
- `plms`
- `dpmpp_sde_karras`
- `unipc`
- `lcm` (for LCM models)

## Troubleshooting

### Connection Issues

**Problem:** `Connection failed: getaddrinfo ENOTFOUND localhost`

**Solutions:**
- Verify the gRPC server is running
- Check the host and port are correct
- Ensure firewall allows the connection
- Try `127.0.0.1` instead of `localhost`

### Authentication Issues

**Problem:** `Server requires authentication (shared secret)`

**Solution:** Add the shared secret to your config file or use `--secret` flag:
```bash
draw-things generate "test" --secret "your-secret-here"
```

### Model Not Found

**Problem:** Generated images are empty or server returns errors

**Solutions:**
- Verify the model file exists on the server with `list-models`
- Check the model name in your configuration
- Use the `echo` endpoint to verify available models

## Development

### Building from Source

```bash
# Install dependencies
npm install

# Build TypeScript
npm run build

# Run from dist
node dist/cli.js generate "test"
```

### Development Mode with ts-node

```bash
npm run dev -- generate "test"
npm run dev -- list-models
npm run dev -- init-config
```

### Watch Mode

```bash
tsc --watch
```

## Project Structure

```
tools/grpc-image-generator-node/
├── src/
│   ├── cli.ts                 # Main CLI entry point
│   ├── imageGenerator.ts      # gRPC client implementation
│   ├── configBuilder.ts       # FlatBuffer configuration builder
│   └── commands/
│       ├── generate.ts        # Generate command handler
│       ├── listModels.ts      # List models command handler
│       └── initConfig.ts      # Init config command handler
├── dist/                      # Compiled JavaScript output
├── package.json
├── tsconfig.json
└── README.md
```

## Dependencies

- **@grpc/grpc-js** - gRPC client library
- **@grpc/proto-loader** - Protocol Buffer loader
- **commander** - CLI argument parsing
- **chalk** - Terminal styling
- **js-yaml** - YAML parsing
- **flatbuffers** - FlatBuffer serialization
- **sharp** - Image processing (optional, for advanced image handling)

## Comparison with Python Version

This Node.js/TypeScript version provides feature parity with the Python version:

| Feature | Python | Node.js |
|---------|--------|---------|
| Generate images | ✓ | ✓ |
| List models | ✓ | ✓ |
| Configuration files | ✓ | ✓ |
| Command-line overrides | ✓ | ✓ |
| Progress display | ✓ | ✓ |
| Authentication | ✓ | ✓ |
| Batch generation | ✓ | ✓ |
| LoRA/ControlNet support | ✓ | ✓ |
| Image saving | ✓ | ✓ |
| Type safety | - | ✓ |

## Performance Notes

- Node.js version may have different performance characteristics than Python
- gRPC streaming is fully supported
- Image saving uses synchronous I/O for simplicity (can be optimized for async)

## See Also

- [Draw Things gRPC API Documentation](../../GRPC_API.md)
- [Python CLI Version](../grpc-image-generator/)
- [Main Repository README](../../README.md)
- [Draw Things App](https://apps.apple.com/us/app/draw-things-ai-generation/id6444050820)

## License

This tool is part of the Draw Things Community repository. See the main repository LICENSE for details.
