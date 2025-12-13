#!/usr/bin/env python3
"""
Draw Things gRPC Image Generator CLI

A command-line tool for generating images using the Draw Things gRPC API.
"""
import argparse
import sys
import os
import time
import yaml
import grpc
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from PIL import Image
import io
import random

# Add generated module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'generated'))

import imageService_pb2
import imageService_pb2_grpc
from config_builder import ConfigurationBuilder, SamplerType, SeedMode


class ImageGenerator:
    """gRPC client for Draw Things image generation."""

    def __init__(self, host: str = "localhost", port: int = 50051, shared_secret: Optional[str] = None):
        """
        Initialize the image generator client.

        Args:
            host: Server hostname
            port: Server port
            shared_secret: Optional authentication secret
        """
        self.host = host
        self.port = port
        self.shared_secret = shared_secret
        self.channel = None
        self.stub = None

    def connect(self):
        """Establish connection to gRPC server."""
        address = f"{self.host}:{self.port}"
        print(f"Connecting to {address}...")
        self.channel = grpc.insecure_channel(address)
        self.stub = imageService_pb2_grpc.ImageGenerationServiceStub(self.channel)

        # Test connection with Echo
        try:
            echo_request = imageService_pb2.EchoRequest(
                name="draw-things-cli",
                sharedSecret=self.shared_secret if self.shared_secret else ""
            )
            response = self.stub.Echo(echo_request)
            print(f"✓ Connected: {response.message}")
            if response.sharedSecretMissing:
                print("⚠ Warning: Server requires authentication (shared secret)")
                return False
            return True
        except grpc.RpcError as e:
            print(f"✗ Connection failed: {e.details()}")
            return False

    def disconnect(self):
        """Close the gRPC channel."""
        if self.channel:
            self.channel.close()

    def list_models(self, model_type: Optional[str] = None, verbose: bool = False) -> bool:
        """
        List available models on the server.

        Args:
            model_type: Filter by type (models, loras, controlnets, textual_inversions, upscalers)
            verbose: Show detailed information

        Returns:
            True if successful, False otherwise
        """
        if not self.stub:
            print("Error: Not connected to server")
            return False

        try:
            echo_request = imageService_pb2.EchoRequest(
                name="draw-things-cli-list",
                sharedSecret=self.shared_secret if self.shared_secret else ""
            )
            response = self.stub.Echo(echo_request)

            if response.sharedSecretMissing:
                print("✗ Authentication required (shared secret)")
                return False

            # Check if model browsing is available
            if not response.HasField('override'):
                print("⚠ Server does not have model browsing enabled")
                print("  The server may not be configured to share model information.")
                return False

            override = response.override
            has_data = False

            # Parse and display models
            if (not model_type or model_type == 'models') and override.models:
                try:
                    models = json.loads(override.models)
                    if models:
                        has_data = True
                        print("\n📦 Models:")
                        print("=" * 80)
                        self._print_specifications(models, verbose)
                except json.JSONDecodeError:
                    pass

            # Parse and display LoRAs
            if (not model_type or model_type == 'loras') and override.loras:
                try:
                    loras = json.loads(override.loras)
                    if loras:
                        has_data = True
                        print("\n🎨 LoRAs:")
                        print("=" * 80)
                        self._print_specifications(loras, verbose)
                except json.JSONDecodeError:
                    pass

            # Parse and display ControlNets
            if (not model_type or model_type == 'controlnets') and override.controlNets:
                try:
                    controlnets = json.loads(override.controlNets)
                    if controlnets:
                        has_data = True
                        print("\n🎮 ControlNets:")
                        print("=" * 80)
                        self._print_specifications(controlnets, verbose)
                except json.JSONDecodeError:
                    pass

            # Parse and display Textual Inversions
            if (not model_type or model_type == 'textual_inversions') and override.textualInversions:
                try:
                    textual_inversions = json.loads(override.textualInversions)
                    if textual_inversions:
                        has_data = True
                        print("\n✏️  Textual Inversions:")
                        print("=" * 80)
                        self._print_specifications(textual_inversions, verbose)
                except json.JSONDecodeError:
                    pass

            # Parse and display Upscalers
            if (not model_type or model_type == 'upscalers') and override.upscalers:
                try:
                    upscalers = json.loads(override.upscalers)
                    if upscalers:
                        has_data = True
                        print("\n⬆️  Upscalers:")
                        print("=" * 80)
                        self._print_specifications(upscalers, verbose)
                except json.JSONDecodeError:
                    pass

            # Display raw .ckpt files if available
            if (not model_type or model_type == 'files') and response.files:
                has_data = True
                print("\n📁 Raw Model Files (.ckpt):")
                print("=" * 80)
                for f in sorted(response.files):
                    print(f"  • {f}")

            if not has_data:
                print("\n⚠ No models available on the server")
                print("  The server may not have any models downloaded yet.")
                return False

            print()
            return True

        except grpc.RpcError as e:
            print(f"✗ Failed to list models: {e.details()}")
            return False

    def _print_specifications(self, specs: List[Dict], verbose: bool):
        """Print model specifications in a readable format."""
        for spec in specs:
            file = spec.get('file', 'Unknown')

            if verbose:
                # Detailed view
                print(f"\n  📄 {file}")

                # Show all available fields
                for key, value in spec.items():
                    if key == 'file':
                        continue

                    # Format the key nicely
                    display_key = key.replace('_', ' ').title()

                    # Format the value based on type
                    if isinstance(value, dict):
                        print(f"    {display_key}:")
                        for k, v in value.items():
                            print(f"      {k}: {v}")
                    elif isinstance(value, list):
                        print(f"    {display_key}: {', '.join(map(str, value))}")
                    else:
                        print(f"    {display_key}: {value}")
            else:
                # Simple view - just the filename and name if available
                name = spec.get('name', spec.get('display_name', ''))
                if name and name != file:
                    print(f"  • {file} ({name})")
                else:
                    print(f"  • {file}")

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        config: Optional[Dict[str, Any]] = None,
        output_dir: str = "output",
        show_progress: bool = True
    ) -> bool:
        """
        Generate an image based on the prompt.

        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative prompt
            config: Generation configuration dict
            output_dir: Directory to save generated images
            show_progress: Whether to show progress updates

        Returns:
            True if successful, False otherwise
        """
        if not self.stub:
            print("Error: Not connected to server")
            return False

        # Build configuration
        if config is None:
            config = self._default_config()

        builder = ConfigurationBuilder()
        config_bytes = builder.build(config)

        # Create request
        request = imageService_pb2.ImageGenerationRequest(
            prompt=prompt,
            negativePrompt=negative_prompt,
            configuration=config_bytes,
            scaleFactor=1,
            user="draw-things-cli",
            device=imageService_pb2.LAPTOP,
            sharedSecret=self.shared_secret if self.shared_secret else "",
            chunked=True
        )

        try:
            print(f"\n🎨 Generating image...")
            print(f"   Prompt: {prompt}")
            if negative_prompt:
                print(f"   Negative: {negative_prompt}")
            print(f"   Model: {config.get('model', 'default')}")
            print(f"   Steps: {config.get('steps', 20)}")
            print(f"   Size: {config.get('start_width', 512)}x{config.get('start_height', 512)}")
            print()

            # Stream responses
            image_chunks = []
            current_image_index = 0
            last_signpost = None
            start_time = time.time()

            for response in self.stub.GenerateImage(request):
                # Handle progress signposts
                if response.HasField('currentSignpost') and show_progress:
                    signpost = self._format_signpost(response.currentSignpost)
                    if signpost != last_signpost:
                        print(f"  ⏳ {signpost}")
                        last_signpost = signpost

                # Handle preview images (optional - could save these)
                if response.previewImage and show_progress:
                    print(f"  👁  Preview available ({len(response.previewImage)} bytes)")

                # Handle download size projection
                if response.HasField('downloadSize'):
                    size_mb = response.downloadSize / (1024 * 1024)
                    print(f"  📦 Expecting {size_mb:.2f} MB of image data")

                # Handle remote download progress
                if response.HasField('remoteDownload'):
                    rd = response.remoteDownload
                    progress = (rd.bytesReceived / rd.bytesExpected * 100) if rd.bytesExpected > 0 else 0
                    print(f"  ⬇  Downloading models: {progress:.1f}% ({rd.item + 1}/{rd.itemsExpected})")

                # Handle generated images
                if response.generatedImages:
                    for img_data in response.generatedImages:
                        image_chunks.append(img_data)

                    # Check if this is the last chunk
                    if response.chunkState == imageService_pb2.LAST_CHUNK:
                        # Reassemble chunks
                        full_image_data = b''.join(image_chunks)
                        image_chunks = []

                        # Save image
                        duration = time.time() - start_time
                        output_path = self._save_image(
                            full_image_data,
                            output_dir,
                            prompt,
                            current_image_index
                        )
                        if output_path:
                            print(f"\n✓ Image saved: {output_path}")
                            print(f"  Time: {duration:.1f}s")
                            current_image_index += 1
                        else:
                            print(f"\n✗ Failed to save image")
                            return False

            if current_image_index == 0:
                print("\n✗ No images were generated")
                return False

            return True

        except grpc.RpcError as e:
            print(f"\n✗ Generation failed: {e.details()}")
            return False

    def _format_signpost(self, signpost) -> str:
        """Format a signpost message for display."""
        if signpost.HasField('textEncoded'):
            return "Text encoded"
        elif signpost.HasField('imageEncoded'):
            return "Image encoded"
        elif signpost.HasField('sampling'):
            step = signpost.sampling.step
            return f"Sampling step {step}"
        elif signpost.HasField('imageDecoded'):
            return "Image decoded"
        elif signpost.HasField('secondPassImageEncoded'):
            return "Second pass: Image encoded"
        elif signpost.HasField('secondPassSampling'):
            step = signpost.secondPassSampling.step
            return f"Second pass: Sampling step {step}"
        elif signpost.HasField('secondPassImageDecoded'):
            return "Second pass: Image decoded"
        elif signpost.HasField('faceRestored'):
            return "Face restoration applied"
        elif signpost.HasField('imageUpscaled'):
            return "Image upscaled"
        return "Processing..."

    def _save_image(
        self,
        image_data: bytes,
        output_dir: str,
        prompt: str,
        index: int
    ) -> Optional[str]:
        """
        Save image data to file.

        Args:
            image_data: Raw image bytes
            output_dir: Output directory
            prompt: Original prompt (for filename)
            index: Image index

        Returns:
            Path to saved file or None on failure
        """
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Generate filename from prompt
        safe_prompt = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in prompt)
        safe_prompt = safe_prompt[:50].strip().replace(' ', '_')
        timestamp = int(time.time())
        filename = f"{safe_prompt}_{timestamp}_{index}.png"
        output_path = os.path.join(output_dir, filename)

        try:
            # The image data might be compressed or in tensor format
            # Try to interpret it as a standard image first
            try:
                img = Image.open(io.BytesIO(image_data))
                img.save(output_path)
            except Exception:
                # If that fails, save raw bytes (might need decompression/decoding)
                with open(output_path, 'wb') as f:
                    f.write(image_data)
                print(f"  ⚠ Saved raw data (may require additional processing)")

            return output_path
        except Exception as e:
            print(f"  Error saving image: {e}")
            return None

    def _default_config(self) -> Dict[str, Any]:
        """Return default generation configuration."""
        return {
            'id': 0,
            'model': 'v1-5-pruned-emaonly.ckpt',
            'start_width': 512,
            'start_height': 512,
            'steps': 20,
            'guidance_scale': 7.5,
            'sampler': 'dpmpp_2m_karras',
            'seed': random.randint(0, 2**32 - 1),
            'batch_count': 1,
            'batch_size': 1,
            'clip_skip': 1,
            'strength': 0.75,
        }


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


def create_default_config(output_path: str = "config.yaml"):
    """Create a default configuration file with comprehensive documentation."""
    config_content = """# Draw Things gRPC Image Generator Configuration
#
# This file contains settings for the gRPC image generation client.
# See README.md for complete documentation.

# =============================================================================
# SERVER SETTINGS
# =============================================================================
server:
  # Hostname or IP address of the gRPC server
  host: localhost

  # Port the gRPC server is listening on (default is 50051)
  port: 50051

  # Optional shared secret for server authentication
  # Set this if your server requires authentication
  shared_secret: null

# =============================================================================
# GENERATION SETTINGS
# =============================================================================
generation:
  # -----------------------------------------------------------------------
  # Core Model & Dimensions
  # -----------------------------------------------------------------------

  # Model checkpoint file name to use for generation
  # Use 'list-models' command to see available models
  model: v1-5-pruned-emaonly.ckpt

  # Image width and height in pixels (must be multiples of 64)
  # Common sizes: 512x512, 768x768, 1024x1024
  start_width: 512
  start_height: 512

  # -----------------------------------------------------------------------
  # Sampling Parameters
  # -----------------------------------------------------------------------

  # Number of diffusion steps (20-50 recommended)
  # More steps = better quality but slower generation
  steps: 20

  # Guidance scale (CFG scale) controls how closely to follow the prompt
  # Range: 1-20 (higher = more faithful to prompt but less creative)
  # Common values: 7.0-8.5 for balanced results
  guidance_scale: 7.5

  # Sampling algorithm to use during generation
  # Available samplers:
  #   - dpmpp_2m_karras (recommended, good balance)
  #   - euler_a (fast, decent quality)
  #   - ddim (legacy, slower)
  #   - plms (legacy, very slow)
  #   - dpmpp_sde_karras (high quality, slower)
  #   - unipc (fast high-quality)
  #   - lcm (for LCM-optimized models only)
  sampler: dpmpp_2m_karras

  # Random seed for reproducibility
  # Use the same seed to generate identical images
  # Set to a random value for variation
  seed: 42

  # Denoising strength (for img2img operations, 0-1)
  # 0 = no change to input image, 1 = complete regeneration
  strength: 0.75

  # -----------------------------------------------------------------------
  # Batch Settings
  # -----------------------------------------------------------------------

  # Number of batches to generate (each batch contains batch_size images)
  batch_count: 1

  # Images to generate per batch (parallel generation)
  batch_size: 1

  # CLIP layer skip (1-2 for most models)
  # Higher values = less detailed prompts but sometimes better results
  clip_skip: 1

  # -----------------------------------------------------------------------
  # High-Resolution Fix (2-Pass Generation)
  # -----------------------------------------------------------------------
  # Uncomment to enable high-resolution fix for better quality at larger sizes

  # hires_fix: false
  #
  # # First pass dimensions (usually smaller)
  # hires_fix_start_width: 512
  # hires_fix_start_height: 512
  #
  # # Denoising strength for second pass (0.5-0.9)
  # hires_fix_strength: 0.7
  #
  # # Upscaler to use for hi-res fix
  # # Available upscalers can be listed with: list-models --type upscalers
  # upscaler: RealESRGAN_x4plus.pth

  # -----------------------------------------------------------------------
  # Post-Processing
  # -----------------------------------------------------------------------
  # Uncomment to enable post-processing effects

  # # Face restoration model (improves generated faces)
  # # Examples: GFPGANv1.4.pth, codeformer
  # face_restoration: GFPGANv1.4.pth

  # -----------------------------------------------------------------------
  # SDXL Specific (for SDXL models)
  # -----------------------------------------------------------------------
  # Uncomment these settings when using SDXL models

  # # SDXL refiner model for two-stage generation
  # refiner_model: sd_xl_refiner_1.0.ckpt
  #
  # # When to switch to refiner (0.0-1.0, typically 0.7-0.8)
  # refiner_start: 0.7

  # -----------------------------------------------------------------------
  # Inpainting & Advanced
  # -----------------------------------------------------------------------
  # Uncomment for specialized features

  # # Mask blur radius for inpainting (pixels)
  # mask_blur: 8.0
  #
  # # Timestep shift for Flux/SD3 models (typically 0.5-2.0)
  # shift: 1.0

  # -----------------------------------------------------------------------
  # LoRA Weights (Style & Character Modifiers)
  # -----------------------------------------------------------------------
  # Uncomment and modify to apply LoRA weights
  # Use 'list-models --type loras' to see available LoRAs

  # loras:
  #   # Simple LoRA with just filename
  #   - file: add_detail.safetensors
  #     weight: 0.8  # 0-1: how strongly to apply this LoRA
  #
  #   # Another LoRA
  #   - file: anime_style_v2.safetensors
  #     weight: 0.6

  # -----------------------------------------------------------------------
  # ControlNet (Guided Generation)
  # -----------------------------------------------------------------------
  # Uncomment and modify to use ControlNet for guided generation
  # Use 'list-models --type controlnets' to see available ControlNets

  # controls:
  #   - file: control_v11p_sd15_canny.ckpt
  #     weight: 1.0  # How strongly to apply the ControlNet
  #     guidance_start: 0.0  # When to start applying (0-1)
  #     guidance_end: 1.0    # When to stop applying (0-1)
  #     control_mode: 0      # 0=Balanced, 1=Prompt-weighted, 2=Control-weighted
  #     no_prompt: false     # Ignore text prompt if true
  #     down_sampling_rate: 1.0  # Downsample ratio for efficiency

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================
output:
  # Directory where generated images will be saved
  directory: output

  # Whether to show progress updates during generation
  show_progress: true
"""

    try:
        with open(output_path, 'w') as f:
            f.write(config_content)
        print(f"✓ Created default configuration: {output_path}")
        print("\nEdit this file to customize your generation settings.")
        print("See README.md for complete documentation.")
        return True
    except Exception as e:
        print(f"✗ Error creating config: {e}")
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Draw Things gRPC Image Generator CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with default settings
  %(prog)s generate "a beautiful sunset over mountains"

  # Use a config file
  %(prog)s generate "cyberpunk city" -c config.yaml

  # Override config settings
  %(prog)s generate "portrait" -c config.yaml --steps 30 --width 768 --height 768

  # Create default config file
  %(prog)s init-config

  # List available models and LoRAs
  %(prog)s list-models

  # List only LoRAs with details
  %(prog)s list-models --type loras --verbose

  # Specify output directory
  %(prog)s generate "landscape" -o my_images/
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate an image')
    gen_parser.add_argument('prompt', help='Text prompt for image generation')
    gen_parser.add_argument('-n', '--negative-prompt', default='', help='Negative prompt')
    gen_parser.add_argument('-c', '--config', help='Path to configuration file')
    gen_parser.add_argument('-o', '--output', help='Output directory for generated images')
    gen_parser.add_argument('--host', help='Server hostname')
    gen_parser.add_argument('--port', type=int, help='Server port')
    gen_parser.add_argument('--secret', help='Shared secret for authentication')

    # Generation parameter overrides
    gen_parser.add_argument('--model', help='Model file name')
    gen_parser.add_argument('--steps', type=int, help='Number of generation steps')
    gen_parser.add_argument('--width', type=int, help='Image width')
    gen_parser.add_argument('--height', type=int, help='Image height')
    gen_parser.add_argument('--cfg', '--guidance-scale', type=float, dest='guidance_scale',
                           help='Guidance scale (CFG)')
    gen_parser.add_argument('--sampler', help='Sampler type')
    gen_parser.add_argument('--seed', type=int, help='Random seed')
    gen_parser.add_argument('--batch-count', type=int, help='Number of batches')
    gen_parser.add_argument('--batch-size', type=int, help='Images per batch')

    # Init-config command
    init_parser = subparsers.add_parser('init-config', help='Create default configuration file')
    init_parser.add_argument('-o', '--output', default='config.yaml',
                            help='Output path for config file (default: config.yaml)')

    # List-models command
    list_parser = subparsers.add_parser('list-models', help='List available models on the server')
    list_parser.add_argument('-c', '--config', help='Path to configuration file')
    list_parser.add_argument('--host', help='Server hostname')
    list_parser.add_argument('--port', type=int, help='Server port')
    list_parser.add_argument('--secret', help='Shared secret for authentication')
    list_parser.add_argument('-t', '--type',
                            choices=['models', 'loras', 'controlnets', 'textual_inversions', 'upscalers', 'files'],
                            help='Filter by model type')
    list_parser.add_argument('-v', '--verbose', action='store_true',
                            help='Show detailed information')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Handle init-config command
    if args.command == 'init-config':
        return 0 if create_default_config(args.output) else 1

    # Handle generate command
    if args.command == 'generate':
        # Load config file if provided
        config_data = {}
        if args.config:
            config_data = load_config(args.config)

        # Get server settings
        server_config = config_data.get('server', {})
        host = args.host or server_config.get('host', 'localhost')
        port = args.port or server_config.get('port', 50051)
        secret = args.secret or server_config.get('shared_secret')

        # Get generation settings
        gen_config = config_data.get('generation', {})

        # Apply command-line overrides
        if args.model:
            gen_config['model'] = args.model
        if args.steps:
            gen_config['steps'] = args.steps
        if args.width:
            gen_config['start_width'] = args.width
        if args.height:
            gen_config['start_height'] = args.height
        if args.guidance_scale:
            gen_config['guidance_scale'] = args.guidance_scale
        if args.sampler:
            gen_config['sampler'] = args.sampler
        if args.seed is not None:
            gen_config['seed'] = args.seed
        if args.batch_count:
            gen_config['batch_count'] = args.batch_count
        if args.batch_size:
            gen_config['batch_size'] = args.batch_size

        # Get output settings
        output_config = config_data.get('output', {})
        output_dir = args.output or output_config.get('directory', 'output')
        show_progress = output_config.get('show_progress', True)

        # Create client and generate
        client = ImageGenerator(host, port, secret)

        if not client.connect():
            return 1

        try:
            success = client.generate_image(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                config=gen_config if gen_config else None,
                output_dir=output_dir,
                show_progress=show_progress
            )
            return 0 if success else 1
        finally:
            client.disconnect()

    # Handle list-models command
    if args.command == 'list-models':
        # Load config file if provided
        config_data = {}
        if args.config:
            config_data = load_config(args.config)

        # Get server settings
        server_config = config_data.get('server', {})
        host = args.host or server_config.get('host', 'localhost')
        port = args.port or server_config.get('port', 50051)
        secret = args.secret or server_config.get('shared_secret')

        # Create client and list models
        client = ImageGenerator(host, port, secret)

        if not client.connect():
            return 1

        try:
            success = client.list_models(
                model_type=args.type,
                verbose=args.verbose
            )
            return 0 if success else 1
        finally:
            client.disconnect()

    return 0


if __name__ == '__main__':
    sys.exit(main())
