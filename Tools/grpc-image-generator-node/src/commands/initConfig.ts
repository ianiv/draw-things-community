import * as fs from "fs";
import chalk from "chalk";

const CONFIG_TEMPLATE = `# Draw Things gRPC Image Generator Configuration
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
`;

export function initConfigCommand(options: {
  output: string;
}): void {
  try {
    fs.writeFileSync(options.output, CONFIG_TEMPLATE);
    console.log(chalk.green(`✓ Created default configuration: ${options.output}`));
    console.log("\nEdit this file to customize your generation settings.");
    console.log("See README.md for complete documentation.");
  } catch (error: any) {
    console.error(chalk.red(`✗ Error creating config: ${error.message}`));
    process.exit(1);
  }
}
