import * as grpc from "@grpc/grpc-js";
import * as protoLoader from "@grpc/proto-loader";
import chalk from "chalk";
import path from "path";
import { ConfigBuilder } from "./configBuilder";

export interface GenerationConfig {
  [key: string]: any;
}

export class ImageGenerator {
  private host: string;
  private port: number;
  private sharedSecret?: string;
  private client?: any;
  private channel?: grpc.Channel;

  constructor(
    host: string = "localhost",
    port: number = 50051,
    sharedSecret?: string
  ) {
    this.host = host;
    this.port = port;
    this.sharedSecret = sharedSecret;
  }

  async connect(): Promise<boolean> {
    try {
      const address = `${this.host}:${this.port}`;
      console.log(`Connecting to ${address}...`);

      // Load proto files
      const protoPath = path.join(
        __dirname,
        "../../..",
        "GRPC/Models/Sources/imageService/imageService.proto"
      );

      const packageDef = await protoLoader.load(protoPath, {
        keepCase: true,
        longs: String,
        enums: String,
        defaults: true,
        oneofs: true,
      });

      const imageServiceProto = grpc.loadPackageDefinition(packageDef);
      const ImageGenerationService =
        (imageServiceProto.ImageGenerationService as any) || imageServiceProto;

      // Create channel and client
      this.channel = new grpc.Channel(address, grpc.ChannelCredentials.createInsecure());
      this.client = new (ImageGenerationService as any).ImageGenerationService(
        address,
        grpc.credentials.createInsecure()
      );

      // Test connection with Echo
      const echoResponse = await this.echo("draw-things-cli");

      console.log(chalk.green(`✓ Connected: ${echoResponse.message}`));

      if (echoResponse.sharedSecretMissing) {
        console.log(
          chalk.yellow(
            "⚠ Warning: Server requires authentication (shared secret)"
          )
        );
        return false;
      }

      return true;
    } catch (error: any) {
      console.log(chalk.red(`✗ Connection failed: ${error.message}`));
      return false;
    }
  }

  disconnect(): void {
    if (this.channel) {
      this.channel.close();
    }
  }

  private async echo(name: string): Promise<any> {
    return new Promise((resolve, reject) => {
      if (!this.client) {
        reject(new Error("Not connected"));
        return;
      }

      this.client.echo(
        {
          name,
          sharedSecret: this.sharedSecret || "",
        },
        (error: any, response: any) => {
          if (error) {
            reject(error);
          } else {
            resolve(response);
          }
        }
      );
    });
  }

  async listModels(
    modelType?: string,
    verbose: boolean = false
  ): Promise<boolean> {
    try {
      if (!this.client) {
        console.log(chalk.red("Error: Not connected to server"));
        return false;
      }

      const response = await this.echo("draw-things-cli-list");

      if (response.sharedSecretMissing) {
        console.log(chalk.red("✗ Authentication required (shared secret)"));
        return false;
      }

      // Check if model browsing is available
      if (!response.override) {
        console.log(chalk.yellow("⚠ Server does not have model browsing enabled"));
        console.log("  The server may not be configured to share model information.");
        return false;
      }

      const override = response.override;
      let hasData = false;

      // Parse and display models
      if (
        (!modelType || modelType === "models") &&
        override.models &&
        override.models.length > 0
      ) {
        try {
          const models = JSON.parse(override.models);
          if (models.length > 0) {
            hasData = true;
            console.log("\n📦 Models:");
            console.log("=".repeat(80));
            this.printSpecifications(models, verbose);
          }
        } catch (e) {
          // Ignore parsing errors
        }
      }

      // Parse and display LoRAs
      if (
        (!modelType || modelType === "loras") &&
        override.loras &&
        override.loras.length > 0
      ) {
        try {
          const loras = JSON.parse(override.loras);
          if (loras.length > 0) {
            hasData = true;
            console.log("\n🎨 LoRAs:");
            console.log("=".repeat(80));
            this.printSpecifications(loras, verbose);
          }
        } catch (e) {
          // Ignore parsing errors
        }
      }

      // Parse and display ControlNets
      if (
        (!modelType || modelType === "controlnets") &&
        override.controlNets &&
        override.controlNets.length > 0
      ) {
        try {
          const controlnets = JSON.parse(override.controlNets);
          if (controlnets.length > 0) {
            hasData = true;
            console.log("\n🎮 ControlNets:");
            console.log("=".repeat(80));
            this.printSpecifications(controlnets, verbose);
          }
        } catch (e) {
          // Ignore parsing errors
        }
      }

      // Parse and display Textual Inversions
      if (
        (!modelType || modelType === "textual_inversions") &&
        override.textualInversions &&
        override.textualInversions.length > 0
      ) {
        try {
          const textualInversions = JSON.parse(override.textualInversions);
          if (textualInversions.length > 0) {
            hasData = true;
            console.log("\n✏️  Textual Inversions:");
            console.log("=".repeat(80));
            this.printSpecifications(textualInversions, verbose);
          }
        } catch (e) {
          // Ignore parsing errors
        }
      }

      // Parse and display Upscalers
      if (
        (!modelType || modelType === "upscalers") &&
        override.upscalers &&
        override.upscalers.length > 0
      ) {
        try {
          const upscalers = JSON.parse(override.upscalers);
          if (upscalers.length > 0) {
            hasData = true;
            console.log("\n⬆️  Upscalers:");
            console.log("=".repeat(80));
            this.printSpecifications(upscalers, verbose);
          }
        } catch (e) {
          // Ignore parsing errors
        }
      }

      // Display raw .ckpt files if available
      if (
        (!modelType || modelType === "files") &&
        response.files &&
        response.files.length > 0
      ) {
        hasData = true;
        console.log("\n📁 Raw Model Files (.ckpt):");
        console.log("=".repeat(80));
        response.files.sort().forEach((file: string) => {
          console.log(`  • ${file}`);
        });
      }

      if (!hasData) {
        console.log("\n⚠ No models available on the server");
        console.log("  The server may not have any models downloaded yet.");
        return false;
      }

      console.log();
      return true;
    } catch (error: any) {
      console.log(chalk.red(`✗ Failed to list models: ${error.message}`));
      return false;
    }
  }

  private printSpecifications(specs: any[], verbose: boolean): void {
    specs.forEach((spec) => {
      const file = spec.file || "Unknown";

      if (verbose) {
        // Detailed view
        console.log(`\n  📄 ${file}`);

        // Show all available fields
        Object.entries(spec).forEach(([key, value]) => {
          if (key === "file") return;

          // Format the key nicely
          const displayKey = key
            .replace(/_/g, " ")
            .split(" ")
            .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
            .join(" ");

          // Format the value based on type
          if (typeof value === "object" && value !== null) {
            console.log(`    ${displayKey}:`);
            Object.entries(value).forEach(([k, v]) => {
              console.log(`      ${k}: ${v}`);
            });
          } else if (Array.isArray(value)) {
            console.log(`    ${displayKey}: ${value.join(", ")}`);
          } else {
            console.log(`    ${displayKey}: ${value}`);
          }
        });
      } else {
        // Simple view - just the filename and name if available
        const name = spec.name || spec.display_name || "";
        if (name && name !== file) {
          console.log(`  • ${file} (${name})`);
        } else {
          console.log(`  • ${file}`);
        }
      }
    });
  }

  async generateImage(
    prompt: string,
    negativePrompt: string = "",
    config?: GenerationConfig,
    outputDir: string = "output",
    showProgress: boolean = true
  ): Promise<boolean> {
    try {
      if (!this.client) {
        console.log(chalk.red("Error: Not connected to server"));
        return false;
      }

      // Build configuration
      if (!config) {
        config = this.defaultConfig();
      }

      const configBuilder = new ConfigBuilder();
      const configBytes = configBuilder.build(config);

      console.log("\n🎨 Generating image...");
      console.log(`   Prompt: ${prompt}`);
      if (negativePrompt) {
        console.log(`   Negative: ${negativePrompt}`);
      }
      console.log(`   Model: ${config.model || "default"}`);
      console.log(`   Steps: ${config.steps || 20}`);
      console.log(
        `   Size: ${config.start_width || 512}x${config.start_height || 512}`
      );
      console.log();

      // Stream responses
      const imageChunks: Buffer[] = [];
      let currentImageIndex = 0;
      let lastSignpost = "";
      const startTime = Date.now();

      return new Promise((resolve) => {
        if (!this.client) {
          resolve(false);
          return;
        }

        const call = this.client.generateImage({
          prompt,
          negativePrompt,
          configuration: configBytes,
          scaleFactor: 1,
          user: "draw-things-cli",
          device: 2, // LAPTOP
          sharedSecret: this.sharedSecret || "",
          chunked: true,
        });

        call.on("data", (response: any) => {
          // Handle progress signposts
          if (response.currentSignpost && showProgress) {
            const signpost = this.formatSignpost(response.currentSignpost);
            if (signpost !== lastSignpost) {
              console.log(`  ⏳ ${signpost}`);
              lastSignpost = signpost;
            }
          }

          // Handle preview images
          if (response.previewImage && showProgress) {
            const sizeMB = response.previewImage.length / (1024 * 1024);
            console.log(`  👁  Preview available (${sizeMB.toFixed(2)} MB)`);
          }

          // Handle download size projection
          if (response.downloadSize) {
            const sizeMB = response.downloadSize / (1024 * 1024);
            console.log(`  📦 Expecting ${sizeMB.toFixed(2)} MB of image data`);
          }

          // Handle remote download progress
          if (response.remoteDownload) {
            const rd = response.remoteDownload;
            const progress =
              rd.bytesExpected > 0
                ? ((rd.bytesReceived / rd.bytesExpected) * 100).toFixed(1)
                : "0";
            console.log(
              `  ⬇  Downloading models: ${progress}% (${rd.item + 1}/${
                rd.itemsExpected
              })`
            );
          }

          // Handle generated images
          if (response.generatedImages && response.generatedImages.length > 0) {
            response.generatedImages.forEach((imgData: Buffer) => {
              imageChunks.push(imgData);
            });

            // Check if this is the last chunk
            if (response.chunkState === 0) {
              // LAST_CHUNK = 0
              // Reassemble chunks
              const fullImageData = Buffer.concat(imageChunks);
              imageChunks.length = 0;

              // Save image
              const duration = ((Date.now() - startTime) / 1000).toFixed(1);
              const outputPath = this.saveImage(
                fullImageData,
                outputDir,
                prompt,
                currentImageIndex
              );

              if (outputPath) {
                console.log(`\n✓ Image saved: ${outputPath}`);
                console.log(`  Time: ${duration}s`);
                currentImageIndex++;
              } else {
                console.log(`\n✗ Failed to save image`);
                resolve(false);
              }
            }
          }
        });

        call.on("error", (error: any) => {
          console.log(chalk.red(`\n✗ Generation failed: ${error.message}`));
          resolve(false);
        });

        call.on("end", () => {
          if (currentImageIndex === 0) {
            console.log(chalk.red("\n✗ No images were generated"));
            resolve(false);
          } else {
            resolve(true);
          }
        });
      });
    } catch (error: any) {
      console.log(chalk.red(`\n✗ Generation failed: ${error.message}`));
      return false;
    }
  }

  private formatSignpost(signpost: any): string {
    if (signpost.textEncoded) return "Text encoded";
    if (signpost.imageEncoded) return "Image encoded";
    if (signpost.sampling) return `Sampling step ${signpost.sampling.step}`;
    if (signpost.imageDecoded) return "Image decoded";
    if (signpost.secondPassImageEncoded)
      return "Second pass: Image encoded";
    if (signpost.secondPassSampling)
      return `Second pass: Sampling step ${signpost.secondPassSampling.step}`;
    if (signpost.secondPassImageDecoded) return "Second pass: Image decoded";
    if (signpost.faceRestored) return "Face restoration applied";
    if (signpost.imageUpscaled) return "Image upscaled";
    return "Processing...";
  }

  private saveImage(
    imageData: Buffer,
    outputDir: string,
    prompt: string,
    index: number
  ): string | null {
    try {
      const fs = require("fs").promises;
      const pathModule = require("path");

      // Create output directory
      const dir = pathModule.resolve(outputDir);
      require("fs").mkdirSync(dir, { recursive: true });

      // Generate filename from prompt
      const safePrompt = prompt
        .replace(/[^a-zA-Z0-9 _-]/g, "_")
        .substring(0, 50)
        .trim()
        .replace(/ /g, "_");
      const timestamp = Math.floor(Date.now() / 1000);
      const filename = `${safePrompt}_${timestamp}_${index}.png`;
      const outputPath = pathModule.join(dir, filename);

      // Save the file synchronously for now
      require("fs").writeFileSync(outputPath, imageData);

      return outputPath;
    } catch (error: any) {
      console.log(chalk.yellow(`  Error saving image: ${error.message}`));
      return null;
    }
  }

  private defaultConfig(): GenerationConfig {
    return {
      model: "v1-5-pruned-emaonly.ckpt",
      start_width: 512,
      start_height: 512,
      steps: 20,
      guidance_scale: 7.5,
      sampler: "dpmpp_2m_karras",
      seed: Math.floor(Math.random() * Math.pow(2, 32)),
      batch_count: 1,
      batch_size: 1,
      clip_skip: 1,
      strength: 0.75,
    };
  }
}
