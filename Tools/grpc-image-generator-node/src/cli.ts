#!/usr/bin/env node
import { Command } from "commander";
import chalk from "chalk";
import { generateCommand } from "./commands/generate";
import { listModelsCommand } from "./commands/listModels";
import { initConfigCommand } from "./commands/initConfig";

const program = new Command();

program
  .name("draw-things")
  .description("CLI for generating images using the Draw Things gRPC API")
  .version("1.0.0");

// Generate command
program
  .command("generate <prompt>")
  .description("Generate an image from a text prompt")
  .option("-n, --negative-prompt <text>", "Negative prompt")
  .option("-c, --config <path>", "Path to configuration file")
  .option("-o, --output <dir>", "Output directory for generated images")
  .option("--host <host>", "Server hostname")
  .option("--port <port>", "Server port", "50051")
  .option("--secret <secret>", "Shared secret for authentication")
  .option("--model <name>", "Model file name")
  .option("--steps <num>", "Number of generation steps")
  .option("--width <num>", "Image width")
  .option("--height <num>", "Image height")
  .option("--cfg <num>", "Guidance scale (CFG)")
  .option("--sampler <type>", "Sampler type")
  .option("--seed <num>", "Random seed")
  .option("--batch-count <num>", "Number of batches")
  .option("--batch-size <num>", "Images per batch")
  .action(generateCommand);

// List models command
program
  .command("list-models")
  .description("List available models on the server")
  .option("-c, --config <path>", "Path to configuration file")
  .option("--host <host>", "Server hostname")
  .option("--port <port>", "Server port", "50051")
  .option("--secret <secret>", "Shared secret for authentication")
  .option(
    "-t, --type <type>",
    "Filter by model type (models, loras, controlnets, textual_inversions, upscalers, files)"
  )
  .option("-v, --verbose", "Show detailed information")
  .action(listModelsCommand);

// Init config command
program
  .command("init-config")
  .description("Create default configuration file")
  .option("-o, --output <path>", "Output path for config file", "config.yaml")
  .action(initConfigCommand);

program.parse(process.argv);

if (process.argv.length < 3) {
  program.outputHelp();
}
