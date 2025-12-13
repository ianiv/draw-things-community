import * as fs from "fs";
import * as yaml from "js-yaml";
import { ImageGenerator, GenerationConfig } from "../imageGenerator";
import chalk from "chalk";

export async function generateCommand(
  prompt: string,
  options: {
    negativePrompt?: string;
    config?: string;
    output?: string;
    host?: string;
    port: string;
    secret?: string;
    model?: string;
    steps?: string;
    width?: string;
    height?: string;
    cfg?: string;
    sampler?: string;
    seed?: string;
    batchCount?: string;
    batchSize?: string;
  }
): Promise<void> {
  try {
    // Load config file if provided
    let configData: { [key: string]: any } = {};
    if (options.config) {
      try {
        const fileContent = fs.readFileSync(options.config, "utf-8");
        configData = (yaml.load(fileContent) as { [key: string]: any }) || {};
      } catch (error: any) {
        console.error(chalk.red(`Error loading config: ${error.message}`));
        process.exit(1);
      }
    }

    // Get server settings
    const serverConfig = configData.server || {};
    const host = options.host || serverConfig.host || "localhost";
    const port = parseInt(options.port) || serverConfig.port || 50051;
    const secret = options.secret || serverConfig.shared_secret;

    // Get generation settings
    let genConfig = (configData.generation || {}) as GenerationConfig;

    // Apply command-line overrides
    if (options.model) genConfig.model = options.model;
    if (options.steps) genConfig.steps = parseInt(options.steps);
    if (options.width) genConfig.start_width = parseInt(options.width);
    if (options.height) genConfig.start_height = parseInt(options.height);
    if (options.cfg) genConfig.guidance_scale = parseFloat(options.cfg);
    if (options.sampler) genConfig.sampler = options.sampler;
    if (options.seed) genConfig.seed = parseInt(options.seed);
    if (options.batchCount) genConfig.batch_count = parseInt(options.batchCount);
    if (options.batchSize) genConfig.batch_size = parseInt(options.batchSize);

    // Get output settings
    const outputConfig = configData.output || {};
    const outputDir = options.output || outputConfig.directory || "output";
    const showProgress = outputConfig.show_progress !== false;

    // Create client and generate
    const client = new ImageGenerator(host, port, secret);

    if (!(await client.connect())) {
      process.exit(1);
    }

    try {
      const success = await client.generateImage(
        prompt,
        options.negativePrompt || "",
        genConfig,
        outputDir,
        showProgress
      );

      process.exit(success ? 0 : 1);
    } finally {
      client.disconnect();
    }
  } catch (error: any) {
    console.error(chalk.red(`Error: ${error.message}`));
    process.exit(1);
  }
}
