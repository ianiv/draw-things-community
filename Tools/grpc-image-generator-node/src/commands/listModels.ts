import * as fs from "fs";
import * as yaml from "js-yaml";
import { ImageGenerator } from "../imageGenerator";
import chalk from "chalk";

export async function listModelsCommand(options: {
  config?: string;
  host?: string;
  port: string;
  secret?: string;
  type?: string;
  verbose?: boolean;
}): Promise<void> {
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

    // Create client and list models
    const client = new ImageGenerator(host, port, secret);

    if (!(await client.connect())) {
      process.exit(1);
    }

    try {
      const success = await client.listModels(options.type, options.verbose || false);
      process.exit(success ? 0 : 1);
    } finally {
      client.disconnect();
    }
  } catch (error: any) {
    console.error(chalk.red(`Error: ${error.message}`));
    process.exit(1);
  }
}
