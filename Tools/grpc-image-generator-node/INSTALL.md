# Installation Guide

This guide covers different ways to install the Draw Things gRPC CLI.

## Option 1: Install from Local Source (Recommended for Development)

For development or local testing, you can install directly from the repository.

### Prerequisites
- Node.js 16+
- npm 7+

### Steps

```bash
# Navigate to the tool directory
cd tools/grpc-image-generator-node

# Install dependencies
npm install

# Build TypeScript
npm run build

# Install globally from source
npm install -g .
```

After installation, you can use the CLI directly:

```bash
draw-things init-config
draw-things list-models
draw-things generate "a beautiful sunset"
```

### Uninstall

```bash
npm uninstall -g draw-things-grpc-cli
```

## Option 2: Local Installation (for current project only)

If you want to use it only within the project:

```bash
cd tools/grpc-image-generator-node

npm install
npm run build

# Use with npx
npx draw-things init-config
npx draw-things list-models
npx draw-things generate "test"
```

## Option 3: Create a Symlink (Development Workflow)

For active development, create a symlink instead:

```bash
cd tools/grpc-image-generator-node

npm install
npm run build

# Create global symlink
npm link

# Now you can use draw-things directly
draw-things init-config

# To unlink later
npm unlink -g draw-things-grpc-cli
```

This is useful if you're making frequent changes, as you can rebuild and test immediately.

## Verifying Installation

After installation, verify the CLI works:

```bash
# Check version
draw-things --version

# Check help
draw-things --help

# List available commands
draw-things

# Test connection to server
draw-things list-models
```

## Configuration File Location

Configuration files are created in the current working directory. We recommend creating a dedicated directory for your generations:

```bash
# Create a project directory
mkdir my-generations
cd my-generations

# Initialize config
draw-things init-config

# Start generating
draw-things generate "your prompt"

# Images will be saved in ./output/
```

## Using the CLI

Once installed globally, the command syntax is:

```bash
draw-things <command> [options]
```

### Quick Examples

```bash
# Initialize configuration
draw-things init-config

# Create custom config
draw-things init-config -o my-settings.yaml

# List all available models
draw-things list-models

# List only LoRAs
draw-things list-models --type loras

# Generate an image
draw-things generate "a beautiful sunset over mountains"

# Generate with custom config
draw-things generate "landscape" -c my-settings.yaml

# Override settings via command line
draw-things generate "portrait" --steps 40 --width 768 --height 768

# Use remote server
draw-things generate "test" --host 192.168.1.100 --port 50051

# With authentication
draw-things generate "test" --secret my-secret-key
```

## Troubleshooting

### Command not found

If you get `command not found: draw-things`, try:

```bash
# Verify installation
npm list -g draw-things-grpc-cli

# Check npm global bin directory
npm config get prefix

# Try using the full path
$(npm config get prefix)/bin/draw-things init-config
```

### Permission denied

If you get permission errors on Linux/Mac:

```bash
# Use sudo
sudo npm install -g .

# Or fix npm permissions
mkdir ~/.npm-global
npm config set prefix '~/.npm-global'
export PATH=~/.npm-global/bin:$PATH
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
```

### Module not found errors

If you get errors about missing modules:

```bash
cd tools/grpc-image-generator-node
npm install
npm run build
npm install -g .
```

### Windows Users

On Windows, the command should work the same way. If not, try:

```bash
# Using node directly
node "C:\path\to\npm\global\draw-things" init-config

# Or use the .cmd wrapper if it exists
draw-things.cmd init-config
```

## Upgrading

To upgrade to a newer version:

```bash
cd tools/grpc-image-generator-node

# Pull latest code
git pull

# Rebuild
npm run build

# Reinstall globally
npm install -g .
```

## Development Workflow

If you're developing the CLI:

```bash
cd tools/grpc-image-generator-node

# Install dependencies
npm install

# Create global symlink
npm link

# Now make changes and rebuild
npm run build

# Test immediately without reinstalling
draw-things generate "test"

# When done, unlink
npm unlink -g
```

## Installation Locations

When you run `npm install -g`, the files are installed to npm's global directory:

- **Linux/Mac**: `~/.npm-global/` or `/usr/local/lib/node_modules/`
- **Windows**: `%APPDATA%\npm\node_modules\`

The executable symlink is placed in:
- **Linux/Mac**: `~/.npm-global/bin/` or `/usr/local/bin/`
- **Windows**: `%APPDATA%\npm\`

You can find this with:
```bash
npm config get prefix
```

## Using with npm scripts

You can also use it in npm scripts within other projects:

```json
{
  "scripts": {
    "generate-image": "draw-things generate",
    "list-models": "draw-things list-models",
    "setup": "draw-things init-config"
  }
}
```

Then run with `npm run generate-image "your prompt"`.

## Docker (Optional)

For reproducible environments, you can create a Dockerfile:

```dockerfile
FROM node:18-alpine

WORKDIR /app

# Copy the CLI
COPY tools/grpc-image-generator-node /app

# Build and install
RUN npm install && npm run build && npm install -g .

ENTRYPOINT ["draw-things"]
```

Build and use:
```bash
docker build -t draw-things .
docker run -it -v $(pwd):/output draw-things generate "test"
```

## See Also

- [README.md](./README.md) - User guide
- [SETUP.md](./SETUP.md) - Development guide
- [../grpc-image-generator/](../grpc-image-generator/) - Python version
