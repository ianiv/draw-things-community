# Node.js CLI Setup Guide

This guide covers setup and development of the TypeScript/Node.js gRPC image generation CLI.

## Prerequisites

- Node.js 16+ (check with `node --version`)
- npm 7+ or yarn (comes with Node.js)
- Basic knowledge of TypeScript

## Installation

```bash
# Navigate to the directory
cd tools/grpc-image-generator-node

# Install dependencies
npm install

# Build TypeScript to JavaScript
npm run build

# Verify installation
node dist/cli.js --version
```

## Running Commands

### After Global Installation (Recommended)

```bash
draw-things init-config
draw-things list-models
draw-things generate "test prompt"
```

See [INSTALL.md](./INSTALL.md) for installation instructions.

### During Development

Via npm start:
```bash
npm start -- init-config
npm start -- list-models
npm start -- generate "test prompt"
```

Via compiled JavaScript:
```bash
node dist/cli.js init-config
node dist/cli.js list-models
node dist/cli.js generate "test prompt"
```

Via ts-node (fastest for development):
```bash
npm run dev -- init-config
npm run dev -- list-models
npm run dev -- generate "test prompt"
```

## Development Workflow

### Making Changes

1. Edit TypeScript files in `src/`
2. Rebuild: `npm run build`
3. Test: `npm start -- <command>`

Or use watch mode:

```bash
# Terminal 1: Watch for changes and rebuild
tsc --watch

# Terminal 2: Test your changes
npm start -- generate "test"
```

### Code Organization

- **src/cli.ts** - CLI command definitions using Commander.js
- **src/imageGenerator.ts** - Main gRPC client implementation
- **src/configBuilder.ts** - FlatBuffer configuration serialization
- **src/commands/** - Command handlers (generate, list-models, init-config)

### Key Implementation Details

#### Type Safety

The project uses strict TypeScript settings. Key type definitions:

```typescript
interface GenerationConfig {
  [key: string]: any;
}

enum SamplerType {
  DPMPP2MKarras = 0,
  EulerA = 1,
  // ...
}

class ImageGenerator {
  async connect(): Promise<boolean> { }
  async generateImage(...): Promise<boolean> { }
  async listModels(...): Promise<boolean> { }
}
```

#### gRPC Communication

Uses `@grpc/grpc-js` and `@grpc/proto-loader` to:
- Load proto files dynamically
- Create gRPC channels
- Stream responses for image generation

#### Configuration

YAML files are parsed into TypeScript objects and can be strongly typed if needed.

#### Image Handling

Images are saved synchronously with `fs.writeFileSync()`. Can be optimized to async with `fs.promises` if needed.

## Differences from Python Version

### Advantages
- Stronger type system (TypeScript)
- Faster execution in some scenarios
- Better IDE support and autocomplete
- Can be bundled for distribution

### Differences
- No external dependency on gRPC compilation (proto-loader handles it)
- Different async patterns (Promises instead of coroutines)
- Image processing library is optional (not included by default)

## Performance Optimization

### For Development
- Use ts-node for direct execution without build step
- Use watch mode for faster iteration

### For Production
- Always use compiled JavaScript (`npm run build` → `node dist/cli.js`)
- Consider bundling with esbuild or webpack for single binary

## Troubleshooting

### Module Not Found Errors

If you get `Cannot find module '@grpc/grpc-js'`:
```bash
npm install
npm run build
```

### TypeScript Errors

If you get compilation errors:
```bash
# Clear build cache and rebuild
npm run clean
npm run build
```

### gRPC Connection Issues

Enable debugging with environment variable:
```bash
DEBUG=grpc:* npm start -- list-models
```

## Testing

Current implementation doesn't include automated tests. To add testing:

```bash
npm install --save-dev jest @types/jest ts-jest
```

Then create test files in `src/__tests__/` directory.

## Building for Distribution

To create a standalone executable:

```bash
npm install --save-dev pkg

# In package.json add:
"pkg": {
  "targets": ["node16-macos-x64", "node16-linux-x64", "node16-win-x64"],
  "output": "draw-things"
}

npm run build
npx pkg dist/cli.js
```

This creates executables for different platforms.

## Next Steps

1. Run `npm install` to set up dependencies
2. Run `npm run build` to compile TypeScript
3. Run `npm start -- init-config` to create a config file
4. Run `npm start -- list-models` to test connection to server
5. Start generating images with `npm start -- generate "your prompt"`

## See Also

- [README.md](./README.md) - User guide
- [src/cli.ts](./src/cli.ts) - CLI structure
- [src/imageGenerator.ts](./src/imageGenerator.ts) - Core implementation
- [@grpc/grpc-js documentation](https://grpc.io/docs/languages/node/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
