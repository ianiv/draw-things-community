# Draw Things gRPC API Documentation

This document describes the gRPC API exposed by the Draw Things gRPC server. The server implements two main services: **ImageGenerationService** for image generation operations and **ControlPanelService** for server management.

## Quick Start

Looking to generate images right away? Check out the **[Python CLI Tool](tools/grpc-image-generator/)** - a ready-to-use command-line client for the gRPC API.

## Table of Contents

- [ImageGenerationService](#imagegenerationservice)
  - [GenerateImage](#generateimage)
  - [FilesExist](#filesexist)
  - [UploadFile](#uploadfile)
  - [Echo](#echo)
  - [Pubkey](#pubkey)
  - [Hours](#hours)
- [ControlPanelService](#controlpanelservice)
  - [ManageGPUServer](#managegpuserver)
  - [UpdateThrottlingConfig](#updatethrottlingconfig)
  - [UpdatePem](#updatepem)
  - [UpdateModelList](#updatemodellist)
  - [UpdateSharedSecret](#updatesharedsecret)
  - [UpdatePrivateKey](#updateprivatekey)
  - [UpdateComputeUnit](#updatecomputeunit)
- [Common Data Types](#common-data-types)

---

## ImageGenerationService

The ImageGenerationService handles AI image generation, file management, and server discovery operations.

### GenerateImage

Generates images based on the provided configuration and optional input image.

**RPC Definition:**
```protobuf
rpc GenerateImage(ImageGenerationRequest) returns (stream ImageGenerationResponse);
```

**Request:** `ImageGenerationRequest`

| Field | Type | Description |
|-------|------|-------------|
| `image` | bytes (optional) | Input image data as SHA256 content hash |
| `scaleFactor` | int32 | Scale factor for image generation |
| `mask` | bytes (optional) | Optional mask data as SHA256 content hash |
| `hints` | repeated HintProto | List of control hints (ControlNet, etc.) |
| `prompt` | string | Text prompt for generation |
| `negativePrompt` | string | Negative prompt to avoid certain features |
| `configuration` | bytes | Configuration data as FlatBuffer |
| `override` | MetadataOverride | Override metadata for models, LoRAs, etc. |
| `keywords` | repeated string | Keywords for remote generation |
| `user` | string | Client username |
| `device` | DeviceType | Device type (PHONE, TABLET, LAPTOP) |
| `contents` | repeated bytes | Content-addressable image data |
| `sharedSecret` | string (optional) | Authentication secret if required |
| `chunked` | bool | Enable chunked response for large images |

**Response:** `ImageGenerationResponse` (streamed)

| Field | Type | Description |
|-------|------|-------------|
| `generatedImages` | repeated bytes | Generated image data |
| `currentSignpost` | ImageGenerationSignpostProto (optional) | Current generation stage |
| `signposts` | repeated ImageGenerationSignpostProto | Collection of generation stages |
| `previewImage` | bytes (optional) | Preview of image being generated |
| `scaleFactor` | int32 (optional) | Scale factor of the generated image |
| `tags` | repeated string | Server tags for tracking |
| `downloadSize` | int64 (optional) | Size of next payload |
| `chunkState` | ChunkState | LAST_CHUNK or MORE_CHUNKS |
| `remoteDownload` | RemoteDownloadResponse (optional) | Remote download progress |

**Generation Signposts:**
The server streams progress updates through signposts that indicate different stages:
- `textEncoded` - Text prompt has been encoded
- `imageEncoded` - Input image has been encoded
- `sampling` - Diffusion sampling in progress (includes step number)
- `imageDecoded` - Image has been decoded
- `secondPassImageEncoded` - Second pass encoding (for high-res fix)
- `secondPassSampling` - Second pass sampling
- `secondPassImageDecoded` - Second pass decoding
- `faceRestored` - Face restoration applied
- `imageUpscaled` - Image upscaling applied

**Content-Addressable Storage:**
Images and tensors are referenced by their SHA256 hash. The actual content is provided in the `contents` field, allowing deduplication and efficient transfer.

---

### FilesExist

Checks which model files exist on the server and optionally returns their SHA256 hashes.

**RPC Definition:**
```protobuf
rpc FilesExist(FileListRequest) returns (FileExistenceResponse);
```

**Request:** `FileListRequest`

| Field | Type | Description |
|-------|------|-------------|
| `files` | repeated string | List of file names to check |
| `filesWithHash` | repeated string | Files to compute SHA256 hash for |
| `sharedSecret` | string (optional) | Authentication secret if required |

**Response:** `FileExistenceResponse`

| Field | Type | Description |
|-------|------|-------------|
| `files` | repeated string | Echo of requested file names |
| `existences` | repeated bool | Whether each file exists |
| `hashes` | repeated bytes | SHA256 hashes (empty if not requested) |

---

### UploadFile

Uploads model files to the server using a streaming protocol with SHA256 verification.

**RPC Definition:**
```protobuf
rpc UploadFile(stream FileUploadRequest) returns (stream UploadResponse);
```

**Request:** `FileUploadRequest` (streamed)

Each request contains one of:
- `initRequest` - Initial upload metadata
- `chunk` - File data chunk

**InitUploadRequest:**

| Field | Type | Description |
|-------|------|-------------|
| `filename` | string | Name of the file to upload |
| `sha256` | bytes | Expected SHA256 hash of the file |
| `totalSize` | int64 | Total file size in bytes |

**FileChunk:**

| Field | Type | Description |
|-------|------|-------------|
| `content` | bytes | Chunk data |
| `filename` | string | File name |
| `offset` | int64 | Byte offset in the file |

**Response:** `UploadResponse` (streamed)

| Field | Type | Description |
|-------|------|-------------|
| `chunkUploadSuccess` | bool | Whether chunk was successful |
| `receivedOffset` | int64 | Total bytes received so far |
| `message` | string | Status message |
| `filename` | string | File name |

**Upload Protocol:**
1. Client sends `InitUploadRequest` with filename, SHA256, and total size
2. Server creates temporary file (`.part` extension)
3. Client streams `FileChunk` messages in order
4. Server validates each chunk's offset
5. On completion, server verifies SHA256 and moves file to final location
6. Partial uploads are cleaned up on disconnection

---

### Echo

Discovers server capabilities and available models. Used for server health checks and metadata retrieval.

**RPC Definition:**
```protobuf
rpc Echo(EchoRequest) returns (EchoReply);
```

**Request:** `EchoRequest`

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Client identifier |
| `sharedSecret` | string (optional) | Authentication secret if required |

**Response:** `EchoReply`

| Field | Type | Description |
|-------|------|-------------|
| `message` | string | Echo response (e.g., "HELLO client") |
| `files` | repeated string | Available `.ckpt` files (if browsing enabled) |
| `override` | MetadataOverride (optional) | Available models, LoRAs, etc. |
| `sharedSecretMissing` | bool | Whether authentication is required |
| `thresholds` | ComputeUnitThreshold (optional) | Current usage limits |
| `serverIdentifier` | uint64 | Unique server identifier |

**Model Browsing:**
When enabled, the server returns all available model metadata including:
- Downloaded models (Stable Diffusion checkpoints)
- LoRA weights
- ControlNet models
- Textual inversions
- Upscaler models

---

### Pubkey

Retrieves the server's public key (currently returns empty response).

**RPC Definition:**
```protobuf
rpc Pubkey(PubkeyRequest) returns (PubkeyResponse);
```

**Request:** `PubkeyRequest`

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Client identifier |

**Response:** `PubkeyResponse`

| Field | Type | Description |
|-------|------|-------------|
| `message` | string | Status message |
| `pubkey` | string | Public key (currently empty) |

---

### Hours

Retrieves current compute unit thresholds and usage limits.

**RPC Definition:**
```protobuf
rpc Hours(HoursRequest) returns (HoursResponse);
```

**Request:** `HoursRequest`

No fields.

**Response:** `HoursResponse`

| Field | Type | Description |
|-------|------|-------------|
| `thresholds` | ComputeUnitThreshold | Current usage thresholds |

---

## ControlPanelService

The ControlPanelService provides administrative functions for managing the gRPC proxy server.

### ManageGPUServer

Adds or removes GPU servers from the proxy pool.

**RPC Definition:**
```protobuf
rpc ManageGPUServer(GPUServerRequest) returns (GPUServerResponse);
```

**Request:** `GPUServerRequest`

| Field | Type | Description |
|-------|------|-------------|
| `operation` | Operation | ADD, REMOVE, or UNSPECIFIED |
| `server_config` | ServerConfig | Server configuration details |

**ServerConfig:**

| Field | Type | Description |
|-------|------|-------------|
| `address` | string | Server IP address or hostname |
| `port` | int32 | Server port number |
| `isHighPriority` | bool | Whether server has high priority |

**Response:** `GPUServerResponse`

| Field | Type | Description |
|-------|------|-------------|
| `message` | string | Status message |

---

### UpdateThrottlingConfig

Updates rate limiting configuration for request throttling.

**RPC Definition:**
```protobuf
rpc UpdateThrottlingConfig(ThrottlingRequest) returns (ThrottlingResponse);
```

**Request:** `ThrottlingRequest`

| Field | Type | Description |
|-------|------|-------------|
| `limit_config` | map<string, int32> | Rate limits (e.g., {"request_in_5min": 10}) |

**Response:** `ThrottlingResponse`

| Field | Type | Description |
|-------|------|-------------|
| `message` | string | Status message |

---

### UpdatePem

Updates TLS certificate (PEM format).

**RPC Definition:**
```protobuf
rpc UpdatePem(UpdatePemRequest) returns (UpdatePemResponse);
```

**Request:** `UpdatePemRequest`

| Field | Type | Description |
|-------|------|-------------|
| `message` | string | PEM certificate data |

**Response:** `UpdatePemResponse`

| Field | Type | Description |
|-------|------|-------------|
| `message` | string | Status message |

---

### UpdateModelList

Updates the list of available models on the server.

**RPC Definition:**
```protobuf
rpc UpdateModelList(UpdateModelListRequest) returns (UpdateModelListResponse);
```

**Request:** `UpdateModelListRequest`

| Field | Type | Description |
|-------|------|-------------|
| `message` | string | Status or metadata |
| `files` | repeated string | List of model files |

**Response:** `UpdateModelListResponse`

| Field | Type | Description |
|-------|------|-------------|
| `message` | string | Status message |

---

### UpdateSharedSecret

Updates the shared secret for authentication.

**RPC Definition:**
```protobuf
rpc UpdateSharedSecret(UpdateSharedSecretRequest) returns (UpdateSharedSecretResponse);
```

**Request:** `UpdateSharedSecretRequest`

| Field | Type | Description |
|-------|------|-------------|
| `message` | string | New shared secret |

**Response:** `UpdateSharedSecretResponse`

| Field | Type | Description |
|-------|------|-------------|
| `message` | string | Status message |

---

### UpdatePrivateKey

Updates the server's private key.

**RPC Definition:**
```protobuf
rpc UpdatePrivateKey(UpdatePrivateKeyRequest) returns (UpdatePrivateKeyResponse);
```

**Request:** `UpdatePrivateKeyRequest`

| Field | Type | Description |
|-------|------|-------------|
| `message` | string | New private key data |

**Response:** `UpdatePrivateKeyResponse`

| Field | Type | Description |
|-------|------|-------------|
| `message` | string | Status message |

---

### UpdateComputeUnit

Updates compute unit thresholds for usage-based limits.

**RPC Definition:**
```protobuf
rpc UpdateComputeUnit(UpdateComputeUnitRequest) returns (UpdateComputeUnitResponse);
```

**Request:** `UpdateComputeUnitRequest`

| Field | Type | Description |
|-------|------|-------------|
| `cu_config` | map<string, int32> | Compute units (e.g., {"community": 15000, "plus": 40000}) |
| `expiration_timestamp` | int64 | Unix timestamp when limits expire |

**Response:** `UpdateComputeUnitResponse`

| Field | Type | Description |
|-------|------|-------------|
| `message` | string | Status message |

---

## Common Data Types

### DeviceType

Enum representing the type of client device.

| Value | Description |
|-------|-------------|
| `PHONE` (0) | Mobile phone |
| `TABLET` (1) | Tablet device |
| `LAPTOP` (2) | Laptop or desktop |

### ChunkState

Enum for chunked response state.

| Value | Description |
|-------|-------------|
| `LAST_CHUNK` (0) | Final chunk of data |
| `MORE_CHUNKS` (1) | More chunks to follow |

### MetadataOverride

Override metadata for various model types.

| Field | Type | Description |
|-------|------|-------------|
| `models` | bytes | JSON-encoded model specifications |
| `loras` | bytes | JSON-encoded LoRA specifications |
| `controlNets` | bytes | JSON-encoded ControlNet specifications |
| `textualInversions` | bytes | JSON-encoded textual inversion specs |
| `upscalers` | bytes | JSON-encoded upscaler specifications |

### ComputeUnitThreshold

Usage limits for different subscription tiers.

| Field | Type | Description |
|-------|------|-------------|
| `community` | double | Compute units for community tier |
| `plus` | double | Compute units for plus tier |
| `expireAt` | int64 | Unix timestamp when limits reset |

### HintProto

Control hint for guided image generation (ControlNet).

| Field | Type | Description |
|-------|------|-------------|
| `hintType` | string | Type of hint (e.g., "canny", "depth", "scribble") |
| `tensors` | repeated TensorAndWeight | Hint tensors with weights |

### TensorAndWeight

Tensor data with associated weight/strength.

| Field | Type | Description |
|-------|------|-------------|
| `tensor` | bytes | Tensor data as SHA256 hash |
| `weight` | float | Weight/strength value (0.0 - 1.0) |

### RemoteDownloadResponse

Progress information for remote model downloads.

| Field | Type | Description |
|-------|------|-------------|
| `bytesReceived` | int64 | Bytes downloaded so far |
| `bytesExpected` | int64 | Total bytes to download |
| `item` | int32 | Current item index |
| `itemsExpected` | int32 | Total items to download |
| `tag` | string | Download identifier |

---

## Authentication

Many endpoints support optional authentication via the `sharedSecret` field. When the server is configured with a shared secret:

1. Client includes the secret in requests
2. Server validates the secret matches its configuration
3. If validation fails, server returns `sharedSecretMissing = true` (Echo) or an error

The Echo endpoint can be used to discover if authentication is required.

---

## Implementation Notes

### Image Data Encoding

Images and tensors can be sent in two ways:

1. **Direct**: Raw bytes in the field
2. **Content-Addressable**: 32-byte SHA256 hash referencing data in the `contents` array

The server automatically detects SHA256 hashes (32 bytes) and looks them up in the provided contents.

### Compression

Image data supports compression codecs:
- **ZIP**: Standard compression
- **FPZIP**: Floating-point specific compression

When `responseCompression` is enabled on the server, responses use these codecs.

### Chunking

Large images (>4 MiB) can be chunked when `chunked = true`:
- Server sends multiple `ImageGenerationResponse` messages
- Each has `chunkState` indicating if more chunks follow
- Client reassembles chunks in order

---

## Example Usage

### Basic Image Generation

```python
# Create request
request = ImageGenerationRequest(
    prompt="a beautiful sunset over mountains",
    negativePrompt="blurry, low quality",
    scaleFactor=1,
    configuration=flatbuffer_config,  # FlatBuffer encoded config
    user="example_client",
    device=DeviceType.LAPTOP,
    chunked=True
)

# Stream responses
for response in stub.GenerateImage(request):
    if response.HasField('currentSignpost'):
        print(f"Stage: {response.currentSignpost}")
    if response.previewImage:
        display_preview(response.previewImage)
    if response.generatedImages:
        save_images(response.generatedImages)
```

### Checking File Availability

```python
request = FileListRequest(
    files=["model_v1.ckpt", "lora_style.safetensors"],
    filesWithHash=["model_v1.ckpt"]  # Also get hash for this file
)

response = stub.FilesExist(request)
for i, file in enumerate(response.files):
    exists = response.existences[i]
    hash_hex = response.hashes[i].hex() if response.hashes[i] else "N/A"
    print(f"{file}: exists={exists}, hash={hash_hex}")
```

### Uploading a Model File

```python
def upload_file(stub, filepath, filename):
    # Calculate SHA256
    with open(filepath, 'rb') as f:
        data = f.read()
        sha256 = hashlib.sha256(data).digest()

    # Send init request
    def request_generator():
        yield FileUploadRequest(
            initRequest=InitUploadRequest(
                filename=filename,
                sha256=sha256,
                totalSize=len(data)
            )
        )

        # Send chunks
        offset = 0
        chunk_size = 1024 * 1024  # 1 MiB
        while offset < len(data):
            chunk = data[offset:offset + chunk_size]
            yield FileUploadRequest(
                chunk=FileChunk(
                    content=chunk,
                    filename=filename,
                    offset=offset
                )
            )
            offset += len(chunk)

    # Upload and track progress
    for response in stub.UploadFile(request_generator()):
        if response.receivedOffset > 0:
            progress = response.receivedOffset / len(data) * 100
            print(f"Upload progress: {progress:.1f}%")
```

---

## Server Configuration

The gRPC server can be configured through the `ImageGenerationServiceImpl` class:

- `sharedSecret`: Enable authentication
- `enableModelBrowsing`: Allow clients to discover available models
- `responseCompression`: Enable compression for responses
- `usesBackupQueue`: Use backup dispatch queue for requests
- `bridgeMode`: Enable bridge mode for proxy scenarios

---

## Proto Files Location

The protocol buffer definitions are located at:
- `Libraries/GRPC/Models/Sources/imageService/imageService.proto`
- `Libraries/GRPC/Models/Sources/controlPanel/controlPanel.proto`

Generated Swift code:
- `Libraries/GRPC/Models/Sources/imageService/imageService.grpc.swift`
- `Libraries/GRPC/Models/Sources/controlPanel/controlPanel.grpc.swift`

Server implementation:
- `Libraries/GRPC/Server/Sources/ImageGenerationServiceImpl.swift`

## Client Tools

### Python CLI Tool

A complete Python command-line client is available at [`tools/grpc-image-generator/`](tools/grpc-image-generator/). Features include:

- Simple command-line interface for image generation
- YAML configuration file support
- Real-time progress display
- Automatic image saving
- Support for all generation parameters (LoRA, ControlNet, hi-res fix, etc.)

**Quick example:**
```bash
cd tools/grpc-image-generator
pip install -r requirements.txt
./draw_things_cli.py init-config
./draw_things_cli.py generate "a beautiful sunset over mountains"
```

See the [CLI tool README](tools/grpc-image-generator/README.md) for complete documentation.
