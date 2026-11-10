import ArgumentParser
import Diffusion
import Foundation
import ModelOp
import ModelZoo
import NNC

@main
struct Converter: ParsableCommand {
  @Option(
    name: .shortAndLong,
    help: "The model file that is either the safetensors or the PyTorch checkpoint.")
  var file: String
  @Option(name: .shortAndLong, help: "The name of the model.")
  var name: String
  @Option(help: "The autoencoder file.")
  var autoencoderFile: String? = nil
  @Option(name: .shortAndLong, help: "The directory to write the output files to.")
  var outputDirectory: String
  @Flag(help: "Whether to convert the text encoder(s).")
  var textEncoders = false
  @Flag(
    help:
      "Treat the input file as a Whisper-large-v3 audio encoder safetensors and convert it alone.")
  var audioEncoder = false
  @Flag(
    help:
      "Treat the input as Fibo 2.0's text_encoder directory (Qwen3-VL-4B safetensors) and convert it alone."
  )
  var fibo2TextEncoder = false
  @Flag(
    help:
      "Convert the MiniMax H3 Image VAE: --file is its safetensors file (vae/diffusion_pytorch_model.safetensors of MiniMaxAI/MiniMax-H3-Image, or ComfyUI's minimax_h3_image_vae_fp16.safetensors)."
  )
  var h3ImageVae = false

  private struct Specification: Codable {
    var name: String
    var file: String
    var version: ModelVersion
    var modifier: SamplerModifier
    var textEncoder: String?
    var autoencoder: String?
    var clipEncoder: String?
    var t5Encoder: String?
    var guidanceEmbed: Bool?
  }

  mutating func run() throws {
    ModelZoo.externalUrls = [URL(fileURLWithPath: outputDirectory)]
    if audioEncoder {
      try convertAudioEncoder()
      return
    }
    if fibo2TextEncoder {
      try convertFibo2TextEncoder()
      return
    }
    if h3ImageVae {
      try convertMiniMaxH3ImageVAE()
      return
    }
    let fileName = Importer.cleanup(filename: name)
    let importer = ModelImporter(
      filePath: file, modelName: fileName,
      isTextEncoderCustomized: textEncoders,
      autoencoderFilePath: autoencoderFile, textEncoderFilePath: nil, textEncoder2FilePath: nil)
    let (filePaths, modelVersion, modifier, inspectionResult) = try importer.import { _ in
    } progress: { _ in
    }
    let fileNames = filePaths.map { ($0 as NSString).lastPathComponent }
    var autoencoder = fileNames.first {
      $0.hasSuffix("_vae_f16.ckpt")
    }
    var clipEncoder: String? = nil
    let textEncoder: String?
    var t5Encoder: String? = nil
    switch modelVersion {
    case .v1:
      textEncoder = fileNames.first {
        $0.hasSuffix("_clip_vit_l14_f16.ckpt")
      }
    case .v2:
      textEncoder = fileNames.first {
        $0.hasSuffix("_open_clip_vit_h14_f16.ckpt")
      }
    case .sdxlBase, .ssd1b:
      clipEncoder = fileNames.first {
        $0.hasSuffix("_clip_vit_l14_f16.ckpt")
      }
      textEncoder = fileNames.first {
        $0.hasSuffix("_open_clip_vit_bigg14_f16.ckpt")
      }
      if autoencoder == nil {
        autoencoder = "sdxl_vae_v1.0_f16.ckpt"
      }
    case .sd3:
      clipEncoder = fileNames.first {
        $0.hasSuffix("_clip_vit_l14_f16.ckpt")
      }
      textEncoder = fileNames.first {
        $0.hasSuffix("_open_clip_vit_bigg14_f16.ckpt")
      }
      t5Encoder = fileNames.first {
        $0.hasSuffix("_t5_xxl_encoder_f16.ckpt")
      }
      if autoencoder == nil {
        autoencoder = "sd3_vae_f16.ckpt"
      }
    case .sd3Large:
      clipEncoder = fileNames.first {
        $0.hasSuffix("_clip_vit_l14_f16.ckpt")
      }
      textEncoder = fileNames.first {
        $0.hasSuffix("_open_clip_vit_bigg14_f16.ckpt")
      }
      t5Encoder = fileNames.first {
        $0.hasSuffix("_t5_xxl_encoder_f16.ckpt")
      }
      if autoencoder == nil {
        autoencoder = "sd3_vae_f16.ckpt"
      }
    case .pixart:
      textEncoder = fileNames.first {
        $0.hasSuffix("_t5_xxl_encoder_f16.ckpt")
      }
      if autoencoder == nil {
        autoencoder = "sdxl_vae_v1.0_f16.ckpt"
      }
    case .sdxlRefiner:
      textEncoder =
        fileNames.first {
          $0.hasSuffix("_open_clip_vit_bigg14_f16.ckpt")
        } ?? "open_clip_vit_bigg14_f16.ckpt"
      if autoencoder == nil {
        autoencoder = "sdxl_vae_v1.0_f16.ckpt"
      }
    case .svdI2v:
      textEncoder =
        fileNames.first {
          $0.hasSuffix("_open_clip_vit_h14_f16.ckpt")
        } ?? "open_clip_vit_h14_vision_model_f16.ckpt"
      if clipEncoder == nil {
        clipEncoder = "open_clip_vit_h14_visual_proj_f16.ckpt"
      }
    case .flux1:
      textEncoder = "t5_xxl_encoder_q6p.ckpt"
      clipEncoder = "clip_vit_l14_f16.ckpt"
      if autoencoder == nil {
        autoencoder = "flux_1_vae_f16.ckpt"
      }
    case .hunyuanVideo:
      textEncoder = "llava_llama_3_8b_v1.1_q8p.ckpt"
      clipEncoder = "clip_vit_l14_f16.ckpt"
      if autoencoder == nil {
        autoencoder = "hunyuan_video_vae_f16.ckpt"
      }
    case .wan21_1_3b, .wan21_14b, .wan22_5b:
      textEncoder = "umt5_xxl_encoder_q8p.ckpt"
      if modifier == .inpainting {
        clipEncoder = "open_clip_xlm_roberta_large_vit_h14_f16.ckpt"
      }
      clipEncoder = "clip_vit_l14_f16.ckpt"
      if autoencoder == nil {
        if modelVersion == .wan22_5b {
          autoencoder = "wan_v2.2_video_vae_f16.ckpt"
        } else {
          autoencoder = "wan_v2.1_video_vae_f16.ckpt"
        }
      }
    case .longcatVideoAvatar1_5:
      textEncoder = "umt5_xxl_encoder_q8p.ckpt"
      if autoencoder == nil {
        autoencoder = "wan_v2.1_video_vae_f16.ckpt"
      }
    case .minimaxH3Image:
      // The text encoder is MiniMax H3's; the VAE is converted separately with --h3-image-vae.
      textEncoder = "qwen_3_vl_32b_50_i8x.ckpt"
      if autoencoder == nil {
        autoencoder = "minimax_h3_image_vae_f16.ckpt"
      }
    case .hiDreamI1, .hiDreamO1, .minimaxH3:
      fatalError()
    case .qwenImage, .qwenImage2_1:
      fatalError()
    case .cosmos2_5_2b, .cosmos2_5_2_9b:
      textEncoder = "qwen_3_0.6b_f16.ckpt"
      clipEncoder = "\(fileName)_f16.ckpt"
      if autoencoder == nil {
        autoencoder = "qwen_image_vae_f16.ckpt"
      }
    case .zImage:
      fatalError()
    case .ernieImage:
      textEncoder =
        fileNames.first {
          $0.hasSuffix("_ministral_3_3b_f16.ckpt")
        } ?? "ministral_3_3b_f16.ckpt"
      if autoencoder == nil {
        autoencoder = "flux_2_vae_f16.ckpt"
      }
    case .ideogram4:
      textEncoder =
        fileNames.first {
          $0.hasSuffix("_qwen_3_vl_8b_instruct_f16.ckpt")
            || $0.hasSuffix("_qwen_3_vl_8b_instruct_q8p.ckpt")
        } ?? "qwen_3_vl_8b_instruct_q8p.ckpt"
      if autoencoder == nil {
        autoencoder = "flux_2_vae_f16.ckpt"
      }
    case .krea2:
      textEncoder =
        fileNames.first {
          $0.hasSuffix("_qwen_3_vl_4b_f16.ckpt") || $0.hasSuffix("_qwen_3_vl_4b_q8p.ckpt")
        } ?? "qwen_3_vl_4b_q8p.ckpt"
      clipEncoder = "\(fileName)_f16.ckpt"
      if autoencoder == nil {
        autoencoder = "qwen_image_vae_f16.ckpt"
      }
    case .seedvr2_3b, .seedvr2_7b:
      textEncoder = "\(fileName)_f16.ckpt"
      if autoencoder == nil {
        autoencoder = "seedvr2_vae_f16.ckpt"
      }
    case .fibo2:
      // Converted separately with --fibo2-text-encoder.
      textEncoder = "fibo_2_qwen_3_vl_4b_f16.ckpt"
      if autoencoder == nil {
        autoencoder = "flux_2_vae_f16.ckpt"
      }
    case .flux2, .flux2_9b, .flux2_4b, .flux3_5b, .mossImage:
      fatalError()
    case .ltx2, .ltx2_3:
      fatalError()
    case .kandinsky21, .wurstchenStageC, .wurstchenStageB, .auraflow:
      fatalError()
    }
    var specification = Specification(
      name: name, file: "\(fileName)_f16.ckpt", version: modelVersion, modifier: modifier,
      textEncoder: textEncoder, autoencoder: autoencoder, clipEncoder: clipEncoder,
      t5Encoder: t5Encoder)
    if inspectionResult.hasGuidanceEmbed {
      specification.guidanceEmbed = true
    }
    let jsonEncoder = JSONEncoder()
    jsonEncoder.keyEncodingStrategy = .convertToSnakeCase
    jsonEncoder.outputFormatting = .prettyPrinted
    let jsonData = try jsonEncoder.encode(specification)
    print(String(decoding: jsonData, as: UTF8.self))
  }

  private enum AudioEncoderError: Swift.Error {
    case cannotOpenFile
    case missingTensor(String)
  }

  private func convertAudioEncoder() throws {
    guard let safeTensors = SafeTensors(url: URL(fileURLWithPath: file)) else {
      throw AudioEncoderError.cannotOpenFile
    }
    // transformers saves WhisperForConditionalGeneration keys with a "model." prefix; the
    // WhisperModel layout has none. Normalize to the bare "encoder." layout the mapper uses.
    var stateDict = [String: TensorDescriptor]()
    for (key, value) in safeTensors.states {
      if key.hasPrefix("model.") {
        stateDict[String(key.dropFirst("model.".count))] = value
      } else {
        stateDict[key] = value
      }
    }
    let (mapper, encoder) = WhisperEncoder(
      width: 1_280, layers: 32, heads: 20, melBins: 128, frames: 3_000, intermediateSize: 5_120,
      usesFlashAttention: false)
    let graph = DynamicGraph()
    try graph.withNoGrad {
      let mel = graph.variable(.CPU, .NCHW(1, 128, 1, 3_000), of: FloatType.self)
      encoder.compile(inputs: mel)
      let mapping = mapper(.diffusers)
      let fileName = Importer.cleanup(filename: name)
      let outputPath = URL(fileURLWithPath: outputDirectory)
        .appendingPathComponent("\(fileName)_f16.ckpt").path
      try graph.openStore(outputPath) { store in
        try store.withTransaction {
          for (key, value) in mapping.sorted(by: { $0.key < $1.key }) {
            guard let descriptor = stateDict[key] else {
              throw AudioEncoderError.missingTensor(key)
            }
            try safeTensors.with(descriptor) { tensor in
              var tensor = Tensor<FloatType>(from: tensor)
              let shape = tensor.shape
              if shape.count == 3 {
                // Conv1d weights map onto [O, I, 1, W] 2D convolutions.
                tensor = tensor.reshaped(format: .NCHW, shape: [shape[0], shape[1], 1, shape[2]])
              }
              value.write(
                graph: graph, to: store, tensor: tensor, format: value.format, isDiagonalUp: false,
                isDiagonalDown: false
              ) {
                return "__audio_encoder__[\($0)]"
              }
            }
          }
        }
      }
      print("Wrote audio encoder to \(outputPath)")
    }
  }

  private enum Fibo2TextEncoderError: Error {
    case cannotOpenFile(String)
    case missingTensor(String)
  }

  /// Converts the MiniMax H3 Image VAE to the names `MiniMaxH3VideoEncoder` (the encoder is the H3
  /// video encoder) and `MiniMaxH3ImageDecoder` read. The reference rotates the two halves of the
  /// first 32 channels of each 64-wide head, the engine rotates adjacent pairs, so the query and
  /// key projections are reordered per head. The feed-forward's first projection is [gate; up].
  private func convertMiniMaxH3ImageVAE() throws {
    guard let safeTensors = SafeTensors(url: URL(fileURLWithPath: file)) else {
      throw Fibo2TextEncoderError.cannotOpenFile(file)
    }
    func read(_ key: String) throws -> Tensor<Float> {
      guard let descriptor = safeTensors.states[key] else {
        throw Fibo2TextEncoderError.missingTensor(key)
      }
      return try safeTensors.with(descriptor) { Tensor<Float>(from: $0).toCPU() }
    }
    func values(_ tensor: Tensor<Float>) -> [Float] {
      let count = tensor.shape.reduce(1, *)
      return tensor.withUnsafeBytes {
        Array(UnsafeBufferPointer(
          start: $0.baseAddress!.assumingMemoryBound(to: Float.self), count: count))
      }
    }
    func tensor(_ values: [Float], format: TensorFormat, shape: [Int]) -> Tensor<FloatType> {
      precondition(values.count == shape.reduce(1, *))
      return Tensor<FloatType>(
        from: Tensor<Float>(values, kind: .CPU, format: format, shape: TensorShape(shape)))
    }
    // Rows of a [rows, columns] matrix (or a vector), with the first 32 rows of every 64-row head
    // interleaved: rows i and i + 16 become rows 2i and 2i + 1.
    func interleave(_ values: [Float], rows: Int) -> [Float] {
      let columns = values.count / rows
      var result = values
      for row in 0..<rows {
        let head = row / 64
        let i = row % 64
        guard i < 32 else { continue }
        let target = head * 64 + (i < 16 ? i * 2 : (i - 16) * 2 + 1)
        for column in 0..<columns {
          result[target * columns + column] = values[row * columns + column]
        }
      }
      return result
    }
    let graph = DynamicGraph()
    let fileName = Importer.cleanup(filename: name)
    let outputPath = URL(fileURLWithPath: outputDirectory)
      .appendingPathComponent("\(fileName)_f16.ckpt").path
    try graph.openStore(outputPath) { store in
      try store.withTransaction {
        // Encoder: causal 3D convolutions and group norms, named by order of appearance.
        func encoder(_ name: String, _ index: Int, _ key: String, norm: Bool = false) throws {
          for (parameter, suffix) in ["weight", "bias"].enumerated() {
            let source = try read("\(key).\(suffix)")
            var shape = Array(source.shape)
            if norm {
              shape = [1, shape[0], 1, 1]
            }
            store.write(
              "__video_encoder__[t-\(name)-\(index)-\(parameter)]",
              tensor: tensor(values(source), format: .NCHW, shape: shape))
          }
        }
        try encoder("conv_in", 0, "encoder.conv_in")
        var shortcuts = 0
        for level in 0..<6 {
          for block in 0..<2 {
            let prefix = "encoder.down.\(level).block.\(block)"
            let index = level * 2 + block
            try encoder("norm1", index, "\(prefix).norm1", norm: true)
            try encoder("conv1", index, "\(prefix).conv1")
            try encoder("norm2", index, "\(prefix).norm2", norm: true)
            try encoder("conv2", index, "\(prefix).conv2")
            if safeTensors.states["\(prefix).nin_shortcut.weight"] != nil {
              try encoder("nin_shortcut", shortcuts, "\(prefix).nin_shortcut")
              shortcuts += 1
            }
          }
          if level < 4 {
            try encoder("downsample_\(level)", 0, "encoder.down.\(level).downsample.conv")
          }
        }
        try encoder("norm_out", 0, "encoder.norm_out", norm: true)
        try encoder("conv_out", 0, "encoder.conv_out")
        try encoder("quant_conv", 0, "quant_conv")
        // Decoder: a 24-layer, 1,024-wide 2D ViT.
        func decoder(_ name: String, _ index: Int, _ parameter: Int, _ values: [Float], shape: [Int])
        {
          store.write(
            "__video_decoder__[t-\(name)-\(index)-\(parameter)]",
            tensor: tensor(values, format: .NHWC, shape: shape))
        }
        func dense(_ name: String, _ index: Int, _ key: String) throws {
          let weight = try read("\(key).weight")
          let bias = try read("\(key).bias")
          decoder(name, index, 0, values(weight), shape: [weight.shape[0], weight.shape[1]])
          decoder(name, index, 1, values(bias), shape: [bias.shape[0]])
        }
        let hiddenSize = 1_024
        try dense("post_quant_conv", 0, "post_quant_conv")
        try dense("decoder_proj_in", 0, "decoder.x_embedder")
        try dense("decoder_proj_out", 0, "decoder.proj_out")
        decoder(
          "decoder_norm_out", 0, 0, values(try read("decoder.norm_out.weight")),
          shape: [1, 1, hiddenSize])
        decoder(
          "decoder_norm_out", 0, 1, values(try read("decoder.norm_out.bias")),
          shape: [1, 1, hiddenSize])
        decoder(
          "register_tokens", 0, 0, values(try read("decoder.register_tokens")),
          shape: [1, 4, hiddenSize])
        for layer in 0..<24 {
          let prefix = "decoder.transformer_blocks.\(layer)"
          let qkvWeight = values(try read("\(prefix).attn.to_qkv.weight"))
          let qkvBias = values(try read("\(prefix).attn.to_qkv.bias"))
          for (index, projection) in ["to_q", "to_k", "to_v"].enumerated() {
            var weight = Array(
              qkvWeight[(index * hiddenSize * hiddenSize)..<((index + 1) * hiddenSize * hiddenSize)])
            var bias = Array(qkvBias[(index * hiddenSize)..<((index + 1) * hiddenSize)])
            if index < 2 {
              weight = interleave(weight, rows: hiddenSize)
              bias = interleave(bias, rows: hiddenSize)
            }
            decoder(projection, layer, 0, weight, shape: [hiddenSize, hiddenSize])
            decoder(projection, layer, 1, bias, shape: [hiddenSize])
          }
          try dense("to_out", layer, "\(prefix).attn.to_out")
          let inner = 4 * hiddenSize
          let w1Weight = values(try read("\(prefix).ff.w1.weight"))
          let w1Bias = values(try read("\(prefix).ff.w1.bias"))
          decoder(
            "ff_gate", layer, 0, Array(w1Weight[0..<(inner * hiddenSize)]),
            shape: [inner, hiddenSize])
          decoder("ff_gate", layer, 1, Array(w1Bias[0..<inner]), shape: [inner])
          decoder(
            "ff_up", layer, 0, Array(w1Weight[(inner * hiddenSize)...]), shape: [inner, hiddenSize])
          decoder("ff_up", layer, 1, Array(w1Bias[inner...]), shape: [inner])
          try dense("ff_down", layer, "\(prefix).ff.w2")
          for name in ["norm1", "norm2", "scale1", "scale2"] {
            decoder(
              name, layer, 0, values(try read("\(prefix).\(name)\(name.hasPrefix("norm") ? ".weight" : "")")),
              shape: [1, 1, hiddenSize])
          }
        }
      }
    }
    print("Wrote \(outputPath)")
  }

  /// Converts Qwen3-VL-4B (text and vision) to the names Fibo 2.0's encoder reads. Hugging Face
  /// rotates the two halves of each head, the engine rotates adjacent pairs, so the query and key
  /// projections (and their norms) are reordered per head.
  private func convertFibo2TextEncoder() throws {
    let directory = URL(fileURLWithPath: file)
    let urls = try FileManager.default.contentsOfDirectory(
      at: directory, includingPropertiesForKeys: nil
    ).filter { $0.pathExtension == "safetensors" }.sorted { $0.path < $1.path }
    var sources = [String: (SafeTensors, TensorDescriptor)]()
    for url in urls {
      guard let safeTensors = SafeTensors(url: url) else {
        throw Fibo2TextEncoderError.cannotOpenFile(url.path)
      }
      for (key, value) in safeTensors.states {
        sources[key] = (safeTensors, value)
      }
    }
    func read(_ key: String) throws -> Tensor<Float> {
      guard let (safeTensors, descriptor) = sources[key] else {
        throw Fibo2TextEncoderError.missingTensor(key)
      }
      return try safeTensors.with(descriptor) { Tensor<Float>(from: $0).toCPU() }
    }
    // Rows [head * dim + i] and [head * dim + i + dim / 2] become rows 2i and 2i + 1 of the head.
    func interleave(_ tensor: Tensor<Float>, headDimension: Int) -> Tensor<Float> {
      let shape = tensor.shape
      let rows = shape[0]
      let columns = shape.count > 1 ? shape[1] : 1
      var result = Tensor<Float>(.CPU, format: .NCHW, shape: shape)
      let half = headDimension / 2
      tensor.withUnsafeBytes { source in
        let source = source.baseAddress!.assumingMemoryBound(to: Float.self)
        result.withUnsafeMutableBytes { destination in
          let destination = destination.baseAddress!.assumingMemoryBound(to: Float.self)
          for row in 0..<rows {
            let head = row / headDimension
            let i = row % headDimension
            let target = head * headDimension + (i < half ? i * 2 : (i - half) * 2 + 1)
            for column in 0..<columns {
              destination[target * columns + column] = source[row * columns + column]
            }
          }
        }
      }
      return result
    }
    func rows(_ tensor: Tensor<Float>, _ range: Range<Int>) -> Tensor<Float> {
      let shape = tensor.shape
      if shape.count == 1 {
        return tensor[range].copied()
      }
      return tensor[range, 0..<shape[1]].copied()
    }
    let graph = DynamicGraph()
    let fileName = Importer.cleanup(filename: name)
    let outputPath = URL(fileURLWithPath: outputDirectory)
      .appendingPathComponent("\(fileName)_f16.ckpt").path
    try graph.openStore(outputPath) { store in
      func write(_ name: String, _ tensor: Tensor<Float>) {
        store.write(name, tensor: Tensor<FloatType>(from: tensor))
      }
      try store.withTransaction {
        // Text model: 36 layers, 32 query heads and 8 key / value heads of 128.
        let text = "model.language_model"
        write(
          "__text_model__[t-tok_embeddings-0-0]", try read("\(text).embed_tokens.weight"))
        write("__text_model__[t-norm-0-0]", try read("\(text).norm.weight"))
        for layer in 0..<36 {
          let prefix = "\(text).layers.\(layer)"
          write(
            "__text_model__[t-q_proj-\(layer)-0]",
            interleave(try read("\(prefix).self_attn.q_proj.weight"), headDimension: 128))
          write(
            "__text_model__[t-k_proj-\(layer)-0]",
            interleave(try read("\(prefix).self_attn.k_proj.weight"), headDimension: 128))
          write("__text_model__[t-v_proj-\(layer)-0]", try read("\(prefix).self_attn.v_proj.weight"))
          write(
            "__text_model__[t-out_proj-\(layer)-0]", try read("\(prefix).self_attn.o_proj.weight"))
          write(
            "__text_model__[t-norm_q-\(layer)-0]",
            interleave(try read("\(prefix).self_attn.q_norm.weight"), headDimension: 128))
          write(
            "__text_model__[t-norm_k-\(layer)-0]",
            interleave(try read("\(prefix).self_attn.k_norm.weight"), headDimension: 128))
          write(
            "__text_model__[t-input_layernorm-\(layer)-0]",
            try read("\(prefix).input_layernorm.weight"))
          write(
            "__text_model__[t-post_attention_layernorm-\(layer)-0]",
            try read("\(prefix).post_attention_layernorm.weight"))
          // The MLP is a nested model named "mlp", so its weights carry the layer in the model part.
          write(
            "__text_model__[t-mlp-\(layer)-mlp_gate_proj-0-0]",
            try read("\(prefix).mlp.gate_proj.weight"))
          write(
            "__text_model__[t-mlp-\(layer)-mlp_up_proj-0-0]", try read("\(prefix).mlp.up_proj.weight"))
          write(
            "__text_model__[t-mlp-\(layer)-mlp_down_proj-0-0]",
            try read("\(prefix).mlp.down_proj.weight"))
        }
        // Vision tower: 24 blocks, 16 heads of 64, under its Hugging Face names.
        let visual = "model.visual"
        func vision(_ name: String, _ parameter: Int, _ tensor: Tensor<Float>) {
          write("__vision_model__[t-\(name)-0-\(parameter)]", tensor)
        }
        let patchWeight = try read("\(visual).patch_embed.proj.weight")
        vision(
          "\(visual).patch_embed.proj", 0,
          patchWeight.reshaped(format: .NCHW, shape: [patchWeight.shape[0], 3 * 2 * 16 * 16]))
        vision("\(visual).patch_embed.proj", 1, try read("\(visual).patch_embed.proj.bias"))
        write("__vision_model__[t-pos_embed-0-0]", try read("\(visual).pos_embed.weight"))
        for block in 0..<24 {
          let prefix = "\(visual).blocks.\(block)"
          let qkvWeight = try read("\(prefix).attn.qkv.weight")
          let qkvBias = try read("\(prefix).attn.qkv.bias")
          for (index, projection) in ["q_proj", "k_proj", "v_proj"].enumerated() {
            var weight = rows(qkvWeight, (index * 1_024)..<((index + 1) * 1_024))
            var bias = rows(qkvBias, (index * 1_024)..<((index + 1) * 1_024))
            if index < 2 {
              weight = interleave(weight, headDimension: 64)
              bias = interleave(bias, headDimension: 64)
            }
            vision("\(prefix).attn.\(projection)", 0, weight)
            vision("\(prefix).attn.\(projection)", 1, bias)
          }
          for name in ["attn.proj", "norm1", "norm2", "mlp.linear_fc1", "mlp.linear_fc2"] {
            vision("\(prefix).\(name)", 0, try read("\(prefix).\(name).weight"))
            vision("\(prefix).\(name)", 1, try read("\(prefix).\(name).bias"))
          }
        }
        var mergers = ["\(visual).merger"]
        mergers += (0..<3).map { "\(visual).deepstack_merger_list.\($0)" }
        for merger in mergers {
          for name in ["norm", "linear_fc1", "linear_fc2"] {
            vision("\(merger).\(name)", 0, try read("\(merger).\(name).weight"))
            vision("\(merger).\(name)", 1, try read("\(merger).\(name).bias"))
          }
        }
      }
    }
    print("Wrote Fibo 2.0 text encoder to \(outputPath)")
  }
}
