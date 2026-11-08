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
