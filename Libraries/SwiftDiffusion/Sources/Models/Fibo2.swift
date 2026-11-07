import DiffusionMappings
import Foundation
import NNC

// Fibo 2.0 (Bria): a Z-Image style single-stream DiT that reads Qwen3-VL-4B. A Perceiver squeezes
// one bundle of text encoder layers into "gist" tokens that lead the stream, and five injection
// blocks cross-attend to their own bundles. Images to edit are clean context tokens that follow
// the image tokens. Reference: diffusers `BriaFibo2Transformer2DModel`.

public let Fibo2Channels = 3_584
public let Fibo2Layers = 44
public let Fibo2InjectionLayers = [6, 14, 22, 30, 38]
public let Fibo2TextBundleChannels = 7_680
public let Fibo2TextBundles = 6
public let Fibo2MaxGistCount = 512
// The Qwen3-VL hidden states (0 is the embedding) behind each bundle: the Perceiver's, then one
// per injection block.
public let Fibo2TextEncoderLayers: [[Int]] = [
  [9, 20, 31], [5, 16, 27], [7, 18, 29], [9, 20, 31], [11, 22, 33], [13, 24, 35],
]

/// Number of gist tokens for a prompt of `textLength` tokens.
public func Fibo2GistCount(textLength: Int) -> Int {
  let minGist = 256
  let maxGist = Fibo2MaxGistCount
  let gistStep = 64
  let minTextLength = 256
  let maxTextLength = 2_048
  let ratio = min(
    1.0, max(0.0, Double(textLength - minTextLength) / Double(maxTextLength - minTextLength)))
  let raw = Double(minGist) + ratio * Double(maxGist - minGist)
  let quantized = Int(ceil(raw / Double(gistStep))) * gistStep
  return max(minGist, min(maxGist, quantized))
}

/// Rotary embedding for the whole stream: gist tokens, image tokens, then each context image.
/// Gist token i sits at (1 + i, 0, 0), the image on plane `maxGistCount + 1`, and context image k
/// on the k-th plane after it, each with its own rows and columns.
public func Fibo2RotaryPositionEmbedding(
  height: Int, width: Int, gistCount: Int, referenceSizes: [(height: Int, width: Int)]
) -> Tensor<Float> {
  let referenceLength = referenceSizes.reduce(0) { $0 + $1.height * $1.width }
  let sequenceLength = gistCount + height * width + referenceLength
  var rotTensor = Tensor<Float>(.CPU, .NHWC(1, sequenceLength, 1, 128))
  let dims = [32, 48, 48]
  func setPositions(_ i: Int, _ positions: (Int, Int, Int)) {
    let values = [positions.0, positions.1, positions.2]
    var offset = 0
    for (axis, dim) in dims.enumerated() {
      for k in 0..<(dim / 2) {
        let theta = Double(values[axis]) / pow(256, Double(k) * 2 / Double(dim))
        rotTensor[0, i, 0, (offset + k) * 2] = Float(cos(theta))
        rotTensor[0, i, 0, (offset + k) * 2 + 1] = Float(sin(theta))
      }
      offset += dim / 2
    }
  }
  for i in 0..<gistCount {
    setPositions(i, (1 + i, 0, 0))
  }
  let imagePlane = Fibo2MaxGistCount + 1
  var index = gistCount
  for y in 0..<height {
    for x in 0..<width {
      setPositions(index, (imagePlane, y, x))
      index += 1
    }
  }
  for (k, referenceSize) in referenceSizes.enumerated() {
    for y in 0..<referenceSize.height {
      for x in 0..<referenceSize.width {
        setPositions(index, (imagePlane + k + 1, y, x))
        index += 1
      }
    }
  }
  return rotTensor
}

private func Fibo2ScaledDotProductAttention(
  queries: Model.IO, keys: Model.IO, values: Model.IO, b: Int, h: Int, k: Int,
  queryLength: Int, keyValueLength: Int, usesFlashAttention: FlashAttentionLevel
) -> Model.IO {
  switch usesFlashAttention {
  case .none:
    let keys = keys.transposed(1, 2)
    let queries = ((1.0 / Float(k).squareRoot()) * queries).transposed(1, 2)
    let values = values.transposed(1, 2)
    var outs = [Model.IO]()
    for i in 0..<(b * h) {
      let key = keys.reshaped(
        [1, keyValueLength, k], offset: [i, 0, 0], strides: [keyValueLength * k, k, 1])
      let query = queries.reshaped(
        [1, queryLength, k], offset: [i, 0, 0], strides: [queryLength * k, k, 1])
      let value = values.reshaped(
        [1, keyValueLength, k], offset: [i, 0, 0], strides: [keyValueLength * k, k, 1])
      var dot = Matmul(transposeB: (1, 2))(query, key)
      if let last = outs.last {
        dot.add(dependencies: [last])
      }
      dot = dot.reshaped([queryLength, keyValueLength])
      dot = dot.softmax()
      dot = dot.reshaped([1, queryLength, keyValueLength])
      outs.append(dot * value)
    }
    return Concat(axis: 0)(outs).reshaped([b, h, queryLength, k]).transposed(1, 2).reshaped([
      b, queryLength, h * k,
    ])
  case .scale1:
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    return scaledDotProductAttention((1.0 / Float(k).squareRoot()) * queries, keys, values)
      .reshaped([b, queryLength, h * k])
  case .scaleMerged, .quantized:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(), useHadamard: true,
      flags: usesFlashAttention == .quantized ? [.Int8] : [.Float16])
    return scaledDotProductAttention(queries, keys, values).reshaped([b, queryLength, h * k])
  }
}

private func Fibo2FeedForward(hiddenSize: Int, intermediateSize: Int, name: String) -> (
  Model, Model, Model, Model
) {
  let x = Input()
  let w1 = Dense(count: intermediateSize, noBias: true, flags: [.Float16], name: "\(name)_w1")
  let w3 = Dense(count: intermediateSize, noBias: true, name: "\(name)_w3")
  let w2 = Dense(count: hiddenSize, noBias: true, name: "\(name)_w2")
  // Scale the gated product down by 32 to keep FP16 in range. Callers either normalize the output
  // (scale invariant) or restore it in FP32.
  let out = w2(Functional.swishMul(value: w3(x), gate: w1(x), scale: 1.0 / 32)).to(.Float32)
  return (w1, w2, w3, Model([x], [out], name: name))
}

private func Fibo2TransformerBlock(
  prefix: String, name: String, k: Int, h: Int, b: Int, segments: [Int], textLength: Int?,
  intermediateSize: Int, usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  // Each segment (gist, image, context) has its own scale and gate: noisy for the image, clean
  // (t = 0) for the gist and the context.
  let chunks = segments.map { _ in (0..<4).map { _ in Input() } }
  let crossInputs: (keys: Input, values: Input, gate: Input)? =
    textLength != nil ? (Input(), Input(), Input()) : nil
  let t = segments.reduce(0, +)
  func modulate(_ input: Model.IO, _ index: Int) -> Model.IO {
    guard segments.count > 1 else {
      return input .* chunks[0][index]
    }
    var offset = 0
    var outs = [Model.IO]()
    for (i, segment) in segments.enumerated() {
      let segmentIn = input.reshaped(
        [b, segment, k * h], offset: [0, offset, 0], strides: [t * k * h, k * h, 1]
      ).contiguous()
      outs.append(segmentIn .* chunks[i][index])
      offset += segment
    }
    return Concat(axis: 1)(outs)
  }
  // Self-attention.
  let attentionNorm1 = RMSNorm(epsilon: 1e-5, axis: [2], name: "\(name)_attention_norm1")
  var out = modulate(attentionNorm1(x), 0).to(.Float16)
  let toqueries = Dense(count: k * h, noBias: true, flags: [.Float16], name: "\(name)_q")
  let tokeys = Dense(count: k * h, noBias: true, name: "\(name)_k")
  let tovalues = Dense(count: k * h, noBias: true, name: "\(name)_v")
  let normQ = RMSNorm(epsilon: 1e-5, axis: [3], name: "\(name)_norm_q")
  let normK = RMSNorm(epsilon: 1e-5, axis: [3], name: "\(name)_norm_k")
  var queries = normQ(toqueries(out).reshaped([b, t, h, k]))
  var keys = normK(tokeys(out).reshaped([b, t, h, k]))
  let values = tovalues(out).reshaped([b, t, h, k])
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  out = Fibo2ScaledDotProductAttention(
    queries: queries, keys: keys, values: values, b: b, h: h, k: k, queryLength: t,
    keyValueLength: t, usesFlashAttention: usesFlashAttention)
  let unifyheads = Dense(count: k * h, noBias: true, name: "\(name)_o")
  // attention_norm2 is scale invariant: scale down by 4 for FP16 and fold it into the epsilon.
  let attentionNorm2 = RMSNorm(epsilon: 1e-5 / 16, axis: [2], name: "\(name)_attention_norm2")
  var residual = x + modulate(attentionNorm2(unifyheads((1.0 / 4) * out).to(.Float32)), 1)
  // Gated cross-attention to the text encoder (injection blocks only). Keys and values come from
  // the fixed model.
  var crossMappers = [(String, Model)]()
  if let crossInputs = crossInputs, let textLength = textLength {
    let crossNorm1 = RMSNorm(epsilon: 1e-5, axis: [2], name: "\(name)_cross_attn_norm1")
    let crossToQueries = Dense(
      count: k * h, noBias: true, flags: [.Float16], name: "\(name)_cross_q")
    let crossNormQ = RMSNorm(epsilon: 1e-5, axis: [3], name: "\(name)_cross_norm_q")
    let crossQueries = crossNormQ(
      crossToQueries(crossNorm1(residual).to(.Float16)).reshaped([b, t, h, k]))
    let crossOut = Fibo2ScaledDotProductAttention(
      queries: crossQueries, keys: crossInputs.keys, values: crossInputs.values, b: b, h: h, k: k,
      queryLength: t, keyValueLength: textLength, usesFlashAttention: usesFlashAttention)
    let crossUnifyheads = Dense(count: k * h, noBias: true, name: "\(name)_cross_o")
    let crossNorm2 = RMSNorm(epsilon: 1e-5 / 16, axis: [2], name: "\(name)_cross_attn_norm2")
    residual =
      residual
      + crossInputs.gate .* crossNorm2(crossUnifyheads((1.0 / 4) * crossOut).to(.Float32))
    crossMappers = [
      ("cross_attn_norm1.weight", crossNorm1), ("cross_attention.to_q.weight", crossToQueries),
      ("cross_attention.norm_q.weight", crossNormQ),
      ("cross_attention.to_out.0.weight", crossUnifyheads),
      ("cross_attn_norm2.weight", crossNorm2),
    ]
  }
  // Feed forward.
  let ffnNorm1 = RMSNorm(epsilon: 1e-5, axis: [2], name: "\(name)_ffn_norm1")
  let (w1, w2, w3, ffn) = Fibo2FeedForward(
    hiddenSize: k * h, intermediateSize: intermediateSize, name: "\(name)_ffn")
  // ffn_norm2 is scale invariant, the FFN output is scaled down by 32.
  let ffnNorm2 = RMSNorm(epsilon: 1e-5 / 1_024, axis: [2], name: "\(name)_ffn_norm2")
  out = residual + modulate(ffnNorm2(ffn(modulate(ffnNorm1(residual), 2).to(.Float16))), 3)
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).attention.to_q.weight"] = [toqueries.weight.name]
    mapping["\(prefix).attention.to_k.weight"] = [tokeys.weight.name]
    mapping["\(prefix).attention.to_v.weight"] = [tovalues.weight.name]
    mapping["\(prefix).attention.to_out.0.weight"] = [unifyheads.weight.name]
    mapping["\(prefix).attention.norm_q.weight"] = [normQ.weight.name]
    mapping["\(prefix).attention.norm_k.weight"] = [normK.weight.name]
    mapping["\(prefix).attention_norm1.weight"] = [attentionNorm1.weight.name]
    mapping["\(prefix).attention_norm2.weight"] = [attentionNorm2.weight.name]
    mapping["\(prefix).ffn_norm1.weight"] = [ffnNorm1.weight.name]
    mapping["\(prefix).ffn_norm2.weight"] = [ffnNorm2.weight.name]
    mapping["\(prefix).feed_forward.w1.weight"] = [w1.weight.name]
    mapping["\(prefix).feed_forward.w2.weight"] = [w2.weight.name]
    mapping["\(prefix).feed_forward.w3.weight"] = [w3.weight.name]
    for (key, model) in crossMappers {
      mapping["\(prefix).\(key)"] = [model.weight.name]
    }
    return mapping
  }
  var inputs: [Input] = [x, rot] + chunks.flatMap { $0 }
  if let crossInputs = crossInputs {
    inputs += [crossInputs.keys, crossInputs.values, crossInputs.gate]
  }
  return (mapper, Model(inputs, [out]))
}

/// The denoiser. Inputs, in order: latents, rotary, gist tokens, keys and values for each
/// injection block, their gates, the patchified context latents (when there are images to edit),
/// the clean chunks of every block, then the noisy chunks of every block and the final scale.
public func Fibo2(
  batchSize: Int, height: Int, width: Int, textLength: Int, gistCount: Int,
  referenceSequenceLength: Int, channels: Int, layers: Int, injectionLayers: [Int],
  usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  let gist = Input()
  let crossKeyValues = injectionLayers.map { _ in (Input(), Input()) }
  let crossGates = injectionLayers.map { _ in Input() }
  let reference: Input? = referenceSequenceLength > 0 ? Input() : nil
  let blockCount = 2 + layers
  let cleanChunks = (0..<blockCount).map { _ in (0..<4).map { _ in Input() } }
  let noisyChunks = (0..<blockCount).map { _ in (0..<4).map { _ in Input() } }
  let finalScale = Input()
  let h = height / 2
  let w = width / 2
  let imageLength = h * w
  let k = 128
  let heads = channels / k
  let intermediateSize = (channels * 8 / 3) / 64 * 64
  // 2x2 latent patches become tokens, features ordered (channel, dy, dx).
  var tokens = x.reshaped([batchSize, h, 2, w, 2, 32]).permuted(0, 1, 3, 5, 2, 4).contiguous()
    .reshaped([batchSize, imageLength, 32 * 2 * 2], format: .NHWC)
  if let reference = reference {
    tokens = Functional.concat(axis: 1, tokens, reference)
  }
  let xEmbedder = Dense(count: channels, name: "x_embedder")
  var out = xEmbedder(tokens).to(.Float32)
  let sequenceLength = gistCount + imageLength + referenceSequenceLength
  let rotResized = rot.reshaped(.NHWC(1, sequenceLength, 1, 128))
  let refinerRot = rotResized.reshaped(
    [1, imageLength + referenceSequenceLength, 1, 128], offset: [0, gistCount, 0, 0],
    strides: [sequenceLength * 128, 128, 128, 1]
  ).contiguous()
  var mappers = [ModelWeightMapper]()
  // Noise refiner: the image tokens are noisy, the context tokens clean.
  let refinerSegments =
    referenceSequenceLength > 0 ? [imageLength, referenceSequenceLength] : [imageLength]
  for i in 0..<2 {
    let (mapper, block) = Fibo2TransformerBlock(
      prefix: "noise_refiner.\(i)", name: "noise_refiner", k: k, h: heads, b: batchSize,
      segments: refinerSegments, textLength: nil, intermediateSize: intermediateSize,
      usesFlashAttention: usesFlashAttention)
    let chunks = referenceSequenceLength > 0 ? noisyChunks[i] + cleanChunks[i] : noisyChunks[i]
    out = block([out, refinerRot] + chunks)
    mappers.append(mapper)
  }
  // One stream: gist tokens first, then the image tokens, then the context tokens.
  out = Functional.concat(axis: 1, gist.to(.Float32), out)
  let segments =
    referenceSequenceLength > 0
    ? [gistCount, imageLength, referenceSequenceLength] : [gistCount, imageLength]
  for i in 0..<layers {
    let injectionIndex = injectionLayers.firstIndex(of: i)
    let (mapper, block) = Fibo2TransformerBlock(
      prefix: "layers.\(i)", name: "", k: k, h: heads, b: batchSize, segments: segments,
      textLength: injectionIndex != nil ? textLength : nil, intermediateSize: intermediateSize,
      usesFlashAttention: usesFlashAttention)
    var chunks = cleanChunks[i + 2] + noisyChunks[i + 2]
    if referenceSequenceLength > 0 {
      chunks += cleanChunks[i + 2]
    }
    var blockInputs: [Model.IO] = [out, rotResized] + chunks
    if let injectionIndex = injectionIndex {
      blockInputs += [
        crossKeyValues[injectionIndex].0, crossKeyValues[injectionIndex].1,
        crossGates[injectionIndex],
      ]
    }
    out = block(blockInputs)
    mappers.append(mapper)
  }
  // Final layer on the image tokens only, then unpatchify.
  out = out.reshaped(
    [batchSize, imageLength, channels], offset: [0, gistCount, 0],
    strides: [sequenceLength * channels, channels, 1]
  ).contiguous()
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = (normFinal(out) .* finalScale).to(.Float16)
  let projOut = Dense(count: 2 * 2 * 32, name: "final_linear")
  out = projOut(out).reshaped([batchSize, h, w, 32, 2, 2]).permuted(0, 1, 4, 2, 5, 3)
    .contiguous().reshaped([batchSize, h * 2, w * 2, 32])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["x_embedder.weight"] = [xEmbedder.weight.name]
    mapping["x_embedder.bias"] = [xEmbedder.bias.name]
    mapping["final_layer.linear.weight"] = [projOut.weight.name]
    mapping["final_layer.linear.bias"] = [projOut.bias.name]
    return mapping
  }
  var inputs: [Input] = [x, rot, gist]
  inputs += crossKeyValues.flatMap { [$0.0, $0.1] }
  inputs += crossGates
  if let reference = reference {
    inputs.append(reference)
  }
  inputs += cleanChunks.flatMap { $0 }
  inputs += noisyChunks.flatMap { $0 }
  inputs.append(finalScale)
  return (mapper, Model(inputs, [out]))
}

/// Number of trailing conditions that depend on the timestep (noisy chunks and the final scale).
public func Fibo2NoisyConditionCount(layers: Int) -> Int {
  return (2 + layers) * 4 + 1
}

private func Fibo2PerceiverLayer(
  prefix: String, k: Int, h: Int, gistCount: Int, textLength: Int, intermediateSize: Int,
  usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let latents = Input()
  let text = Input()
  let channels = k * h
  // Gist tokens read the text encoder.
  let crossNorm = RMSNorm(epsilon: 1e-5, axis: [2], name: "perceiver_cross_attn_norm")
  let crossToQueries = Dense(
    count: channels, noBias: true, flags: [.Float16], name: "perceiver_cross_q")
  let crossToKeys = Dense(count: channels, noBias: true, name: "perceiver_cross_k")
  let crossToValues = Dense(count: channels, noBias: true, name: "perceiver_cross_v")
  let crossNormQ = RMSNorm(epsilon: 1e-5, axis: [3], name: "perceiver_cross_norm_q")
  let crossNormK = RMSNorm(epsilon: 1e-5, axis: [3], name: "perceiver_cross_norm_k")
  let crossUnifyheads = Dense(count: channels, noBias: true, name: "perceiver_cross_o")
  var out = Fibo2ScaledDotProductAttention(
    queries: crossNormQ(
      crossToQueries(crossNorm(latents).to(.Float16)).reshaped([1, gistCount, h, k])),
    keys: crossNormK(crossToKeys(text).reshaped([1, textLength, h, k])),
    values: crossToValues(text).reshaped([1, textLength, h, k]), b: 1, h: h, k: k,
    queryLength: gistCount, keyValueLength: textLength, usesFlashAttention: usesFlashAttention)
  var residual = latents + crossUnifyheads(out).to(.Float32)
  // Then each other.
  let selfNorm = RMSNorm(epsilon: 1e-5, axis: [2], name: "perceiver_self_attn_norm")
  let selfToQueries = Dense(
    count: channels, noBias: true, flags: [.Float16], name: "perceiver_self_q")
  let selfToKeys = Dense(count: channels, noBias: true, name: "perceiver_self_k")
  let selfToValues = Dense(count: channels, noBias: true, name: "perceiver_self_v")
  let selfNormQ = RMSNorm(epsilon: 1e-5, axis: [3], name: "perceiver_self_norm_q")
  let selfNormK = RMSNorm(epsilon: 1e-5, axis: [3], name: "perceiver_self_norm_k")
  let selfUnifyheads = Dense(count: channels, noBias: true, name: "perceiver_self_o")
  out = selfNorm(residual).to(.Float16)
  out = Fibo2ScaledDotProductAttention(
    queries: selfNormQ(selfToQueries(out).reshaped([1, gistCount, h, k])),
    keys: selfNormK(selfToKeys(out).reshaped([1, gistCount, h, k])),
    values: selfToValues(out).reshaped([1, gistCount, h, k]), b: 1, h: h, k: k,
    queryLength: gistCount, keyValueLength: gistCount, usesFlashAttention: usesFlashAttention)
  residual = residual + selfUnifyheads(out).to(.Float32)
  // Then an FFN, its output restored in FP32.
  let ffnNorm = RMSNorm(epsilon: 1e-5, axis: [2], name: "perceiver_ffn_norm")
  let (w1, w2, w3, ffn) = Fibo2FeedForward(
    hiddenSize: channels, intermediateSize: intermediateSize, name: "perceiver_ffn")
  out = residual + 32 * ffn(ffnNorm(residual).to(.Float16))
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).cross_attn_norm.weight"] = [crossNorm.weight.name]
    mapping["\(prefix).cross_attention.to_q.weight"] = [crossToQueries.weight.name]
    mapping["\(prefix).cross_attention.to_k.weight"] = [crossToKeys.weight.name]
    mapping["\(prefix).cross_attention.to_v.weight"] = [crossToValues.weight.name]
    mapping["\(prefix).cross_attention.to_out.0.weight"] = [crossUnifyheads.weight.name]
    mapping["\(prefix).cross_attention.norm_q.weight"] = [crossNormQ.weight.name]
    mapping["\(prefix).cross_attention.norm_k.weight"] = [crossNormK.weight.name]
    mapping["\(prefix).self_attn_norm.weight"] = [selfNorm.weight.name]
    mapping["\(prefix).self_attention.to_q.weight"] = [selfToQueries.weight.name]
    mapping["\(prefix).self_attention.to_k.weight"] = [selfToKeys.weight.name]
    mapping["\(prefix).self_attention.to_v.weight"] = [selfToValues.weight.name]
    mapping["\(prefix).self_attention.to_out.0.weight"] = [selfUnifyheads.weight.name]
    mapping["\(prefix).self_attention.norm_q.weight"] = [selfNormQ.weight.name]
    mapping["\(prefix).self_attention.norm_k.weight"] = [selfNormK.weight.name]
    mapping["\(prefix).ffn_norm.weight"] = [ffnNorm.weight.name]
    mapping["\(prefix).ffn.w1.weight"] = [w1.weight.name]
    mapping["\(prefix).ffn.w2.weight"] = [w2.weight.name]
    mapping["\(prefix).ffn.w3.weight"] = [w3.weight.name]
    return mapping
  }
  return (mapper, Model([latents, text], [out]))
}

/// The timestep and text dependent parts, computed once per generation. Inputs: the six text
/// bundles concatenated on channels, the timestep embeddings (one row per step, then a row for
/// t = 0), then each image to edit as latents. Outputs follow the denoiser's input order, after
/// latents and rotary.
public func Fibo2Fixed<FloatType: TensorNumeric & BinaryFloatingPoint>(
  _ dataType: FloatType.Type, timesteps: Int, textLength: Int, gistCount: Int,
  referenceSizes: [(height: Int, width: Int)], channels: Int, layers: Int,
  injectionLayers: [Int], usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let contextIn = Input()
  let t = Input()
  let referenceImages = referenceSizes.map { _ in Input() }
  let k = 128
  let heads = channels / k
  let intermediateSize = (channels * 8 / 3) / 64 * 64
  let totalChannels = Fibo2TextBundles * Fibo2TextBundleChannels
  func bundle(_ index: Int) -> Model.IO {
    return contextIn.reshaped(
      [1, textLength, Fibo2TextBundleChannels], offset: [0, 0, index * Fibo2TextBundleChannels],
      strides: [textLength * totalChannels, totalChannels, 1]
    ).contiguous()
  }
  // Perceiver: the first bundle becomes the gist tokens.
  let latentQueries = Parameter<FloatType>(
    .GPU(0), .WC(Fibo2MaxGistCount, channels), name: "latent_queries")
  var gist = latentQueries.reshaped(
    [1, gistCount, channels], offset: [0, 0, 0], strides: [gistCount * channels, channels, 1]
  ).to(.Float32)
  var mappers = [ModelWeightMapper]()
  let perceiverText = bundle(0)
  for i in 0..<2 {
    let (mapper, layer) = Fibo2PerceiverLayer(
      prefix: "perceiver.layers.\(i)", k: k, h: heads, gistCount: gistCount,
      textLength: textLength, intermediateSize: intermediateSize,
      usesFlashAttention: usesFlashAttention)
    gist = layer(gist, perceiverText)
    mappers.append(mapper)
  }
  let perceiverNormOut = RMSNorm(epsilon: 1e-5, axis: [2], name: "perceiver_norm_out")
  gist = perceiverNormOut(gist).to(.Float16)
  var outputs: [Model.IO] = [gist]
  // Keys and values of each injection block's cross-attention.
  var crossGates = [Model.IO]()
  for (index, layer) in injectionLayers.enumerated() {
    let text = bundle(index + 1)
    let toKeys = Dense(count: channels, noBias: true, name: "cross_k")
    let toValues = Dense(count: channels, noBias: true, name: "cross_v")
    let normK = RMSNorm(epsilon: 1e-5, axis: [3], name: "cross_norm_k")
    outputs.append(normK(toKeys(text).reshaped([1, textLength, heads, k])))
    outputs.append(toValues(text).reshaped([1, textLength, heads, k]))
    let gate = Parameter<FloatType>(.GPU(0), .C(1), name: "cross_attn_gate")
    crossGates.append(gate.reshaped([1, 1, 1]).tanh().to(.Float32))
    mappers.append { _ in
      var mapping = ModelWeightMapping()
      mapping["layers.\(layer).cross_attention.to_k.weight"] = [toKeys.weight.name]
      mapping["layers.\(layer).cross_attention.to_v.weight"] = [toValues.weight.name]
      mapping["layers.\(layer).cross_attention.norm_k.weight"] = [normK.weight.name]
      mapping["layers.\(layer).cross_attn_gate"] = [gate.weight.name]
      return mapping
    }
  }
  outputs += crossGates
  // The images to edit, patchified the same way as the latents.
  if !referenceSizes.isEmpty {
    let references = zip(referenceImages, referenceSizes).map { input, size in
      let h = size.height / 2
      let w = size.width / 2
      return input.reshaped([1, h, 2, w, 2, 32]).permuted(0, 1, 3, 5, 2, 4).contiguous()
        .reshaped([1, h * w, 32 * 2 * 2], format: .NHWC)
    }
    outputs.append(references.count > 1 ? Concat(axis: 1)(references) : references[0])
  }
  // Timestep embedding: the noisy rows, then t = 0 for the gist and context tokens.
  let timeInMlp0 = Dense(count: 1_024, name: "t_embedder_0")
  let timeInMlp2 = Dense(count: 256, name: "t_embedder_1")
  let tEmbed = timeInMlp2(timeInMlp0(t).swish())
  let noisyEmbed = tEmbed.reshaped([timesteps, 1, 256], strides: [256, 256, 1]).contiguous()
  let cleanEmbed = tEmbed.reshaped([1, 1, 256], offset: [timesteps, 0, 0], strides: [256, 256, 1])
    .contiguous()
  var cleanChunks = [Model.IO]()
  var noisyChunks = [Model.IO]()
  for i in 0..<(2 + layers) {
    let prefix = i < 2 ? "noise_refiner.\(i)" : "layers.\(i - 2)"
    let adaLNs = (0..<4).map {
      Dense(count: channels, name: i < 2 ? "refiner_ada_ln_\($0)" : "ada_ln_\($0)")
    }
    func chunks(_ embed: Model.IO) -> [Model.IO] {
      var chunks = adaLNs.map { $0(embed) }
      chunks[0] = (1 + chunks[0]).to(.Float32)
      chunks[1] = chunks[1].tanh().to(.Float32)
      chunks[2] = (1 + chunks[2]).to(.Float32)
      chunks[3] = chunks[3].tanh().to(.Float32)
      return chunks
    }
    cleanChunks += chunks(cleanEmbed)
    noisyChunks += chunks(noisyEmbed)
    mappers.append { _ in
      var mapping = ModelWeightMapping()
      mapping["\(prefix).adaLN_modulation.0.weight"] = ModelWeightElement(
        adaLNs.map { $0.weight.name })
      mapping["\(prefix).adaLN_modulation.0.bias"] = ModelWeightElement(
        adaLNs.map { $0.bias.name })
      return mapping
    }
  }
  let finalScale = Dense(count: channels, name: "ada_ln_final")
  outputs += cleanChunks
  outputs += noisyChunks
  outputs.append(1 + finalScale(noisyEmbed.swish()).to(.Float32))
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["perceiver.latent_queries"] = [latentQueries.weight.name]
    mapping["perceiver.norm_out.weight"] = [perceiverNormOut.weight.name]
    mapping["t_embedder.mlp.0.weight"] = [timeInMlp0.weight.name]
    mapping["t_embedder.mlp.0.bias"] = [timeInMlp0.bias.name]
    mapping["t_embedder.mlp.2.weight"] = [timeInMlp2.weight.name]
    mapping["t_embedder.mlp.2.bias"] = [timeInMlp2.bias.name]
    mapping["final_layer.adaLN_modulation.1.weight"] = [finalScale.weight.name]
    mapping["final_layer.adaLN_modulation.1.bias"] = [finalScale.bias.name]
    return mapping
  }
  return (mapper, Model([contextIn, t] + referenceImages, outputs))
}

/// Shapes of `Fibo2Fixed`'s outputs, for importers that build the denoiser without running it.
public func Fibo2FixedOutputShapes(
  timesteps: Int, textLength: Int, gistCount: Int, referenceSequenceLength: Int, channels: Int,
  layers: Int, injectionLayers: [Int]
) -> [TensorShape] {
  let heads = channels / 128
  var outs = [TensorShape([1, gistCount, channels])]
  for _ in injectionLayers {
    outs.append(TensorShape([1, textLength, heads, 128]))
    outs.append(TensorShape([1, textLength, heads, 128]))
  }
  for _ in injectionLayers {
    outs.append(TensorShape([1, 1, 1]))
  }
  if referenceSequenceLength > 0 {
    outs.append(TensorShape([1, referenceSequenceLength, 2 * 2 * 32]))
  }
  for _ in 0..<((2 + layers) * 4) {
    outs.append(TensorShape([1, 1, channels]))
  }
  for _ in 0..<((2 + layers) * 4 + 1) {
    outs.append(TensorShape([timesteps, 1, channels]))
  }
  return outs
}

/// Removes whitespace outside of strings, so a JSON prompt reads the way Fibo 2.0 was trained
/// (compact separators), keeping its keys in the order they were written.
public func Fibo2CompactJSON(_ text: String) -> String {
  var result = ""
  result.reserveCapacity(text.count)
  var inString = false
  var escaped = false
  for character in text {
    if inString {
      result.append(character)
      if escaped {
        escaped = false
      } else if character == "\\" {
        escaped = true
      } else if character == "\"" {
        inString = false
      }
    } else if character == "\"" {
      inString = true
      result.append(character)
    } else if !character.isWhitespace {
      result.append(character)
    }
  }
  return result
}

/// The prompt Qwen3-VL reads, before the chat template. An edit is one JSON object: a vision marker
/// per image to edit, then the instruction. A prompt that is already a JSON object with an
/// "edit_instruction" key keeps its own keys (such as "structured_caption") after the markers. A
/// generation without images takes the prompt as it is (a JSON caption is made compact).
public func Fibo2Prompt(_ text: String, images: Int) -> String {
  let marker = "<|vision_start|><|image_pad|><|vision_end|>"
  let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
  let isJSONObject: Bool = {
    guard trimmed.hasPrefix("{"), trimmed.hasSuffix("}"), let data = trimmed.data(using: .utf8)
    else { return false }
    return (try? JSONSerialization.jsonObject(with: data)) is [String: Any]
  }()
  guard images > 0 else {
    return isJSONObject ? Fibo2CompactJSON(trimmed) : text
  }
  let markers =
    images == 1
    ? "\"image\":\"\(marker)\""
    : (1...images).map { "\"image_\($0)\":\"\(marker)\"" }.joined(separator: ",")
  if isJSONObject, let data = trimmed.data(using: .utf8),
    let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
    object["edit_instruction"] != nil
  {
    let compact = Fibo2CompactJSON(trimmed)
    let body = compact.dropFirst().dropLast()
    return "{" + markers + (body.isEmpty ? "" : "," + body) + "}"
  }
  let encoder = JSONEncoder()
  encoder.outputFormatting = [.withoutEscapingSlashes]
  let instruction =
    (try? encoder.encode(text)).flatMap { String(data: $0, encoding: .utf8) } ?? "\"\""
  return "{" + markers + ",\"edit_instruction\":" + instruction + "}"
}
