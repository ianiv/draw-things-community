import Foundation
import NNC
import WeightsCache

/// ComfyUI's `dpmpp_2m_sde`, `res_multistep`, `heun` and `er_sde` (comfy/k_diffusion/sampling.py)
/// on a `Denoiser.Scheduler`'s sigmas. The step arithmetic is `ComfySamplerSteps`; this runs the
/// model the way `DPMPP2MSampler` does (conditions, controls, CFG through `applyCfg`, refiner
/// switch, inpainting) and adds up the tensors it says. Heun evaluates twice a step: the fixed
/// encoder and controls get each step's two timesteps interleaved, as `DPMPPSDESampler` does with
/// its midpoints, and MiniMax H3 Image's output heads for both are the step's own interval
/// (`headIntervals`). The SDE samplers draw `randn` noise from the seeded graph, as `EulerASampler`
/// does (ComfyUI's `dpmpp_2m_sde` uses a Brownian tree; the noise has the same distribution).
/// MiniMax H3 video and Würstchen never get here (`LocalImageGenerator.runsComfySamplers`).
public struct ComfyMultistepSampler<
  FloatType: TensorNumeric & BinaryFloatingPoint, UNet: UNetProtocol
>
where UNet.FloatType == FloatType {
  public typealias Discretization = Denoiser.ScheduledDiscretization
  public let kind: Denoiser.ComfySampler
  public let filePath: String
  public let modifier: SamplerModifier
  public let version: ModelVersion
  public let audioShiftRatio: Float
  public let qkNorm: Bool
  public let dualAttentionLayers: [Int]
  public let distilledGuidanceLayers: Int
  public let activationQkScaling: [Int: Int]
  public let activationProjScaling: [Int: Int]
  public let activationFfnProjUpScaling: [Int: Int]
  public let activationFfnScaling: [Int: Int]
  public let usesFlashAttention: UseFlashAttention
  public let usesSolAttention: Bool
  public let solAttentionStart: Int
  public let solAttentionTau: Float
  public let upcastAttention: Bool
  public let externalOnDemand: Bool
  public let injectControls: Bool
  public let injectT2IAdapters: Bool
  public let injectAttentionKV: Bool

  public let injectIPAdapterLengths: [Int]
  public let lora: [LoRAConfiguration]
  public let classifierFreeGuidance: Bool
  public let isGuidanceEmbedEnabled: Bool
  public let isQuantizedModel: Bool
  public let canRunLoRASeparately: Bool
  public let deviceProperties: DeviceProperties
  public let conditioning: Denoiser.Conditioning
  public let tiledDiffusion: TiledConfiguration
  public let teaCache: TeaCacheConfiguration
  public let causalInference: (Int, pad: Int)
  public let cfgZeroStar: CfgZeroStarConfiguration
  public let isBF16: Bool
  private let discretization: Discretization
  private let weightsCache: WeightsCache
  public init(
    kind: Denoiser.ComfySampler, filePath: String, modifier: SamplerModifier, version: ModelVersion,
    audioShiftRatio: Float, qkNorm: Bool,
    dualAttentionLayers: [Int], distilledGuidanceLayers: Int, activationQkScaling: [Int: Int],
    activationProjScaling: [Int: Int], activationFfnProjUpScaling: [Int: Int],
    activationFfnScaling: [Int: Int],
    usesFlashAttention: UseFlashAttention,
    usesSolAttention: Bool = false,
    solAttentionStart: Int = 2, solAttentionTau: Float = 0.5,
    upcastAttention: Bool, externalOnDemand: Bool, injectControls: Bool,
    injectT2IAdapters: Bool, injectAttentionKV: Bool, injectIPAdapterLengths: [Int],
    lora: [LoRAConfiguration],
    classifierFreeGuidance: Bool, isGuidanceEmbedEnabled: Bool, isQuantizedModel: Bool,
    canRunLoRASeparately: Bool, deviceProperties: DeviceProperties,
    conditioning: Denoiser.Conditioning, tiledDiffusion: TiledConfiguration,
    teaCache: TeaCacheConfiguration, causalInference: (Int, pad: Int),
    cfgZeroStar: CfgZeroStarConfiguration, isBF16: Bool, discretization: Discretization,
    weightsCache: WeightsCache
  ) {
    self.kind = kind
    self.filePath = filePath
    self.modifier = modifier
    self.version = version
    self.audioShiftRatio = audioShiftRatio
    self.qkNorm = qkNorm
    self.dualAttentionLayers = dualAttentionLayers
    self.distilledGuidanceLayers = distilledGuidanceLayers
    self.activationQkScaling = activationQkScaling
    self.activationProjScaling = activationProjScaling
    self.activationFfnProjUpScaling = activationFfnProjUpScaling
    self.activationFfnScaling = activationFfnScaling
    self.usesFlashAttention = usesFlashAttention
    self.usesSolAttention = usesSolAttention
    self.solAttentionStart = solAttentionStart
    self.solAttentionTau = solAttentionTau
    self.upcastAttention = upcastAttention
    self.externalOnDemand = externalOnDemand
    self.injectControls = injectControls
    self.injectT2IAdapters = injectT2IAdapters
    self.injectAttentionKV = injectAttentionKV

    self.injectIPAdapterLengths = injectIPAdapterLengths
    self.lora = lora
    self.classifierFreeGuidance = classifierFreeGuidance
    self.isGuidanceEmbedEnabled = isGuidanceEmbedEnabled
    self.isQuantizedModel = isQuantizedModel
    self.canRunLoRASeparately = canRunLoRASeparately
    self.deviceProperties = deviceProperties
    self.conditioning = conditioning
    self.tiledDiffusion = tiledDiffusion
    self.teaCache = teaCache
    self.causalInference = causalInference
    self.cfgZeroStar = cfgZeroStar
    self.isBF16 = isBF16
    self.discretization = discretization
    self.weightsCache = weightsCache
  }
}

extension ComfyMultistepSampler: Sampler {
  public func sample(
    _ x_T: DynamicGraph.Tensor<FloatType>, unets existingUNets: [UNet?],
    sample: DynamicGraph.Tensor<FloatType>?, conditionImage: DynamicGraph.Tensor<FloatType>?,
    referenceImages: [DynamicGraph.Tensor<FloatType>],
    referenceAudios: [DynamicGraph.Tensor<FloatType>],
    mask: DynamicGraph.Tensor<FloatType>?, negMask: DynamicGraph.Tensor<FloatType>?,
    conditioning c: [DynamicGraph.Tensor<FloatType>], tokenLengthUncond: Int, tokenLengthCond: Int,
    extraProjection: DynamicGraph.Tensor<FloatType>?,
    injectedControls: [(
      model: ControlModel<FloatType>, hints: [([DynamicGraph.Tensor<FloatType>], Float)]
    )],
    textGuidanceScale: Float, imageGuidanceScale: Float, guidanceEmbed: Float,
    startStep: (integral: Int, fractional: Float), endStep: (integral: Int, fractional: Float),
    originalSize: (width: Int, height: Int), cropTopLeft: (top: Int, left: Int),
    targetSize: (width: Int, height: Int), aestheticScore: Float,
    negativeOriginalSize: (width: Int, height: Int), negativeAestheticScore: Float,
    zeroNegativePrompt: Bool, refiner: Refiner?, fpsId: Int, motionBucketId: Int, condAug: Float,
    startFrameCfg: Float, sharpness: Float, sampling: Sampling,
    cancellation: (@escaping () -> Void) -> Void, feedback: (Int, Tensor<FloatType>?) -> Bool
  ) -> Result<SamplerOutput<FloatType, UNet>, Error> {
    guard endStep.integral > startStep.integral else {
      return .success(SamplerOutput(x: x_T, unets: [nil]))
    }
    var x = x_T
    let batchSize = x.shape[0]
    let startHeight = x.shape[1]
    let startWidth = x.shape[2]
    let channels = x.shape[3]
    let graph = x.graph
    let isCfgEnabled =
      classifierFreeGuidance
      && isCfgEnabled(
        textGuidanceScale: textGuidanceScale, imageGuidanceScale: imageGuidanceScale,
        startFrameCfg: startFrameCfg, version: version, modifier: modifier)
    let (cfgChannels, inChannels) = cfgChannelsAndInputChannels(
      channels: channels, conditionShape: conditionImage?.shape, isCfgEnabled: isCfgEnabled,
      textGuidanceScale: textGuidanceScale, imageGuidanceScale: imageGuidanceScale,
      version: version, modifier: modifier)
    let isBatchEnabled = isBatchEnabled(version)
    let zeroNegativePrompt = isCfgEnabled && zeroNegativePrompt
    var xIn = graph.variable(
      .GPU(0), .NHWC(cfgChannels * batchSize, startHeight, startWidth, inChannels),
      of: FloatType.self
    )
    var c = c
    updateCfgInputAndConditions(
      xIn: &xIn, conditions: &c, conditionImage: conditionImage, batchSize: batchSize,
      startHeight: startHeight, startWidth: startWidth, channels: channels,
      isCfgEnabled: isCfgEnabled, textGuidanceScale: textGuidanceScale, modifier: modifier)
    var extraProjection = extraProjection
    var tokenLengthUncond = tokenLengthUncond
    if !isCfgEnabled && ![.svdI2v, .minimaxH3, .minimaxH3Image].contains(version) {
      for i in 0..<c.count {
        let shape = c[i].shape
        guard shape[0] >= batchSize * 2 else { continue }
        if shape.count == 3 {
          let conditionalLength = version == .kandinsky21 ? shape[1] : tokenLengthCond
          // Only tokenLengthCond is used.
          c[i] = c[i][batchSize..<(batchSize * 2), 0..<conditionalLength, 0..<shape[2]].copied()
        } else if shape.count == 2 {
          c[i] = c[i][batchSize..<(batchSize * 2), 0..<shape[1]].copied()
        }
      }
      if var projection = extraProjection {
        let shape = projection.shape
        if shape[0] >= batchSize * 2 {
          if shape.count == 3 {
            // Only tokenLengthCond is used.
            projection = projection[batchSize..<(batchSize * 2), 0..<shape[1], 0..<shape[2]]
              .copied()
          } else if shape.count == 2 {
            projection = projection[batchSize..<(batchSize * 2), 0..<shape[1]].copied()
          }
        }
        extraProjection = projection
      }
      // There is no tokenLengthUncond any more.
      tokenLengthUncond = tokenLengthCond
    }
    let streamContext = StreamContext(.GPU(0))
    let result: Result<SamplerOutput<FloatType, UNet>, Error> = graph.withStream(streamContext) {
      let oldC = c
      var conditions: [DynamicGraph.AnyTensor] = c
      let fixedEncoder = UNetFixedEncoder<FloatType>(
        filePath: filePath, version: version, modifier: modifier,
        dualAttentionLayers: dualAttentionLayers, activationQkScaling: activationQkScaling,
        activationProjScaling: activationProjScaling,
        activationFfnProjUpScaling: activationFfnProjUpScaling,
        activationFfnScaling: activationFfnScaling,
        usesFlashAttention: usesFlashAttention,
        zeroNegativePrompt: zeroNegativePrompt, isQuantizedModel: isQuantizedModel,
        canRunLoRASeparately: canRunLoRASeparately, externalOnDemand: externalOnDemand,
        deviceProperties: deviceProperties, weightsCache: weightsCache)
      let injectedControlsC: [[DynamicGraph.Tensor<FloatType>]]
      let alphasCumprod = discretization.alphasCumprod(steps: sampling.steps, shift: sampling.shift)
      let sigmas = alphasCumprod.map { discretization.sigma(from: $0) }
      // The sigmas this run steps through, as ComfyUI's `sigmas` truncated for a denoise below 1.
      // A fractional start interpolates its timestep as `noiseScaleFactor(at:)` does, so the
      // latent's noise is the noise the model is told.
      var runSigmas = (startStep.integral...endStep.integral).map {
        sigmas[max(0, min($0, sigmas.count - 1))]
      }
      if Float(startStep.integral) != startStep.fractional {
        let lowTimestep = discretization.timestep(
          for: alphasCumprod[max(0, min(Int(startStep.integral), alphasCumprod.count - 1))])
        let highTimestep = discretization.timestep(
          for: alphasCumprod[
            max(0, min(Int(startStep.fractional.rounded(.up)), alphasCumprod.count - 1))])
        let timestep =
          lowTimestep
          + Float(highTimestep - lowTimestep) * (startStep.fractional - Float(startStep.integral))
        runSigmas[0] = discretization.sigma(
          from: discretization.alphaCumprod(timestep: timestep, shift: 1))
      }
      let isFlow: Bool
      if case .u(_) = discretization.objective {
        isFlow = true
      } else {
        isFlow = false
      }
      let steps = ComfySamplerSteps(
        kind, sigmas: runSigmas, isFlow: isFlow,
        firstSigmaOffset: discretization.percentToSigma(1e-4, shift: sampling.shift) ?? 1)
      let evaluationsPerStep = kind.evaluationsPerStep
      func conditioningValue(_ sigma: Double) -> Float {
        let alphaCumprod = discretization.alphaCumprod(from: sigma)
        switch conditioning {
        case .noise:
          return discretization.noise(for: alphaCumprod)
        case .timestep:
          return discretization.timestep(for: alphaCumprod)
        }
      }
      // What the fixed encoder and controls are given from step `first` on: one timestep per model
      // evaluation (Heun: the step's sigma, then the next one's for its corrector, except on the
      // Euler step to 0), and for MiniMax H3 Image's head selector each evaluation's interval:
      // both of Heun's use the step's own, σ_i → σ_{i+1}.
      func encoderTimesteps(from first: Int) -> (timesteps: [Float], intervals: [(Float, Float)]) {
        var timesteps: [Float] = []
        var intervals: [(Float, Float)] = []
        for k in first..<steps.count {
          let now = conditioningValue(steps.sigmas[k])
          let next = steps.sigmas[k + 1] > 0 ? conditioningValue(steps.sigmas[k + 1]) : 0
          timesteps.append(now)
          intervals.append((now, next))
          if evaluationsPerStep == 2 && steps.sigmas[k + 1] > 0 {
            timesteps.append(next)
            intervals.append(
              Self.heunHeadsFollowList ? (next, next) : (now, next))
          }
        }
        return (timesteps, intervals)
      }
      let (timesteps, headIntervals) = encoderTimesteps(from: 0)
      let listOfLoras = lora
      var lora = listOfLoras.filter {
        $0.mode == .base || $0.mode == .all
      }
      if UNetFixedEncoder<FloatType>.isFixedEncoderRequired(version: version) {
        let vector = fixedEncoder.vector(
          textEmbedding: c[c.count - 1], originalSize: originalSize,
          cropTopLeft: cropTopLeft,
          targetSize: targetSize, aestheticScore: aestheticScore,
          negativeOriginalSize: negativeOriginalSize,
          negativeAestheticScore: negativeAestheticScore,
          fpsId: fpsId, motionBucketId: motionBucketId, condAug: condAug)
        let (encodings, weightMapper) = fixedEncoder.encode(
          isCfgEnabled: isCfgEnabled, textGuidanceScale: textGuidanceScale,
          guidanceEmbed: guidanceEmbed, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
          distilledGuidanceLayers: distilledGuidanceLayers, modifier: modifier,
          textEncoding: c, timesteps: timesteps, audioShiftRatio: audioShiftRatio,
          batchSize: batchSize, startHeight: startHeight,
          startWidth: startWidth,
          tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond, lora: lora,
          tiledDiffusion: tiledDiffusion, teaCache: teaCache, isBF16: isBF16,
          injectedControls: injectedControls, referenceImages: referenceImages,
          referenceAudios: referenceAudios, headIntervals: headIntervals)
        conditions = vector + encodings
        injectedControlsC = injectedControls.map {
          $0.model.encode(
            isCfgEnabled: isCfgEnabled, textGuidanceScale: textGuidanceScale,
            guidanceEmbed: guidanceEmbed, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
            textEncoding: oldC, timesteps: timesteps, vector: vector.first, batchSize: batchSize,
            startHeight: startHeight,
            startWidth: startWidth, tokenLengthUncond: tokenLengthUncond,
            tokenLengthCond: tokenLengthCond, zeroNegativePrompt: zeroNegativePrompt,
            mainUNetFixed: (fixedEncoder.filePath, weightMapper))
        }
      } else {
        injectedControlsC = injectedControls.map {
          $0.model.encode(
            isCfgEnabled: isCfgEnabled, textGuidanceScale: textGuidanceScale,
            guidanceEmbed: guidanceEmbed, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
            textEncoding: oldC, timesteps: timesteps, vector: nil, batchSize: batchSize,
            startHeight: startHeight,
            startWidth: startWidth, tokenLengthUncond: tokenLengthUncond,
            tokenLengthCond: tokenLengthCond, zeroNegativePrompt: zeroNegativePrompt,
            mainUNetFixed: (fixedEncoder.filePath, nil))
        }
      }
      var unet = existingUNets[0] ?? UNet()
      defer {
        cancellation {}  // In this way, we are not holding the graph any more.
      }
      cancellation {
        unet.cancel()
      }
      let referenceImageCount = referenceImages.count
      let referenceAudioCount = referenceAudios.count
      var controlNets = [Model?](repeating: nil, count: injectedControls.count)
      let injectControlsAndAdapters = InjectControlsAndAdapters(
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectAttentionKV: injectAttentionKV, injectIPAdapterLengths: injectIPAdapterLengths,
        injectControlModels: injectedControls.map { $0.model })
      if existingUNets[0] == nil {
        let firstTimestep =
          discretization.timesteps - discretization.timesteps / Float(sampling.steps) + 1
        let t = unet.timeEmbed(
          graph: graph, batchSize: cfgChannels * batchSize, timestep: firstTimestep,
          version: version)
        let emptyInjectedControlsAndAdapters =
          ControlModel<FloatType>
          .emptyInjectedControlsAndAdapters(
            injecteds: injectedControls, step: 0, version: version, inputs: xIn,
            tiledDiffusion: tiledDiffusion)
        let newC: [DynamicGraph.AnyTensor]
        if version == .svdI2v {
          newC = Array(conditions[0..<(1 + (conditions.count - 1) / 2)])
        } else {
          newC = conditions
        }
        let _ = unet.compileModel(
          filePath: filePath, externalOnDemand: externalOnDemand,
          deviceProperties: deviceProperties,
          version: version,
          modifier: modifier, qkNorm: qkNorm,
          dualAttentionLayers: dualAttentionLayers,
          upcastAttention: upcastAttention, usesFlashAttention: usesFlashAttention,
          usesSolAttention: usesSolAttention,
          solAttentionStart: solAttentionStart, solAttentionTau: solAttentionTau,
          injectControlsAndAdapters: injectControlsAndAdapters, lora: lora,
          isQuantizedModel: isQuantizedModel, canRunLoRASeparately: canRunLoRASeparately,
          inputs: xIn, t,
          UNetExtractConditions(
            of: FloatType.self,
            graph: graph, index: 0, batchSize: cfgChannels * batchSize,
            tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
            conditions: newC,
            referenceImageCount: referenceImageCount, referenceAudioCount: referenceAudioCount,
            version: version, modifier: modifier, isCfgEnabled: isCfgEnabled),
          tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
          isCfgEnabled: isCfgEnabled, extraProjection: extraProjection,
          injectedControlsAndAdapters: emptyInjectedControlsAndAdapters,
          referenceImageCount: referenceImageCount, referenceAudioCount: referenceAudioCount,
          tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
          isBF16: isBF16, activationQkScaling: activationQkScaling,
          activationProjScaling: activationProjScaling,
          activationFfnProjUpScaling: activationFfnProjUpScaling,
          activationFfnScaling: activationFfnScaling, weightsCache: weightsCache)
      }
      var maskNoise: DynamicGraph.Tensor<FloatType>? = nil
      if mask != nil {
        maskNoise = graph.variable(.GPU(0), .NHWC(batchSize, startHeight, startWidth, channels))
      }
      var stepNoise: DynamicGraph.Tensor<FloatType>? = nil
      let condAugFrames: DynamicGraph.Tensor<FloatType>?
      let textGuidanceVector: DynamicGraph.Tensor<FloatType>?
      if version == .svdI2v {
        let scaleCPU = graph.variable(.CPU, .NHWC(batchSize, 1, 1, 1), of: FloatType.self)
        for i in 0..<batchSize {
          scaleCPU[i, 0, 0, 0] = FloatType(
            Float(i) * (textGuidanceScale - startFrameCfg) / Float(batchSize - 1) + startFrameCfg)
        }
        textGuidanceVector = scaleCPU.toGPU(0)
        let maskedImage = conditionImage!
        var frames = graph.variable(
          .GPU(0), .NHWC(batchSize, startHeight, startWidth, channels), of: FloatType.self)
        for i in 0..<batchSize {
          frames[i..<(i + 1), 0..<startHeight, 0..<startWidth, 0..<channels] = maskedImage
        }
        if condAug > 0 {
          let noise = graph.variable(like: frames)
          noise.randn(std: condAug)
          frames = frames .+ noise
        }
        condAugFrames = frames
      } else {
        textGuidanceVector = nil
        condAugFrames = nil
      }
      let blur: Model?
      if sharpness > 0 {
        blur = Blur(filters: channels, sigma: 3.0, size: 13, input: x)
      } else {
        blur = nil
      }
      var refinerKickIn = refiner.map { (1 - $0.start) * discretization.timesteps } ?? -1
      var unets: [UNet?] = [unet]
      var currentModelVersion = version
      var indexOffset = startStep.integral
      // A txt2img start: unit noise to the first sigma (k-diffusion scale), as DPMPP2MSampler.
      if startStep.fractional == 0 && sigmas[0] != 1 {
        x = Float(sigmas[0]) * x
      }
      /// One model evaluation: the denoised prediction D(x, σ) (CONST: x - σ·v), with CFG applied.
      /// `evaluation` is 0, or 1 for Heun's corrector, which is told the next sigma.
      func denoise(
        _ x: DynamicGraph.Tensor<FloatType>, sigma: Double, nextSigma: Double, step i: Int,
        evaluation: Int
      ) -> DynamicGraph.Tensor<FloatType> {
        let alphaCumprod = discretization.alphaCumprod(from: sigma)
        let sqrtAlphaCumprod = alphaCumprod.squareRoot()
        let input: DynamicGraph.Tensor<FloatType>
        switch discretization.objective {
        case .u(_):
          input = x
        case .v, .epsilon:
          input = Float(sqrtAlphaCumprod) * x
        case .edm(let sigmaData):
          input = Float(1.0 / (sigma * sigma + sigmaData * sigmaData).squareRoot()) * x
        }
        let timestep = discretization.timestep(for: alphaCumprod)
        let cNoise = conditioningValue(sigma)
        let cNoiseNext = nextSigma > 0 ? conditioningValue(nextSigma) : 0
        let conditionIndex = evaluationsPerStep * (i - indexOffset) + evaluation
        let controlIndex = evaluationsPerStep * (i - startStep.integral) + evaluation
        let t = unet.timeEmbed(
          graph: graph, batchSize: cfgChannels * batchSize, timestep: cNoise,
          version: currentModelVersion)
        let conditions = UNetExtractConditions(
          of: FloatType.self,
          graph: graph, index: conditionIndex, batchSize: cfgChannels * batchSize,
          tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
          conditions: conditions, referenceImageCount: referenceImageCount,
          referenceAudioCount: referenceAudioCount,
          version: currentModelVersion, modifier: modifier, isCfgEnabled: isCfgEnabled)
        let alpha =
          0.001 * sharpness * (discretization.timesteps - timestep) / discretization.timesteps
        let et: DynamicGraph.Tensor<FloatType>
        if version == .svdI2v, let textGuidanceVector = textGuidanceVector,
          let condAugFrames = condAugFrames
        {
          xIn[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<channels] = input
          xIn[0..<batchSize, 0..<startHeight, 0..<startWidth, channels..<(channels * 2)] =
            condAugFrames
          let injectedIPAdapters = ControlModel<FloatType>
            .injectedIPAdapters(
              injecteds: injectedControls, step: i, version: unet.version,
              usesFlashAttention: usesFlashAttention != .none, inputs: xIn, t, injectedControlsC,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              isCfgEnabled: isCfgEnabled, index: controlIndex,
              mainUNetAndWeightMapper: unet.modelAndWeightMapper,
              controlNets: &controlNets)
          let injectedControlsAndAdapters = ControlModel<FloatType>
            .injectedControlsAndAdapters(
              injecteds: injectedControls, step: i, version: unet.version,
              usesFlashAttention: usesFlashAttention != .none, inputs: xIn, t, injectedControlsC,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              isCfgEnabled: isCfgEnabled, index: controlIndex,
              mainUNetAndWeightMapper: unet.modelAndWeightMapper,
              controlNets: &controlNets)
          let cCond = Array(conditions[0..<(1 + (conditions.count - 1) / 2)])
          var etCond = unet(
            timestep: (now: cNoise, next: cNoiseNext), audioShiftRatio: audioShiftRatio,
            inputs: xIn, t, cCond,
            extraProjection: extraProjection,
            injectedControlsAndAdapters: injectedControlsAndAdapters,
            injectedIPAdapters: injectedIPAdapters, referenceImageCount: referenceImageCount,
            referenceAudioCount: referenceAudioCount,
            step: i, tokenLengthUncond: tokenLengthUncond,
            tokenLengthCond: tokenLengthCond, isCfgEnabled: isCfgEnabled,
            tiledDiffusion: tiledDiffusion,
            controlNets: &controlNets)
          if let blur = blur {
            let etCondDegraded = blur(inputs: etCond)[0].as(of: FloatType.self)
            etCond = Functional.add(
              left: etCondDegraded, right: etCond, leftScalar: alpha, rightScalar: 1 - alpha)
          }
          if isCfgEnabled {
            xIn[0..<batchSize, 0..<startHeight, 0..<startWidth, channels..<(channels * 2)].full(0)
            let cUncond = Array([conditions[0]] + conditions[(1 + (conditions.count - 1) / 2)...])
            let etUncond = unet(
              timestep: (now: cNoise, next: cNoiseNext), audioShiftRatio: audioShiftRatio,
              inputs: xIn, t, cUncond,
              extraProjection: extraProjection,
              injectedControlsAndAdapters: injectedControlsAndAdapters,
              injectedIPAdapters: injectedIPAdapters, referenceImageCount: referenceImageCount,
              referenceAudioCount: referenceAudioCount,
              step: i, tokenLengthUncond: tokenLengthUncond,
              tokenLengthCond: tokenLengthCond, isCfgEnabled: isCfgEnabled,
              tiledDiffusion: tiledDiffusion,
              controlNets: &controlNets)
            et = etUncond + textGuidanceVector .* (etCond - etUncond)
          } else {
            et = etCond
          }
        } else {
          xIn[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<channels] = input
          if isCfgEnabled {
            xIn[batchSize..<(batchSize * 2), 0..<startHeight, 0..<startWidth, 0..<channels] = input
            if xIn.shape[0] >= batchSize * 3 {
              xIn[
                (batchSize * 2)..<(batchSize * 3), 0..<startHeight, 0..<startWidth, 0..<channels] =
                input
            }
          }
          let injectedIPAdapters = ControlModel<FloatType>
            .injectedIPAdapters(
              injecteds: injectedControls, step: i, version: unet.version,
              usesFlashAttention: usesFlashAttention != .none, inputs: xIn, t, injectedControlsC,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              isCfgEnabled: isCfgEnabled, index: controlIndex,
              mainUNetAndWeightMapper: unet.modelAndWeightMapper,
              controlNets: &controlNets)
          let injectedControlsAndAdapters = ControlModel<FloatType>
            .injectedControlsAndAdapters(
              injecteds: injectedControls, step: i, version: unet.version,
              usesFlashAttention: usesFlashAttention != .none, inputs: xIn, t, injectedControlsC,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              isCfgEnabled: isCfgEnabled, index: controlIndex,
              mainUNetAndWeightMapper: unet.modelAndWeightMapper,
              controlNets: &controlNets)
          let etOut = unet(
            timestep: (now: cNoise, next: cNoiseNext), audioShiftRatio: audioShiftRatio,
            inputs: xIn, t, conditions,
            extraProjection: extraProjection,
            injectedControlsAndAdapters: injectedControlsAndAdapters,
            injectedIPAdapters: injectedIPAdapters, referenceImageCount: referenceImageCount,
            referenceAudioCount: referenceAudioCount,
            step: i, tokenLengthUncond: tokenLengthUncond,
            tokenLengthCond: tokenLengthCond, isCfgEnabled: isCfgEnabled,
            tiledDiffusion: tiledDiffusion,
            controlNets: &controlNets)
          et = applyCfg(
            etOut: etOut, blur: blur, batchSize: batchSize, startHeight: startHeight,
            startWidth: startWidth, channels: channels, isCfgEnabled: isCfgEnabled,
            textGuidanceScale: textGuidanceScale, imageGuidanceScale: imageGuidanceScale,
            alpha: alpha, modifier: modifier, step: i, isBatchEnabled: isBatchEnabled,
            cfgZeroStar: cfgZeroStar)
        }
        var denoised: DynamicGraph.Tensor<FloatType>
        switch discretization.objective {
        case .u(_):
          // CONST.calculate_denoised: x - σ·v
          denoised = Functional.add(left: x, right: et, leftScalar: 1, rightScalar: Float(-sigma))
        case .v:
          denoised = Functional.add(
            left: x, right: et, leftScalar: Float(1.0 / (sigma * sigma + 1)),
            rightScalar: Float(-sigma * sqrtAlphaCumprod))
        case .epsilon:
          denoised = Functional.add(left: x, right: et, leftScalar: 1, rightScalar: Float(-sigma))
          if version == .kandinsky21 {
            denoised = clipDenoised(denoised)
          }
        case .edm(let sigmaData):
          let sigmaData2 = sigmaData * sigmaData
          denoised = Functional.add(
            left: x, right: et, leftScalar: Float(sigmaData2 / (sigma * sigma + sigmaData2)),
            rightScalar: Float(sigma * sigmaData / (sigma * sigma + sigmaData2).squareRoot()))
        }
        return denoised
      }
      func scaledSum(
        _ terms: [(Double, DynamicGraph.Tensor<FloatType>?)]
      ) -> DynamicGraph.Tensor<FloatType> {
        let present = terms.compactMap { term in term.1.map { (Float(term.0), $0) } }
          .filter { $0.0 != 0 }
        guard var sum = present.first.map({ $0.0 == 1 ? $0.1 : $0.0 * $0.1 }) else {
          return 0 * x
        }
        for (scale, tensor) in present.dropFirst() {
          sum = Functional.add(left: sum, right: tensor, leftScalar: 1, rightScalar: scale)
        }
        return sum
      }
      var oldDenoised: DynamicGraph.Tensor<FloatType>? = nil
      var olderDenoised: DynamicGraph.Tensor<FloatType>? = nil
      for i in startStep.integral..<endStep.integral {
        let k = i - startStep.integral
        let sigma = steps.sigmas[k]
        let nextSigma = steps.sigmas[k + 1]
        let rawValue: Tensor<FloatType>? =
          (i > max(startStep.integral, sampling.steps / 2) || i % 2 == 1)
          ? (oldDenoised.map { unet.decode($0) })?.rawValue.toCPU() : nil
        if i % 5 == 4, let rawValue = rawValue {
          if isNaN(rawValue) {
            return .failure(SamplerError.isNaN)
          }
        }
        guard feedback(i - startStep.integral, rawValue) else {
          return .failure(SamplerError.cancelled(unets))
        }
        let timestep = discretization.timestep(for: discretization.alphaCumprod(from: sigma))
        if timestep < refinerKickIn, let refiner = refiner {
          let timesteps = encoderTimesteps(from: k)
          unets = [nil]
          unet.unloadModel()
          lora =
            (refiner.builtinLora
              ? [
                LoRAConfiguration(
                  file: refiner.filePath, weight: 1, version: refiner.version, isLoHa: false,
                  modifier: .none, mode: .refiner)
              ] : [])
            + listOfLoras.filter {
              $0.mode == .all || $0.mode == .refiner
            }
          let fixedEncoder = UNetFixedEncoder<FloatType>(
            filePath: refiner.filePath, version: refiner.version,
            modifier: modifier, dualAttentionLayers: refiner.dualAttentionLayers,
            activationQkScaling: refiner.activationQkScaling,
            activationProjScaling: refiner.activationProjScaling,
            activationFfnProjUpScaling: refiner.activationFfnProjUpScaling,
            activationFfnScaling: refiner.activationFfnScaling,
            usesFlashAttention: usesFlashAttention,
            zeroNegativePrompt: zeroNegativePrompt,
            isQuantizedModel: refiner.isQuantizedModel, canRunLoRASeparately: canRunLoRASeparately,
            externalOnDemand: refiner.externalOnDemand, deviceProperties: deviceProperties,
            weightsCache: weightsCache)
          if UNetFixedEncoder<FloatType>.isFixedEncoderRequired(version: refiner.version) {
            conditions = []
            let vector = fixedEncoder.vector(
              textEmbedding: oldC[oldC.count - 1], originalSize: originalSize,
              cropTopLeft: cropTopLeft,
              targetSize: targetSize, aestheticScore: aestheticScore,
              negativeOriginalSize: negativeOriginalSize,
              negativeAestheticScore: negativeAestheticScore, fpsId: fpsId,
              motionBucketId: motionBucketId, condAug: condAug)
            conditions =
              vector
              + fixedEncoder.encode(
                isCfgEnabled: isCfgEnabled, textGuidanceScale: textGuidanceScale,
                guidanceEmbed: guidanceEmbed, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
                distilledGuidanceLayers: refiner.distilledGuidanceLayers, modifier: modifier,
                textEncoding: oldC, timesteps: timesteps.timesteps,
                audioShiftRatio: audioShiftRatio,
                batchSize: batchSize,
                startHeight: startHeight,
                startWidth: startWidth, tokenLengthUncond: tokenLengthUncond,
                tokenLengthCond: tokenLengthCond, lora: lora, tiledDiffusion: tiledDiffusion,
                teaCache: teaCache, isBF16: isBF16, injectedControls: injectedControls,
                referenceImages: referenceImages, referenceAudios: referenceAudios,
                headIntervals: timesteps.intervals
              ).0
            indexOffset = i
          }
          unet = UNet()
          cancellation {
            unet.cancel()
          }
          currentModelVersion = refiner.version
          let firstTimestep =
            discretization.timesteps - discretization.timesteps / Float(sampling.steps) + 1
          let t = unet.timeEmbed(
            graph: graph, batchSize: cfgChannels * batchSize, timestep: firstTimestep,
            version: currentModelVersion)
          let emptyInjectedControlsAndAdapters =
            ControlModel<FloatType>
            .emptyInjectedControlsAndAdapters(
              injecteds: injectedControls, step: 0, version: refiner.version, inputs: xIn,
              tiledDiffusion: tiledDiffusion)
          let newC: [DynamicGraph.AnyTensor]
          if refiner.version == .svdI2v {
            newC = Array(conditions[0..<(1 + (conditions.count - 1) / 2)])
          } else {
            newC = conditions
          }
          let _ = unet.compileModel(
            filePath: refiner.filePath, externalOnDemand: refiner.externalOnDemand,
            deviceProperties: deviceProperties,
            version: refiner.version, modifier: modifier, qkNorm: refiner.qkNorm,
            dualAttentionLayers: refiner.dualAttentionLayers,
            upcastAttention: refiner.upcastAttention,
            usesFlashAttention: usesFlashAttention, usesSolAttention: usesSolAttention,
            solAttentionStart: solAttentionStart, solAttentionTau: solAttentionTau,
            injectControlsAndAdapters: injectControlsAndAdapters,
            lora: lora, isQuantizedModel: refiner.isQuantizedModel,
            canRunLoRASeparately: canRunLoRASeparately,
            inputs: xIn, t,
            UNetExtractConditions(
              of: FloatType.self,
              graph: graph, index: 0, batchSize: cfgChannels * batchSize,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              conditions: newC, referenceImageCount: referenceImageCount,
              referenceAudioCount: referenceAudioCount,
              version: currentModelVersion, modifier: modifier, isCfgEnabled: isCfgEnabled),
            tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
            isCfgEnabled: isCfgEnabled, extraProjection: extraProjection,
            injectedControlsAndAdapters: emptyInjectedControlsAndAdapters,
            referenceImageCount: referenceImageCount, referenceAudioCount: referenceAudioCount,
            tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
            isBF16: refiner.isBF16, activationQkScaling: refiner.activationQkScaling,
            activationProjScaling: refiner.activationProjScaling,
            activationFfnProjUpScaling: refiner.activationFfnProjUpScaling,
            activationFfnScaling: refiner.activationFfnScaling,
            weightsCache: weightsCache)
          refinerKickIn = -1
          unets.append(unet)
        }
        let denoised = denoise(x, sigma: sigma, nextSigma: nextSigma, step: i, evaluation: 0)
        if isNaN(denoised.rawValue.toCPU()) {
          return .failure(SamplerError.isNaN)
        }
        if kind == .heun {
          let step = steps.heunStep(k)
          let x2 = scaledSum([(step.predictor.x, x), (step.predictor.denoised, denoised)])
          if step.isEuler {
            x = x2
          } else {
            let denoised2 = denoise(
              x2, sigma: nextSigma, nextSigma: nextSigma, step: i, evaluation: 1)
            if isNaN(denoised2.rawValue.toCPU()) {
              return .failure(SamplerError.isNaN)
            }
            x = scaledSum([
              (step.x, x), (step.denoised, denoised), (step.x2, x2), (step.denoised2, denoised2),
            ])
          }
        } else {
          let step = steps.step(k)
          var noise: DynamicGraph.Tensor<FloatType>? = nil
          if step.noise > 0 {
            let fresh = stepNoise ?? graph.variable(like: x)
            fresh.randn(std: 1, mean: 0)
            stepNoise = fresh
            noise = fresh
          }
          x = scaledSum([
            (step.x, x), (step.denoised, denoised), (step.previousDenoised, oldDenoised),
            (step.olderDenoised, olderDenoised), (step.noise, noise),
          ])
        }
        olderDenoised = oldDenoised
        oldDenoised = denoised
        if i < endStep.integral - 1, let noise = maskNoise, let sample = sample, let mask = mask,
          let negMask = negMask
        {
          noise.randn(std: 1, mean: 0)
          let qSample: DynamicGraph.Tensor<FloatType>
          if isFlow {
            qSample = Functional.add(
              left: sample, right: noise, leftScalar: Float(1 - nextSigma),
              rightScalar: Float(nextSigma))
          } else {
            qSample = sample + Float(nextSigma) * noise
          }
          x = qSample .* negMask + x .* mask
        }
        if i == endStep.integral - 1 {
          if isNaN(x.rawValue.toCPU()) {
            return .failure(SamplerError.isNaN)
          }
        }
      }
      return .success(SamplerOutput(x: x, unets: unets))
    }
    streamContext.joined()
    return result
  }

  // Same start and noise level as DPMPP2MSampler, so an img2img latent is noised at exactly the
  // sigma the first step tells the model (interpolated in timestep space).
  public func timestep(for strength: Float, sampling: Sampling) -> (
    timestep: Float, startStep: Float, roundedDownStartStep: Int, roundedUpStartStep: Int
  ) {
    let tEnc = strength * discretization.timesteps
    let initTimestep = tEnc
    let alphasCumprod = discretization.alphasCumprod(steps: sampling.steps, shift: sampling.shift)
    var previousTimestep = discretization.timesteps
    for (i, alphaCumprod) in alphasCumprod.enumerated() {
      let timestep = discretization.timestep(for: alphaCumprod)
      if initTimestep >= timestep {
        guard i > 0 else {
          return (
            timestep: timestep, startStep: 0, roundedDownStartStep: 0, roundedUpStartStep: 0
          )
        }
        guard initTimestep > timestep + 1e-3 else {
          return (
            timestep: initTimestep, startStep: Float(i), roundedDownStartStep: i,
            roundedUpStartStep: i
          )
        }
        return (
          timestep: Float(initTimestep),
          startStep: Float(i - 1) + Float(initTimestep - previousTimestep)
            / Float(timestep - previousTimestep), roundedDownStartStep: i - 1, roundedUpStartStep: i
        )
      }
      previousTimestep = timestep
    }
    return (
      timestep: discretization.timestep(for: alphasCumprod[0]),
      startStep: Float(alphasCumprod.count - 1),
      roundedDownStartStep: alphasCumprod.count - 1, roundedUpStartStep: alphasCumprod.count - 1
    )
  }

  public func sampleScaleFactor(at step: Float, sampling: Sampling) -> Float {
    if case .u(_) = discretization.objective {
      return 1 - noiseScaleFactor(at: step, sampling: sampling)
    } else {
      return 1
    }
  }

  public func noiseScaleFactor(at step: Float, sampling: Sampling) -> Float {
    let alphasCumprod = discretization.alphasCumprod(steps: sampling.steps, shift: sampling.shift)
    let lowTimestep = discretization.timestep(
      for: alphasCumprod[max(0, min(Int(step.rounded(.down)), alphasCumprod.count - 1))])
    let highTimestep = discretization.timestep(
      for: alphasCumprod[max(0, min(Int(step.rounded(.up)), alphasCumprod.count - 1))])
    let timestep = lowTimestep + (highTimestep - lowTimestep) * (step - Float(step.rounded(.down)))
    let alphaCumprod = discretization.alphaCumprod(timestep: timestep, shift: 1)
    let sigma = discretization.sigma(from: alphaCumprod)
    return Float(sigma)
  }
}

extension ComfyMultistepSampler {
  /// Heun's corrector on MiniMax H3 Image uses the step's own output heads (σ_i → σ_{i+1}). Set
  /// ATELIER_HEUN_HEADS=adjacent to give it the heads of its own sigma instead (as plain list
  /// adjacency of the interleaved timesteps would), for comparing the two.
  static var heunHeadsFollowList: Bool {
    ProcessInfo.processInfo.environment["ATELIER_HEUN_HEADS"] == "adjacent"
  }
}
