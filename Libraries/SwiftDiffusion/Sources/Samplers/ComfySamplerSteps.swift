import Foundation

extension Denoiser {
  /// Samplers ported from ComfyUI (comfy/k_diffusion/sampling.py) with ComfyUI's settings for its
  /// sampler names: `dpmpp_2m_sde` (midpoint, eta 1), `res_multistep` (eta 0), `heun`
  /// (s_churn 0) and `er_sde` (max_stage 3). They step on a `Scheduler`'s sigmas; the per-step
  /// arithmetic is in `ComfySamplerSteps`, which the samplers turn into tensor operations.
  public enum ComfySampler: String, CaseIterable, Sendable {
    case dpmpp2mSDE = "dpmpp_2m_sde"
    case resMultistep = "res_multistep"
    case heun
    case erSDE = "er_sde"

    /// Whether the model is evaluated twice a step (Heun's corrector).
    public var evaluationsPerStep: Int { self == .heun ? 2 : 1 }

    /// `offset_first_sigma_for_snr` (sampling.py:168) is applied by the SDE samplers only:
    /// their half-log-SNR is infinite at a flow model's sigma 1.
    public var offsetsFirstSigma: Bool { self == .dpmpp2mSDE || self == .erSDE }
  }
}

extension Denoiser.Scheduler {
  /// The schedule a ComfyUI sampler steps on. These samplers have no schedule of their own, so
  /// "Sampler's Own" means Simple on flow models (on MiniMax H3 Image, its sigma grid) and Karras
  /// on DDPM and EDM models; a scheduler a model can't take (`isSupported(by:)`) is replaced the
  /// same way. Every other sampler keeps `self` (`sampler` nil).
  public func resolved(
    for parameterization: Denoiser.Parameterization, sampler: Denoiser.ComfySampler?
  ) -> Denoiser.Scheduler {
    guard sampler != nil else { return self }
    if self != .samplerDefault && isSupported(by: parameterization) {
      return self
    }
    switch parameterization {
    case .rf(_):
      return .simple
    case .ddpm(_), .edm(_):
      return .karras
    }
  }
}

/// The scalar side of the ComfyUI samplers: for each step, the coefficients of the update as a
/// linear combination of the current latent, this step's and earlier steps' denoised predictions
/// and fresh unit noise. All four samplers' updates are linear in those tensors, so the tensor code
/// only adds them up and these numbers can be checked against ComfyUI's own on a toy denoiser.
///
/// `sigmas` are the sigmas the model is told at each step followed by the last one (ComfyUI's
/// `sigmas` tensor, truncated to the steps being run, as ComfyUI does for a denoise below 1), in the
/// model's own sigma: flow 0…1 (ComfyUI's CONST), otherwise the k-diffusion sigma. The first may be
/// an interpolated img2img start; whatever it is, it is the sigma the history (h_last, the
/// er_lambdas) is computed from, as in ComfyUI.
public struct ComfySamplerSteps: Sendable {
  /// x_next = x·`x` + D·`denoised` + D_{i-1}·`previousDenoised` + D_{i-2}·`olderDenoised`
  /// + ε·`noise`, with D the denoised prediction and ε unit Gaussian noise.
  public struct Step: Equatable, Sendable {
    public var x: Double = 0
    public var denoised: Double = 0
    public var previousDenoised: Double = 0
    public var olderDenoised: Double = 0
    public var noise: Double = 0
    public init(
      x: Double = 0, denoised: Double = 0, previousDenoised: Double = 0, olderDenoised: Double = 0,
      noise: Double = 0
    ) {
      self.x = x
      self.denoised = denoised
      self.previousDenoised = previousDenoised
      self.olderDenoised = olderDenoised
      self.noise = noise
    }
  }

  /// Heun's two halves: x₂ = x·`predictor.x` + D·`predictor.denoised` is evaluated at the next sigma
  /// (giving D₂), then x_next = x·`x` + D·`denoised` + x₂·`x2` + D₂·`denoised2`. Nil `x2` and
  /// `denoised2` (the step to sigma 0) mean Euler: x_next is the predictor.
  public struct HeunStep: Equatable, Sendable {
    public var predictor: Step
    public var x: Double
    public var denoised: Double
    public var x2: Double
    public var denoised2: Double
    public var isEuler: Bool
  }

  public let sampler: Denoiser.ComfySampler
  public let isFlow: Bool
  /// The sigmas each step tells the model, then the final one, after `offset_first_sigma_for_snr`.
  public let sigmas: [Double]

  /// `firstSigmaOffset` is `model_sampling.percent_to_sigma(1e-4)`, used in place of a flow
  /// model's first sigma when that is 1 or more and the sampler is an SDE one.
  public init(
    _ sampler: Denoiser.ComfySampler, sigmas: [Double], isFlow: Bool, firstSigmaOffset: Double
  ) {
    self.sampler = sampler
    self.isFlow = isFlow
    var sigmas = sigmas
    // sampling.py:168 `offset_first_sigma_for_snr`: only CONST (flow) models, only sigma >= 1.
    if sampler.offsetsFirstSigma, isFlow, sigmas.count > 1, sigmas[0] >= 1 {
      sigmas[0] = firstSigmaOffset
    }
    self.sigmas = sigmas
  }

  public var count: Int { max(sigmas.count - 1, 0) }

  /// sampling.py:152 `sigma_to_half_log_snr`: log((1 - σ) / σ) for CONST, -log σ otherwise.
  func halfLogSNR(_ sigma: Double) -> Double {
    isFlow ? log((1 - sigma) / sigma) : -log(sigma)
  }

  /// The update of step `i` for the multistep samplers (not Heun).
  public func step(_ i: Int) -> Step {
    precondition(sampler != .heun && i >= 0 && i < count)
    switch sampler {
    case .dpmpp2mSDE:
      return dpmpp2mSDE(i)
    case .resMultistep:
      return resMultistep(i)
    case .erSDE:
      return erSDE(i)
    case .heun:
      fatalError()
    }
  }

  /// sampling.py:890 `sample_dpmpp_2m_sde`, eta 1, s_noise 1, solver_type "midpoint".
  private func dpmpp2mSDE(_ i: Int) -> Step {
    let sigma = sigmas[i]
    let next = sigmas[i + 1]
    guard next > 0 else {
      // "Denoising step"
      return Step(denoised: 1)
    }
    let eta = 1.0
    let lambdaS = halfLogSNR(sigma)
    let lambdaT = halfLogSNR(next)
    let h = lambdaT - lambdaS
    let hEta = h * (eta + 1)
    let alphaT = next * exp(lambdaT)
    let oneMinusExp = -expm1(-hEta)
    var step = Step(x: next / sigma * exp(-h * eta), denoised: alphaT * oneMinusExp)
    // old_denoised is set from the second step on; h_last is the previous step's h.
    if i > 0 {
      let hLast = halfLogSNR(sigmas[i]) - halfLogSNR(sigmas[i - 1])
      let r = hLast / h
      let midpoint = 0.5 * alphaT * oneMinusExp * (1 / r)
      step.denoised += midpoint
      step.previousDenoised = -midpoint
    }
    step.noise = next * (-expm1(-2 * h * eta)).squareRoot()
    return step
  }

  /// sampling.py:1462 `res_multistep` with eta 0 (`sample_res_multistep`): sigma_down is the next
  /// sigma and nothing is added back, so old_sigma_down is this step's sigma. Its t is -log σ even
  /// on flow models, as ComfyUI has it.
  private func resMultistep(_ i: Int) -> Step {
    let sigma = sigmas[i]
    let sigmaDown = sigmas[i + 1]
    guard sigmaDown > 0, i > 0 else {
      // Euler: x + (x - D) / σ · (σ_down - σ)
      let ratio = sigmaDown / sigma
      return Step(x: ratio, denoised: 1 - ratio)
    }
    let t = -log(sigma)
    let tOld = -log(sigma)  // old_sigma_down: the previous step's sigma_down, this sigma.
    let tNext = -log(sigmaDown)
    let tPrev = -log(sigmas[i - 1])
    let h = tNext - t
    let c2 = (tPrev - tOld) / h
    let phi1 = expm1(-h) / -h
    let phi2 = (phi1 - 1) / -h
    // torch.nan_to_num(…, nan=0.0)
    func finite(_ value: Double) -> Double { value.isNaN ? 0 : value }
    let b1 = finite(phi1 - phi2 / c2)
    let b2 = finite(phi2 / c2)
    return Step(x: exp(-h), denoised: h * b1, previousDenoised: h * b2)
  }

  /// sampling.py:1599 `default_er_sde_noise_scaler`.
  static func erNoiseScaler(_ x: Double) -> Double {
    x * (exp(pow(x, 0.3)) + 10)
  }

  /// er_lambda_t = σ_t / α_t: exp(-half_log_snr).
  private func erLambda(_ index: Int) -> Double {
    exp(-halfLogSNR(sigmas[index]))
  }

  /// sampling.py:1593 `sample_er_sde`, s_noise 1, max_stage 3, the default noise scaler and its
  /// 200-point Riemann sums.
  private func erSDE(_ i: Int) -> Step {
    let next = sigmas[i + 1]
    guard next > 0 else {
      return Step(denoised: 1)
    }
    let f = Self.erNoiseScaler
    let stageUsed = min(3, i + 1)
    let lambdaS = erLambda(i)
    let lambdaT = erLambda(i + 1)
    let alphaS = sigmas[i] / lambdaS
    let alphaT = next / lambdaT
    let rAlpha = alphaT / alphaS
    let r = f(lambdaT) / f(lambdaS)
    // Stage 1 Euler
    var step = Step(x: rAlpha * r, denoised: alphaT * (1 - r))
    if stageUsed >= 2 {
      let points = 200
      let dt = lambdaT - lambdaS
      let lambdaStepSize = -dt / Double(points)
      let positions = (0..<points).map { lambdaT + Double($0) * lambdaStepSize }
      // Stage 2: denoised_d = (D - D_{i-1}) / (λ_s - λ_{i-1})
      let s = positions.reduce(0) { $0 + 1 / f($1) } * lambdaStepSize
      let k2 = alphaT * (dt + s * f(lambdaT))
      let g1 = 1 / (lambdaS - erLambda(i - 1))
      step.denoised += k2 * g1
      step.previousDenoised -= k2 * g1
      if stageUsed >= 3 {
        // Stage 3: denoised_u = (denoised_d - old_denoised_d) / ((λ_s - λ_{i-2}) / 2), with
        // old_denoised_d = (D_{i-1} - D_{i-2}) / (λ_{i-1} - λ_{i-2}) from the previous step.
        let sU = positions.reduce(0) { $0 + ($1 - lambdaS) / f($1) } * lambdaStepSize
        let k3 = alphaT * (dt * dt / 2 + sU * f(lambdaT))
        let g3 = 2 / (lambdaS - erLambda(i - 2))
        let g2 = 1 / (erLambda(i - 1) - erLambda(i - 2))
        step.denoised += k3 * g3 * g1
        step.previousDenoised += k3 * g3 * (-g1 - g2)
        step.olderDenoised += k3 * g3 * g2
      }
    }
    // (λ_t² - λ_s² r²).sqrt().nan_to_num(nan=0.0)
    let variance = lambdaT * lambdaT - lambdaS * lambdaS * r * r
    step.noise = variance > 0 ? alphaT * variance.squareRoot() : 0
    return step
  }

  /// sampling.py:269 `sample_heun` with s_churn 0: an Euler predictor to the next sigma, then the
  /// average of both slopes; the step to sigma 0 is Euler alone.
  public func heunStep(_ i: Int) -> HeunStep {
    precondition(sampler == .heun && i >= 0 && i < count)
    let sigma = sigmas[i]
    let next = sigmas[i + 1]
    let dt = next - sigma
    // d = (x - D) / σ; x₂ = x + d·dt
    let predictor = Step(x: 1 + dt / sigma, denoised: -dt / sigma)
    guard next > 0 else {
      return HeunStep(
        predictor: predictor, x: predictor.x, denoised: predictor.denoised, x2: 0, denoised2: 0,
        isEuler: true)
    }
    // x + (d + d₂) / 2 · dt, d₂ = (x₂ - D₂) / σ_next
    return HeunStep(
      predictor: predictor, x: 1 + 0.5 * dt / sigma, denoised: -0.5 * dt / sigma,
      x2: 0.5 * dt / next, denoised2: -0.5 * dt / next, isEuler: false)
  }
}
