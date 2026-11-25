import Foundation

extension Denoiser {
  /// How a sampler spaces its steps, independent of the sampler's math. Raw values are ComfyUI's
  /// scheduler names (`SCHEDULER_HANDLERS` in comfy/samplers.py), and each case computes the same
  /// sigmas ComfyUI does. `samplerDefault` keeps the schedule each sampler type has always used
  /// (Karras for "DPM++ 2M Karras", trailing for "… Trailing", and so on).
  public enum Scheduler: String, Codable, CaseIterable, Sendable {
    case samplerDefault = "default"
    case simple
    case sgmUniform = "sgm_uniform"
    case normal
    case karras
    case exponential
    case linearQuadratic = "linear_quadratic"
    case beta

    /// EDM models have no timestep table for the t-space schedulers to walk; ComfyUI only offers
    /// them the sigma-space ones in practice, so the others fall back to the sampler's own.
    public func isSupported(by parameterization: Parameterization) -> Bool {
      switch parameterization {
      case .edm(_):
        return self == .karras || self == .exponential
      case .rf(_), .ddpm(_):
        return true
      }
    }
  }
}

extension Denoiser {
  /// A discretization whose schedule comes from a `Scheduler` instead of the sampler type. Every
  /// sampler only reads its schedule through `alphasCumprod(steps:shift:)`, so this forwards
  /// everything else to the trailing `LinearDiscretization` (as `RFSigmaGridDiscretization` does):
  /// the model is still told `sigma × conditionScale` (flow) or the continuous training timestep
  /// (DDPM) for whatever sigma a step lands on.
  ///
  /// The sigmas follow ComfyUI's model samplings: `ModelSamplingDiscreteFlow` for flow models
  /// (1000-entry table, `time_snr_shift` by `shift`), `ModelSamplingDiscrete` for DDPM (the
  /// training sigmas, log-interpolated), `ModelSamplingContinuousEDM` for EDM, and
  /// `MiniMaxH3ImageModelSampling` (comfy/ldm/minimax/sampling.py) when a sigma grid is given:
  /// sigma(t) runs piecewise-linearly along the grid, so at shift 1 the schedulers at the grid's
  /// step count land on (or, where ComfyUI's integer table rounds, within 1e-3 of) the grid the
  /// output heads were distilled for; unlike ComfyUI, no step goes below the grid's last nonzero
  /// sigma (`floored(_:at:)`). ComfyUI's H3 Image sampling ignores the shift; Draw Things shifts
  /// every H3 Image schedule with the flow shift (`LinearDiscretization` and the other RF
  /// discretizations), so here the grid schedule takes `shift * s / (1 + (shift - 1) * s)` as a
  /// last step, after the floor, and differs from ComfyUI at any shift but 1. Discrete and EDM
  /// models apply Draw Things' shift (sigma × shift, as `LinearDiscretization` does) to the
  /// result; ComfyUI has none for them.
  public struct ScheduledDiscretization: Discretization {
    private let linearDiscretization: LinearDiscretization
    private let parameterization: Parameterization
    public let scheduler: Scheduler
    /// Ascending sigma grid (index 0 is sigma 0), as `MiniMaxH3ImageModelSampling.grid`.
    private let ascendingGrid: [Double]
    public var timesteps: Float { linearDiscretization.timesteps }
    public var objective: Objective { linearDiscretization.objective }

    /// `sigmaGrid` is a descending grid as `MiniMaxH3Layout.sigmaGrid` stores it (1 … 0); empty
    /// for every model but MiniMax H3 Image.
    public init(
      _ parameterization: Parameterization, objective: Objective, scheduler: Scheduler,
      sigmaGrid: [Double] = []
    ) {
      linearDiscretization = LinearDiscretization(
        parameterization, objective: objective, timestepSpacing: .trailing)
      self.parameterization = parameterization
      self.scheduler = scheduler
      if case .rf(_) = parameterization, sigmaGrid.count >= 2 {
        ascendingGrid = sigmaGrid.reversed()
      } else {
        ascendingGrid = []
      }
    }

    public func timestep(for alphaCumprod: Double) -> Float {
      linearDiscretization.timestep(for: alphaCumprod)
    }
    public func alphaCumprod(timestep: Float, shift: Double) -> Double {
      linearDiscretization.alphaCumprod(timestep: timestep, shift: shift)
    }
    public func alphaCumprod(from sigma: Double) -> Double {
      linearDiscretization.alphaCumprod(from: sigma)
    }
    public func sigma(from alphaCumprod: Double) -> Double {
      linearDiscretization.sigma(from: alphaCumprod)
    }

    public func alphasCumprod(steps: Int, shift: Double) -> [Double] {
      guard let sigmas = sigmas(steps: steps, shift: shift) else {
        return linearDiscretization.alphasCumprod(steps: steps, shift: shift)
      }
      switch parameterization {
      case .rf(_):
        return sigmas.map { 1 - $0 }
      case .ddpm(_), .edm(_):
        return sigmas.map { 1.0 / ($0 * $0 + 1) }
      }
    }

    /// The `steps + 1` sigmas the scheduler gives, descending and ending at 0, in the model's own
    /// sigma (flow: 0…1; DDPM / EDM: sqrt((1 - alphaCumprod) / alphaCumprod)). Nil when the
    /// scheduler has nothing for this model (`samplerDefault`, or a t-space scheduler on EDM).
    public func sigmas(steps: Int, shift: Double) -> [Double]? {
      guard steps > 0, scheduler != .samplerDefault, scheduler.isSupported(by: parameterization)
      else { return nil }
      let sampling = ModelSampling(
        parameterization, shift: shift, ascendingGrid: ascendingGrid)
      var sigmas: [Double]
      switch scheduler {
      case .samplerDefault:
        return nil
      case .simple:
        sigmas = Self.simple(sampling, steps: steps)
      case .sgmUniform:
        sigmas = Self.normal(sampling, steps: steps, sgm: true)
      case .normal:
        sigmas = Self.normal(sampling, steps: steps, sgm: false)
      case .karras:
        sigmas = Self.karras(
          steps: steps, sigmaMin: sampling.sigmaSpaceMin, sigmaMax: sampling.sigmaMax)
      case .exponential:
        sigmas = Self.exponential(
          steps: steps, sigmaMin: sampling.sigmaSpaceMin, sigmaMax: sampling.sigmaMax)
      case .linearQuadratic:
        sigmas = Self.linearQuadratic(steps: steps, sigmaMax: sampling.sigmaMax)
      case .beta:
        sigmas = Self.beta(sampling, steps: steps)
      }
      guard sigmas.count == steps + 1 else { return nil }
      sigmas[steps] = 0
      if let floor = sampling.sigmaFloor {
        sigmas = Self.floored(sigmas, at: floor)
      }
      if sampling.appliesShiftToSigmas && shift != 1 {
        sigmas = sigmas.map { $0 * shift }
      }
      if let gridShift = sampling.gridShift, gridShift != 1 {
        sigmas = sigmas.map { ModelSampling.timeSNRShift($0, shift: gridShift) }
      }
      return sigmas
    }

    /// comfy/model_sampling.py `percent_to_sigma` for flow models (`ModelSamplingDiscreteFlow`, and
    /// `MiniMaxH3ImageModelSampling` on its grid): the sigma at timestep 1 - percent, as
    /// `offset_first_sigma_for_snr` uses in place of sigma 1. Nil for DDPM and EDM models.
    public func percentToSigma(_ percent: Double, shift: Double) -> Double? {
      guard case .rf(_) = parameterization else { return nil }
      guard percent > 0 else { return 1 }
      guard percent < 1 else { return 0 }
      let sampling = ModelSampling(parameterization, shift: shift, ascendingGrid: ascendingGrid)
      let sigma = sampling.sigma(timestep: 1 - percent)
      guard let gridShift = sampling.gridShift, gridShift != 1 else { return sigma }
      return ModelSampling.timeSNRShift(sigma, shift: gridShift)
    }

    /// The nonzero sigmas moved so the lowest is at least `floor`: when it falls below, they're
    /// remapped linearly from [lowest, highest] onto [floor, highest]. Clamping would give repeated
    /// sigmas (zero-length steps) wherever two fall below the floor; the remap keeps every step,
    /// their order and the top of the schedule, and leaves a schedule already above the floor (Simple
    /// at the grid's own step count, Karras and Exponential, which end on it) as it is.
    static func floored(_ sigmas: [Double], at floor: Double) -> [Double] {
      let nonzero = sigmas.filter { $0 > 0 }
      guard let lowest = nonzero.min(), let highest = nonzero.max(), lowest < floor, highest > floor
      else { return sigmas }
      let scale = (highest - floor) / (highest - lowest)
      return sigmas.map { $0 > 0 ? floor + ($0 - lowest) * scale : 0 }
    }

    /// comfy/samplers.py `simple_scheduler`: every (count / steps)-th entry of the model's sigma
    /// table, from the top.
    static func simple(_ sampling: ModelSampling, steps: Int) -> [Double] {
      let table = sampling.table
      let stride = Double(table.count) / Double(steps)
      var sigmas = (0..<steps).map { table[table.count - 1 - Int(Double($0) * stride)] }
      sigmas.append(0)
      return sigmas
    }

    /// comfy/samplers.py `normal_scheduler` (and `sgm_uniform`, its `sgm=True` form): timesteps
    /// evenly spaced from the model's largest sigma to its smallest.
    static func normal(_ sampling: ModelSampling, steps: Int, sgm: Bool) -> [Double] {
      let start = sampling.timestep(sigma: sampling.sigmaMax)
      let end = sampling.timestep(sigma: sampling.sigmaMin)
      var sigmas: [Double]
      if sgm {
        sigmas = linspace(start, end, count: steps + 1).dropLast().map {
          sampling.sigma(timestep: $0)
        }
        sigmas.append(0)
      } else if abs(sampling.sigma(timestep: end)) <= 0.00001 {
        // The table already ends at zero; ComfyUI spends the extra point on it.
        sigmas = linspace(start, end, count: steps + 1).map { sampling.sigma(timestep: $0) }
      } else {
        sigmas = linspace(start, end, count: steps).map { sampling.sigma(timestep: $0) }
        sigmas.append(0)
      }
      return sigmas
    }

    /// k_diffusion/sampling.py `get_sigmas_karras` (rho 7).
    static func karras(steps: Int, sigmaMin: Double, sigmaMax: Double, rho: Double = 7) -> [Double]
    {
      let minInvRho = pow(sigmaMin, 1 / rho)
      let maxInvRho = pow(sigmaMax, 1 / rho)
      return linspace(0, 1, count: steps).map { pow(maxInvRho + $0 * (minInvRho - maxInvRho), rho) }
        + [0]
    }

    /// k_diffusion/sampling.py `get_sigmas_exponential`.
    static func exponential(steps: Int, sigmaMin: Double, sigmaMax: Double) -> [Double] {
      linspace(log(sigmaMax), log(sigmaMin), count: steps).map { exp($0) } + [0]
    }

    /// comfy/samplers.py `linear_quadratic_schedule` (from Mochi): linear for the first half of the
    /// steps down to 0.025 below the top, then quadratic, scaled by the model's largest sigma.
    static func linearQuadratic(
      steps: Int, sigmaMax: Double, thresholdNoise: Double = 0.025
    ) -> [Double] {
      guard steps > 1 else { return [sigmaMax, 0] }
      let linearSteps = steps / 2
      let linear = (0..<linearSteps).map { Double($0) * thresholdNoise / Double(linearSteps) }
      let thresholdNoiseStepDiff = Double(linearSteps) - thresholdNoise * Double(steps)
      let quadraticSteps = Double(steps - linearSteps)
      let quadraticCoef =
        thresholdNoiseStepDiff / (Double(linearSteps) * quadraticSteps * quadraticSteps)
      let linearCoef =
        thresholdNoise / Double(linearSteps) - 2 * thresholdNoiseStepDiff
        / (quadraticSteps * quadraticSteps)
      let const = quadraticCoef * Double(linearSteps * linearSteps)
      let quadratic = (linearSteps..<steps).map {
        quadraticCoef * Double($0 * $0) + linearCoef * Double($0) + const
      }
      return (linear + quadratic + [1]).map { (1 - $0) * sigmaMax }
    }

    /// comfy/samplers.py `beta_scheduler` (arXiv 2407.12173, alpha = beta = 0.6): table entries
    /// at the Beta distribution's quantiles of evenly spaced probabilities. ComfyUI drops repeated
    /// entries (only at hundreds of steps); they are kept here so the schedule has `steps` steps.
    static func beta(_ sampling: ModelSampling, steps: Int, alpha: Double = 0.6, beta: Double = 0.6)
      -> [Double]
    {
      let table = sampling.table
      let total = Double(table.count - 1)
      var sigmas = (0..<steps).map { index -> Double in
        let probability = 1 - Double(index) / Double(steps)
        let t = (betaQuantile(probability, alpha, beta) * total).rounded(.toNearestOrEven)
        return table[min(max(Int(t), 0), table.count - 1)]
      }
      sigmas.append(0)
      return sigmas
    }

    /// numpy.linspace with endpoint (torch.linspace for the k-diffusion schedules).
    static func linspace(_ start: Double, _ end: Double, count: Int) -> [Double] {
      guard count > 1 else { return count == 1 ? [start] : [] }
      let step = (end - start) / Double(count - 1)
      return (0..<count).map { $0 == count - 1 ? end : start + Double($0) * step }
    }

    /// scipy.stats.beta.ppf: the inverse of the regularized incomplete beta function, by
    /// bisection (it is monotonic in x).
    public static func betaQuantile(_ probability: Double, _ a: Double, _ b: Double) -> Double {
      guard probability > 0 else { return 0 }
      guard probability < 1 else { return 1 }
      var low = 0.0
      var high = 1.0
      for _ in 0..<200 {
        let mid = (low + high) / 2
        guard mid > low, mid < high else { break }
        if regularizedIncompleteBeta(mid, a, b) < probability {
          low = mid
        } else {
          high = mid
        }
      }
      return (low + high) / 2
    }

    /// I_x(a, b), by the continued fraction of Numerical Recipes (`betai` / `betacf`).
    static func regularizedIncompleteBeta(_ x: Double, _ a: Double, _ b: Double) -> Double {
      guard x > 0 else { return 0 }
      guard x < 1 else { return 1 }
      let front = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1 - x))
      if x < (a + 1) / (a + b + 2) {
        return front * betaContinuedFraction(x, a, b) / a
      }
      return 1 - front * betaContinuedFraction(1 - x, b, a) / b
    }

    private static func betaContinuedFraction(_ x: Double, _ a: Double, _ b: Double) -> Double {
      let tiny = 1e-300
      let qab = a + b
      let qap = a + 1
      let qam = a - 1
      var c = 1.0
      var d = 1 - qab * x / qap
      if abs(d) < tiny { d = tiny }
      d = 1 / d
      var h = d
      for m in 1...500 {
        let m = Double(m)
        let m2 = 2 * m
        var aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        if abs(d) < tiny { d = tiny }
        c = 1 + aa / c
        if abs(c) < tiny { c = tiny }
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        if abs(d) < tiny { d = tiny }
        c = 1 + aa / c
        if abs(c) < tiny { c = tiny }
        d = 1 / d
        let delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15 { break }
      }
      return h
    }
  }
}

extension Denoiser.ScheduledDiscretization {
  /// The model's sigma(timestep), timestep(sigma) and 1000-entry sigma table, as ComfyUI's model
  /// sampling objects give them to the schedulers. Timesteps are in table units: 0…1 for flow
  /// (ComfyUI's timestep / multiplier), the training timestep index for DDPM.
  struct ModelSampling {
    enum Kind {
      case flow(shift: Double)
      /// The ascending grid, and the flow shift applied after the floor (`gridShift`).
      case grid([Double], shift: Double)
      case discrete(logSigmas: [Double])
      case edm(sigmaMin: Double, sigmaMax: Double)
    }
    let kind: Kind
    let table: [Double]

    init(_ parameterization: Denoiser.Parameterization, shift: Double, ascendingGrid: [Double]) {
      switch parameterization {
      case .rf(_):
        let kind: Kind =
          ascendingGrid.isEmpty ? .flow(shift: shift) : .grid(ascendingGrid, shift: shift)
        self.kind = kind
        // `ModelSamplingDiscreteFlow.set_parameters`: sigma at t = 1/1000 … 1.
        table = (1...1000).map { Self.sigma(kind, timestep: Double($0) / 1000) }
      case .ddpm(let ddpm):
        let sigmas = ddpm.alphasCumprod.map { ((1 - $0) / $0).squareRoot() }
        kind = .discrete(logSigmas: sigmas.map { log($0) })
        table = sigmas
      case .edm(let edm):
        kind = .edm(sigmaMin: edm.sigmaMin, sigmaMax: edm.sigmaMax)
        table = []
      }
    }

    /// Discrete and EDM sigmas take Draw Things' shift afterwards; flow models have it inside
    /// sigma(t), and MiniMax H3 Image takes it afterwards too (`gridShift`).
    var appliesShiftToSigmas: Bool {
      switch kind {
      case .discrete, .edm: return true
      case .flow, .grid: return false
      }
    }

    /// `sigma_min`: the table's first entry (ComfyUI's `sigmas[0]`).
    var sigmaMin: Double {
      switch kind {
      case .edm(let sigmaMin, _):
        return sigmaMin
      case .flow, .grid, .discrete:
        return table[0]
      }
    }

    /// The shift MiniMax H3 Image's schedule takes once the grid has given its sigmas and they're
    /// floored, as `timeSNRShift`: ComfyUI's `MiniMaxH3ImageModelSampling` has no shift, but Draw
    /// Things shifts H3 Image's schedule like any flow model's (`LinearDiscretization`), and
    /// shifting after the floor keeps the floor where the shift puts the grid's last point, as the
    /// RF `KarrasDiscretization` shifts its floored schedule. Nil for other models.
    var gridShift: Double? {
      if case .grid(_, let shift) = kind {
        return shift
      }
      return nil
    }

    /// comfy/model_sampling.py `time_snr_shift` for flow models (Draw Things' RF shift).
    static func timeSNRShift(_ sigma: Double, shift: Double) -> Double {
      shift == 1 ? sigma : shift * sigma / (1 + (shift - 1) * sigma)
    }

    /// MiniMax H3 Image's last nonzero grid point (about 0.23): it never takes input below it, and
    /// steps spent under it came out as noise, so every scheduler's lowest nonzero sigma stays at
    /// or above it, as `LocalImageGenerator.sampler(from:)` floors its Karras schedule. ComfyUI
    /// goes down to the table's first entry. Nil for other models.
    var sigmaFloor: Double? {
      if case .grid(let grid, _) = kind, grid.count >= 2 {
        return grid[1]
      }
      return nil
    }

    /// Where Karras and exponential end: the floor where there is one, which they then end on.
    var sigmaSpaceMin: Double {
      sigmaFloor ?? sigmaMin
    }

    var sigmaMax: Double {
      switch kind {
      case .edm(_, let sigmaMax):
        return sigmaMax
      case .flow, .grid, .discrete:
        return table[table.count - 1]
      }
    }

    /// comfy/model_sampling.py `sigma(timestep)` for each model sampling.
    func sigma(timestep: Double) -> Double {
      Self.sigma(kind, timestep: timestep)
    }

    private static func sigma(_ kind: Kind, timestep: Double) -> Double {
      switch kind {
      case .flow(let shift):
        return timeSNRShift(timestep, shift: shift)
      case .grid(let grid, _):
        // `MiniMaxH3ImageModelSampling.sigma` (unshifted; `gridShift` comes after the floor)
        let n = grid.count - 1
        let position = min(max(timestep, 0), 1) * Double(n)
        let index = min(Int(position.rounded(.down)), n - 1)
        return grid[index] + (grid[index + 1] - grid[index]) * (position - Double(index))
      case .discrete(let logSigmas):
        // `ModelSamplingDiscrete.sigma`
        let t = min(max(timestep, 0), Double(logSigmas.count - 1))
        let low = Int(t.rounded(.down))
        let high = Int(t.rounded(.up))
        let w = t - Double(low)
        return exp((1 - w) * logSigmas[low] + w * logSigmas[high])
      case .edm:
        return timestep
      }
    }

    /// comfy/model_sampling.py `timestep(sigma)`.
    func timestep(sigma: Double) -> Double {
      switch kind {
      case .flow, .grid, .edm:
        return sigma
      case .discrete(let logSigmas):
        // `ModelSamplingDiscrete.timestep`: the nearest training timestep.
        let logSigma = log(sigma)
        var best = 0
        for index in logSigmas.indices
        where abs(logSigma - logSigmas[index]) < abs(logSigma - logSigmas[best]) {
          best = index
        }
        return Double(best)
      }
    }
  }
}
