import Foundation

extension Denoiser {
  /// Rectified-flow discretization that samples a fixed raw sigma grid per step count, as
  /// few-step distilled models (HyperFlow H3 Flash) require. The grid is shifted with the usual
  /// flow shift `shift * s / (1 + (shift - 1) * s)`. Step counts without a grid fall back to the
  /// trailing linear schedule.
  public struct RFSigmaGridDiscretization: Discretization {
    private let linearDiscretization: LinearDiscretization
    private let grids: [Int: [Double]]
    public var timesteps: Float { linearDiscretization.timesteps }
    public var objective: Objective { linearDiscretization.objective }
    public init(_ parameterization: Parameterization, objective: Objective, grids: [Int: [Double]])
    {
      linearDiscretization = LinearDiscretization(
        parameterization, objective: objective, timestepSpacing: .trailing)
      self.grids = grids
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
      guard let grid = grids[steps], grid.count == steps + 1 else {
        return linearDiscretization.alphasCumprod(steps: steps, shift: shift)
      }
      // RF keeps alphaCumprod = 1 - sigma; the grid already ends at sigma = 0.
      return grid.map { sigma in
        let shifted = shift != 1 ? shift * sigma / (1 + (shift - 1) * sigma) : sigma
        return 1 - shifted
      }
    }
  }
}
