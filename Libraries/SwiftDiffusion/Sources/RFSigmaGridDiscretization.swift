import Foundation

extension Denoiser {
  /// Rectified-flow discretization that samples a fixed raw sigma grid per step count, as
  /// few-step distilled models (HyperFlow H3 Flash) require. The grid is shifted with the usual
  /// flow shift `shift * s / (1 + (shift - 1) * s)`. Step counts without a grid fall back to the
  /// trailing linear schedule.
  public struct RFSigmaGridDiscretization: Discretization {
    private let linearDiscretization: LinearDiscretization
    private let grids: [Int: [Double]]
    /// False when the grids are final sigmas (MiniMax H3 Image), not raw ones to shift.
    private let shiftsGrid: Bool
    public var timesteps: Float { linearDiscretization.timesteps }
    public var objective: Objective { linearDiscretization.objective }
    public init(
      _ parameterization: Parameterization, objective: Objective, grids: [Int: [Double]],
      shiftsGrid: Bool = true
    ) {
      linearDiscretization = LinearDiscretization(
        parameterization, objective: objective, timestepSpacing: .trailing)
      self.grids = grids
      self.shiftsGrid = shiftsGrid
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
      guard var grid = grids[steps], grid.count == steps + 1 else {
        guard shiftsGrid else {
          // A PDD checkpoint only has output heads for its own grid points: other step counts
          // sample its finest grid.
          let finest = grids.keys.max().flatMap { grids[$0] } ?? []
          precondition(!finest.isEmpty)
          return finest.map { 1 - $0 }
        }
        return linearDiscretization.alphasCumprod(steps: steps, shift: shift)
      }
      if !shiftsGrid {
        grid = grid.map { $0 }
        return grid.map { 1 - $0 }
      }
      // RF keeps alphaCumprod = 1 - sigma; the grid already ends at sigma = 0.
      return grid.map { sigma in
        let shifted = shift != 1 ? shift * sigma / (1 + (shift - 1) * sigma) : sigma
        return 1 - shifted
      }
    }
  }
}
