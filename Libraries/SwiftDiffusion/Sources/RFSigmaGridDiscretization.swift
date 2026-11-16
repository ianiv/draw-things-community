import Foundation

extension Denoiser {
  /// Rectified-flow discretization that samples a fixed raw sigma grid per step count, as
  /// few-step distilled models (HyperFlow H3 Flash, MiniMax H3 Image) require. The grid is shifted
  /// with the usual flow shift `shift * s / (1 + (shift - 1) * s)` (as `LinearDiscretization` does
  /// for RF), so shift 1 samples it as trained. Step counts without a grid follow the finest grid
  /// stretched to `steps + 1` points when `stretchesFinestGrid` (MiniMax H3 Image, whose heads and
  /// distillation follow that curve; a linear schedule came out distorted and over-contrasted),
  /// else the trailing linear schedule (HyperFlow). Either way the result has `steps + 1` entries.
  public struct RFSigmaGridDiscretization: Discretization {
    private let linearDiscretization: LinearDiscretization
    private let grids: [Int: [Double]]
    private let stretchesFinestGrid: Bool
    public var timesteps: Float { linearDiscretization.timesteps }
    public var objective: Objective { linearDiscretization.objective }
    public init(
      _ parameterization: Parameterization, objective: Objective, grids: [Int: [Double]],
      stretchesFinestGrid: Bool = false
    ) {
      linearDiscretization = LinearDiscretization(
        parameterization, objective: objective, timestepSpacing: .trailing)
      self.grids = grids
      self.stretchesFinestGrid = stretchesFinestGrid
    }

    /// The finest grid resampled to `steps + 1` sigmas, piecewise-linearly over each point's
    /// position in the schedule (0 at the first step, 1 at the end), so the points keep the
    /// trained curve's shape. Nil without a usable grid.
    static func stretched(_ finest: [Double], steps: Int) -> [Double]? {
      guard steps > 0, finest.count >= 2 else { return nil }
      let intervals = Double(finest.count - 1)
      return (0...steps).map { index in
        let position = Double(index) / Double(steps) * intervals
        let lower = min(Int(position.rounded(.down)), finest.count - 2)
        let fraction = position - Double(lower)
        return finest[lower] + (finest[lower + 1] - finest[lower]) * fraction
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
      var grid = grids[steps].flatMap { $0.count == steps + 1 ? $0 : nil }
      if grid == nil, stretchesFinestGrid, let finest = grids.keys.max().flatMap({ grids[$0] }) {
        grid = Self.stretched(finest, steps: steps)
      }
      guard let grid, grid.count == steps + 1 else {
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
