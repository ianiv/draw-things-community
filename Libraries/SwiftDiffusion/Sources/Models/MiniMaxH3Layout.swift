// Block layout of MiniMax H3 checkpoints, including compressed variants such as HyperFlow H3 Flash.
import Foundation
import NNC

/// Describes which transformer blocks a MiniMax H3 checkpoint keeps, which of their linear layers are
/// factored into low-rank pairs, and how its timestep is embedded.
///
/// The stock checkpoint keeps all 50 blocks with dense linears and a single timestep embedder.
/// HyperFlow H3 Flash drops a few blocks, replaces some linears with `up(down(x))` pairs, shrinks the
/// time embedding to 16 channels and adds an endpoint embedder for the step target `r`
/// (`temb = emb_t(t) + gate * (emb_r(r) - emb_t(t))`).
public struct MiniMaxH3Layout: Equatable, Sendable {
  /// Ranks of the factored linears of one block. Zero means the linear is dense.
  public struct Ranks: Equatable, Sendable {
    public var q: Int
    public var k: Int
    public var v: Int
    public var o: Int
    public var fc1: Int
    public var fc2: Int
    public init(q: Int = 0, k: Int = 0, v: Int = 0, o: Int = 0, fc1: Int = 0, fc2: Int = 0) {
      self.q = q
      self.k = k
      self.v = v
      self.o = o
      self.fc1 = fc1
      self.fc2 = fc2
    }
    public static let dense = Ranks()
    public var isDense: Bool { self == .dense }
    /// Checkpoints fuse q, k and v into one projection only when none of them is factored.
    public var splitsQKV: Bool { q > 0 || k > 0 || v > 0 }
    var values: [Int] { [q, k, v, o, fc1, fc2] }
  }

  /// Original indices (0..<50) of the blocks this checkpoint keeps, ascending.
  public var layers: [Int]
  /// Factored ranks keyed by original block index. Blocks that are missing are dense.
  public var ranks: [Int: Ranks]
  /// Output width of the time embedder (2,688 for the stock model, 16 for HyperFlow).
  public var timeEmbeddingSize: Int
  /// Interpolation gate of the endpoint embedder. Nil for single-time checkpoints.
  public var endpointGate: Float?
  /// False for MiniMax H3 Image, which has no audio stream (no audio projections in or out).
  public var hasAudio: Bool
  /// Sigma grid (descending, 1 to 0) a PDD checkpoint was distilled on. It has one stacked output
  /// head per interval: the first is a full head, the others are offsets from it. Empty when the
  /// checkpoint has a single output head.
  public var sigmaGrid: [Double]
  /// Rows of the `adaln_t_table` of curve-form checkpoints, which replace the time embedder with a
  /// table of the time-embedding curve in a small shared basis (`timeEmbeddingSize` wide), sampled
  /// uniformly over t in [0, 1]. Zero when the checkpoint has a time embedder.
  public var timeCurveGrid: Int

  public init(
    layers: [Int], ranks: [Int: Ranks], timeEmbeddingSize: Int, endpointGate: Float?,
    hasAudio: Bool = true, sigmaGrid: [Double] = [], timeCurveGrid: Int = 0
  ) {
    self.layers = layers
    self.ranks = ranks.filter { !$0.value.isDense }
    self.timeEmbeddingSize = timeEmbeddingSize
    self.endpointGate = endpointGate
    self.hasAudio = hasAudio
    self.sigmaGrid = sigmaGrid
    self.timeCurveGrid = timeCurveGrid
  }

  public static let standard = MiniMaxH3Layout(
    layers: Array(0..<50), ranks: [:], timeEmbeddingSize: 2_688, endpointGate: nil)

  public var isStandard: Bool { self == .standard }
  public var isTwoTime: Bool { endpointGate != nil }
  /// Stacked output heads (one per sigma-grid interval), 1 for ordinary checkpoints.
  public var outputHeads: Int { max(1, sigmaGrid.count - 1) }
  public var usesTimeCurve: Bool { timeCurveGrid > 0 }

  /// The sigma grid of the MiniMax-H3-Image release (`pdd_config.json`).
  public static let imageSigmaGrid: [Double] = [
    1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132,
    0.0,
  ]

  public func ranks(ofLayer layer: Int) -> Ranks {
    ranks[layer] ?? .dense
  }

  /// The parameter names a block at `position` instantiates, in addition to the norms.
  func linearNames(atPosition position: Int) -> [String] {
    let ranks = ranks(ofLayer: layers[position])
    var names = [String]()
    for (rank, name) in zip([ranks.q, ranks.k, ranks.v, ranks.o], ["q", "k", "v", "o"]) {
      names += rank > 0 ? ["\(name)_down", "\(name)_up"] : [name]
    }
    names += ranks.fc1 > 0 ? ["fc1_down", "gate_lr", "up_lr"] : ["up", "gate"]
    names += ranks.fc2 > 0 ? ["fc2_down", "fc2_up"] : ["down"]
    return names
  }

  /// The occurrence index NNC assigns to a parameter named `name` in the block at `position`
  /// of the full graph. Partial graphs (startLayer > 0) use it as the model's start index so
  /// their parameter names line up with the full checkpoint.
  func occurrence(of name: String, atPosition position: Int) -> Int {
    guard !ranks.isEmpty else { return position }
    return (0..<position).reduce(0) { count, position in
      count + (linearNames(atPosition: position).contains(name) ? 1 : 0)
    }
  }

  // MARK: - Serialization

  /// The checkpoint key holding a serialized non-standard layout.
  public static let storeKey = "minimax_h3_layout"

  /// Version 1: `[1, gate (or -1), timeEmbeddingSize, layerCount, (layer, q, k, v, o, fc1, fc2)...]`.
  /// Version 2 (written only when a version 1 reader would misread the checkpoint) puts
  /// `hasAudio, timeCurveGrid, sigmaCount, sigmas...` between the header and the layer rows.
  public var values: [Float] {
    let isVersion1 = hasAudio && sigmaGrid.isEmpty && timeCurveGrid == 0
    var values: [Float] = [
      isVersion1 ? 1 : 2, endpointGate ?? -1, Float(timeEmbeddingSize), Float(layers.count),
    ]
    if !isVersion1 {
      values += [hasAudio ? 1 : 0, Float(timeCurveGrid), Float(sigmaGrid.count)]
      values += sigmaGrid.map { Float($0) }
    }
    for layer in layers {
      values.append(Float(layer))
      values += ranks(ofLayer: layer).values.map { Float($0) }
    }
    return values
  }

  public init?(values: [Float]) {
    guard values.count >= 4, values[0] == 1 || values[0] == 2 else { return nil }
    let count = Int(values[3])
    var offset = 4
    var hasAudio = true
    var timeCurveGrid = 0
    var sigmaGrid = [Double]()
    if values[0] == 2 {
      guard values.count >= 7 else { return nil }
      hasAudio = values[4] != 0
      timeCurveGrid = Int(values[5])
      let sigmaCount = Int(values[6])
      guard sigmaCount >= 0, values.count >= 7 + sigmaCount else { return nil }
      sigmaGrid = values[7..<(7 + sigmaCount)].map { Double($0) }
      offset = 7 + sigmaCount
    }
    guard count > 0, values.count == offset + count * 7 else { return nil }
    var layers = [Int]()
    var ranks = [Int: Ranks]()
    for i in 0..<count {
      let row = values[(offset + i * 7)..<(offset + (i + 1) * 7)].map { Int($0) }
      layers.append(row[0])
      ranks[row[0]] = Ranks(q: row[1], k: row[2], v: row[3], o: row[4], fc1: row[5], fc2: row[6])
    }
    self.init(
      layers: layers, ranks: ranks, timeEmbeddingSize: Int(values[2]),
      endpointGate: values[1] < 0 ? nil : values[1], hasAudio: hasAudio, sigmaGrid: sigmaGrid,
      timeCurveGrid: timeCurveGrid)
  }

  public func write(to store: DynamicGraph.Store) {
    guard !isStandard else { return }
    let values = values
    var tensor = Tensor<Float>(.CPU, .C(values.count))
    for (i, value) in values.enumerated() {
      tensor[i] = value
    }
    store.write(Self.storeKey, tensor: tensor)
  }

  public static func read(from store: DynamicGraph.Store) -> MiniMaxH3Layout {
    guard let tensor = store.read(Self.storeKey) else { return .standard }
    let values = Tensor<Float>(from: tensor)
    guard let layout = MiniMaxH3Layout(values: (0..<values.shape[0]).map { values[$0] }) else {
      return .standard
    }
    return layout
  }

  private final class Cache: @unchecked Sendable {
    let lock = NSLock()
    var layouts = [String: MiniMaxH3Layout]()
  }
  private static let cache = Cache()

  /// The layout of the checkpoint at `filePath`, cached per path.
  public static func load(filePath: String) -> MiniMaxH3Layout {
    cache.lock.lock()
    if let layout = cache.layouts[filePath] {
      cache.lock.unlock()
      return layout
    }
    cache.lock.unlock()
    var layout = MiniMaxH3Layout.standard
    if FileManager.default.fileExists(atPath: filePath) {
      let graph = DynamicGraph()
      graph.openStore(
        filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
      ) {
        layout = read(from: $0)
      }
    }
    cache.lock.lock()
    cache.layouts[filePath] = layout
    cache.lock.unlock()
    return layout
  }

  // MARK: - Sampling

  /// HyperFlow's raw (unshifted) sigma grids, keyed by step count. The model is trained on these
  /// grids only; the sampler shifts them with the usual flow shift (12 for video).
  public static let hyperFlowSigmaGrids: [Int: [Double]] = [
    4: [1.0, 0.681872, 0.205019, 0.036496, 0.0],
    6: [1.0, 0.825217, 0.471725, 0.147401, 0.049626, 0.015517, 0.0],
    8: [1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0],
  ]

  /// The raw grid for `steps`, or nil when HyperFlow has no grid for that count.
  public static func hyperFlowSigmaGrid(steps: Int) -> [Double]? {
    hyperFlowSigmaGrids[steps]
  }

  /// The checkpoint key holding the `adaln_t_table` of a curve-form checkpoint.
  public static let timeCurveStoreKey = "minimax_h3_adaln_t_table"

  /// Sampling grids of a PDD checkpoint keyed by step count: its own grid, plus every coarser grid
  /// that only uses its points (a step then spans several heads, see `headSelector`).
  public var sigmaGrids: [Int: [Double]] {
    let heads = sigmaGrid.count - 1
    guard heads > 1 else { return [:] }
    var grids = [Int: [Double]]()
    for steps in 1...heads where heads % steps == 0 {
      grids[steps] = stride(from: 0, through: heads, by: heads / steps).map { sigmaGrid[$0] }
    }
    return grids
  }

  /// Weights of the stacked output heads for the step from `sigma` to `nextSigma`, following
  /// ComfyUI's `_pdd_head`: the first head always counts in full (the others are offsets from it)
  /// and a step takes the dt-weighted mean of the heads of the grid intervals it spans.
  public func headSelector(sigma: Double, nextSigma: Double) -> [Float] {
    let heads = outputHeads
    guard heads > 1 else { return [1] }
    func nearest(_ value: Double) -> Int {
      sigmaGrid.indices.min { abs(sigmaGrid[$0] - value) < abs(sigmaGrid[$1] - value) }!
    }
    let start = min(nearest(sigma), heads - 1)
    let stop = max(nearest(nextSigma), start + 1)
    let deltas = (start..<stop).map { sigmaGrid[$0] - sigmaGrid[$0 + 1] }
    let total = deltas.reduce(0, +)
    var selector = [Float](repeating: 0, count: heads)
    selector[0] = 1
    for index in max(start, 1)..<stop {
      selector[index] = Float(deltas[index - start] / total)
    }
    return selector
  }

  /// Rows of the time-embedding curve at `t`, interpolated like ComfyUI's `_time_embedding`.
  public static func timeCurve(_ table: Tensor<Float>, at t: Float) -> [Float] {
    let rows = table.shape[0]
    let size = table.shape[1]
    let position = min(max(t, 0), 1) * Float(rows - 1)
    let lower = min(Int(position.rounded(.down)), rows - 2)
    let fraction = position - Float(lower)
    return (0..<size).map {
      table[lower, $0] + (table[lower + 1, $0] - table[lower, $0]) * fraction
    }
  }
}

/// Fixed conditions the H3 fixed encoder returns after the leading text / reference outputs. They
/// end with the output shift and scale of the video and audio streams, or without audio (MiniMax H3
/// Image) with those of the image and the output-head selector.
public func MiniMaxH3FixedConditionCount(
  layers: Int, referenceImageCount: Int, referenceAudioCount: Int, hasAudio: Bool = true
) -> Int {
  layers
    * MiniMaxH3PerLayerConditionCount(
      referenceImageCount: referenceImageCount, referenceAudioCount: referenceAudioCount,
      hasAudio: hasAudio)
    + (hasAudio ? 4 : 3)
}

public func MiniMaxH3PerLayerConditionCount(
  referenceImageCount: Int, referenceAudioCount: Int, hasAudio: Bool = true
) -> Int {
  (hasAudio ? 18 : 12) + (referenceImageCount > 0 ? 6 : 0) + (referenceAudioCount > 0 ? 6 : 0)
}

/// Recovers the layer count from a fixed-condition count (layouts may drop blocks).
public func MiniMaxH3LayerCount(
  fixedConditionCount: Int, referenceImageCount: Int, referenceAudioCount: Int,
  hasAudio: Bool = true
) -> Int {
  let perLayer = MiniMaxH3PerLayerConditionCount(
    referenceImageCount: referenceImageCount, referenceAudioCount: referenceAudioCount,
    hasAudio: hasAudio)
  let tail = hasAudio ? 4 : 3
  precondition(fixedConditionCount > tail && (fixedConditionCount - tail) % perLayer == 0)
  return (fixedConditionCount - tail) / perLayer
}
