import ModelZoo
import XCTest

final class ModelZooTests: XCTestCase {
  func testLLMInkling2SmallCatalogAndEncoding() throws {
    let specification = try XCTUnwrap(LLMZoo.specificationForModel("inkling_2_small_i2x.ckpt"))
    XCTAssertEqual(specification.version.rawValue, "inkling_2_small")
    XCTAssertEqual(specification.requiredFiles, [specification.file])
    XCTAssertEqual(specification.files, [specification.file])
    XCTAssertEqual(LLMZoo.subtitleForModel(specification.file), "Thinking Machines Lab")
    XCTAssertFalse(LLMZoo.isModelDeprecated(specification.file))
    XCTAssertEqual(
      try JSONDecoder().decode(
        LLMZoo.Specification.self,
        from: JSONEncoder().encode(specification)), specification)
  }

  func testLLMGLM53FlashCatalogAndEncoding() throws {
    let specification = try XCTUnwrap(LLMZoo.specificationForModel("glm_5.3_flash_i4x.ckpt"))
    XCTAssertEqual(specification.version.rawValue, "glm_5.3_flash")
    XCTAssertEqual(specification.requiredFiles, [specification.file])
    XCTAssertEqual(specification.draftFile, "glm_5.3_flash_dflash2_i5x.ckpt")
    XCTAssertEqual(
      specification.files, [specification.file, "glm_5.3_flash_dflash2_i5x.ckpt"])
    XCTAssertEqual(LLMZoo.subtitleForModel(specification.file), "Z.ai")
    XCTAssertEqual(
      try JSONDecoder().decode(
        LLMZoo.Specification.self,
        from: JSONEncoder().encode(specification)), specification)
  }

  func testLLMDownloadRequiresEngramWithoutChangingPerFileChecks() {
    let file = "\(UUID().uuidString).ckpt"
    let engram = "\(UUID().uuidString).engram"
    let specification = LLMZoo.Specification(
      name: "Engram model", file: file, engram: engram, version: .deepseek_4_1_flash)

    XCTAssertFalse(LLMZoo.isModelDownloaded(specification))
    XCTAssertFalse(LLMZoo.isModelDownloaded(specification, memorizedBy: [file]))
    XCTAssertFalse(LLMZoo.isModelDownloaded(specification, memorizedBy: [engram]))
    XCTAssertTrue(LLMZoo.isModelDownloaded(specification, memorizedBy: [file, engram]))
    XCTAssertTrue(LLMZoo.isModelDownloaded(file, memorizedBy: [file]))
  }

  func testLLMSpecificationDecodesWithoutEngram() throws {
    let json = """
      {"name":"Existing model","file":"existing.ckpt","version":"deepseek_4_flash"}
      """
    let specification = try JSONDecoder().decode(
      LLMZoo.Specification.self, from: Data(json.utf8))

    XCTAssertNil(specification.engram)
    XCTAssertNil(specification.draftFile)
    XCTAssertTrue(LLMZoo.isModelDownloaded(specification, memorizedBy: [specification.file]))
  }

  func testLLMCanRunWithoutDownloadingOptionalDrafter() {
    let file = "\(UUID().uuidString).ckpt"
    let engram = "\(UUID().uuidString).engram"
    let drafter = "\(UUID().uuidString).ckpt"
    let specification = LLMZoo.Specification(
      name: "Drafting model", file: file, engram: engram, draftFile: drafter,
      version: .deepseek_4_1_flash)

    XCTAssertTrue(
      LLMZoo.isModelDownloaded(specification, required: true, memorizedBy: [file, engram]))
    XCTAssertFalse(LLMZoo.isModelDownloaded(specification, memorizedBy: [file, engram]))
    XCTAssertTrue(LLMZoo.isModelDownloaded(specification, memorizedBy: [file, engram, drafter]))

    XCTAssertFalse(
      LLMZoo.isModelDownloaded(specification, required: true, memorizedBy: [file, drafter]))
    XCTAssertFalse(
      LLMZoo.isModelDownloaded(specification, required: true, memorizedBy: [engram, drafter]))
  }

  func testMiniMaxH3FirstBlockCacheUsesIdentityCoefficients() {
    let name = "test-minimax-h3-community-cache.ckpt"
    let previous = ModelZoo.overrideMapping[name]
    defer { ModelZoo.overrideMapping[name] = previous }
    ModelZoo.overrideMapping[name] = ModelZoo.Specification(
      name: name, file: name, prefix: "", version: .minimaxH3)
    let coefficients = ModelZoo.teaCacheCoefficientsForModel(name)
    XCTAssertNotNil(coefficients)
    XCTAssertEqual(coefficients?.0, 0)
    XCTAssertEqual(coefficients?.1, 0)
    XCTAssertEqual(coefficients?.2, 0)
    XCTAssertEqual(coefficients?.3, 1)
    XCTAssertEqual(coefficients?.4, 0)
  }

  func testMiniMaxH3CheckpointSupportsFirstFrameConditioning() {
    XCTAssertEqual(ModelZoo.modifierForModel("minimax_h3_i8x.ckpt"), .fl2va)
  }

  func testMiniMaxH3DefaultLatentsScaling() {
    let name = "test-minimax-h3-default-latents.ckpt"
    let previous = ModelZoo.overrideMapping[name]
    defer { ModelZoo.overrideMapping[name] = previous }
    ModelZoo.overrideMapping[name] = ModelZoo.Specification(
      name: name, file: name, prefix: "", version: .minimaxH3)

    let scaling = ModelZoo.latentsScalingForModel(name)
    XCTAssertEqual(scaling.mean?.count, 24)
    XCTAssertEqual(scaling.std?.count, 24)
    XCTAssertEqual(scaling.audioMean?.count, 32)
    XCTAssertEqual(scaling.audioStd?.count, 32)
    XCTAssertEqual(scaling.mean?.first, 0.85809034)
    XCTAssertEqual(scaling.mean?.last, 0.7056032)
    XCTAssertEqual(scaling.std?.first, 1.2223774)
    XCTAssertEqual(scaling.std?.last, 2.6127844)
    XCTAssertEqual(scaling.audioMean?.first, -0.020211687)
    XCTAssertEqual(scaling.audioMean?.last, 0.39792585)
    XCTAssertEqual(scaling.audioStd?.first, 1.6895524)
    XCTAssertEqual(scaling.audioStd?.last, 1.5613768)
    XCTAssertEqual(scaling.scalingFactor, 1)
    XCTAssertNil(scaling.shiftFactor)

    let builtin = ModelZoo.latentsScalingForModel("minimax_h3_i8x.ckpt")
    XCTAssertEqual(scaling.mean, builtin.mean)
    XCTAssertEqual(scaling.std, builtin.std)
    XCTAssertEqual(scaling.audioMean, builtin.audioMean)
    XCTAssertEqual(scaling.audioStd, builtin.audioStd)
    XCTAssertEqual(scaling.scalingFactor, builtin.scalingFactor)
    XCTAssertEqual(scaling.shiftFactor, builtin.shiftFactor)
  }

  func testMiniMaxH3ExplicitLatentsScalingTakesPrecedence() {
    let name = "test-minimax-h3-explicit-latents.ckpt"
    let previous = ModelZoo.overrideMapping[name]
    defer { ModelZoo.overrideMapping[name] = previous }
    var specification = ModelZoo.Specification(
      name: name, file: name, prefix: "", version: .minimaxH3,
      latentsMean: Array(repeating: 0.5, count: 24),
      latentsStd: Array(repeating: 2, count: 24),
      audioLatentsMean: Array(repeating: -0.5, count: 32),
      audioLatentsStd: Array(repeating: 3, count: 32), latentsScalingFactor: 0.75)
    ModelZoo.overrideMapping[name] = specification

    let scaling = ModelZoo.latentsScalingForModel(name)
    XCTAssertEqual(scaling.mean, specification.latentsMean)
    XCTAssertEqual(scaling.std, specification.latentsStd)
    XCTAssertEqual(scaling.audioMean, specification.audioLatentsMean)
    XCTAssertEqual(scaling.audioStd, specification.audioLatentsStd)
    XCTAssertEqual(scaling.scalingFactor, 0.75)
    XCTAssertNil(scaling.shiftFactor)

    // Preserve the shared contract for specifications that override only scaling.
    specification.latentsMean = nil
    specification.latentsStd = nil
    ModelZoo.overrideMapping[name] = specification
    let scalingOnly = ModelZoo.latentsScalingForModel(name)
    XCTAssertNil(scalingOnly.mean)
    XCTAssertNil(scalingOnly.std)
    XCTAssertEqual(scalingOnly.scalingFactor, 0.75)
    XCTAssertEqual(scalingOnly.audioMean, specification.audioLatentsMean)
    XCTAssertEqual(scalingOnly.audioStd, specification.audioLatentsStd)
  }

  private func withLookup(
    externalUrls: [URL], order: [URL]?, preferred: Bool = false, _ body: () throws -> Void
  ) rethrows {
    let previousUrls = ModelZoo.externalUrls
    let previousOrder = ModelZoo.lookupUrlsOrder
    let previousPreference = ModelZoo.isExternalUrlsPreferred
    defer {
      ModelZoo.externalUrls = previousUrls
      ModelZoo.lookupUrlsOrder = previousOrder
      ModelZoo.isExternalUrlsPreferred = previousPreference
    }
    ModelZoo.externalUrls = externalUrls
    ModelZoo.lookupUrlsOrder = order
    ModelZoo.isExternalUrlsPreferred = preferred
    try body()
  }

  func testLookupUrlsWithoutOrderAreExternalUrls() {
    let a = URL(fileURLWithPath: "/lookup/a", isDirectory: true)
    let b = URL(fileURLWithPath: "/lookup/b", isDirectory: true)
    withLookup(externalUrls: [a, b], order: nil) {
      XCTAssertEqual(ModelZoo.lookupUrls, [a, b])
    }
  }

  func testLookupUrlsDropUnknownFoldersAndAppendUnorderedOnes() {
    let a = URL(fileURLWithPath: "/lookup/a", isDirectory: true)
    let b = URL(fileURLWithPath: "/lookup/b", isDirectory: true)
    let c = URL(fileURLWithPath: "/lookup/c", isDirectory: true)
    let gone = URL(fileURLWithPath: "/lookup/gone", isDirectory: true)
    withLookup(externalUrls: [a, b, c], order: [gone, c, a]) {
      XCTAssertEqual(ModelZoo.lookupUrls, [c, a, b])
    }
  }

  func testLookupOrderMovesModelFilesButNotCustomSpecifications() throws {
    let fileManager = FileManager.default
    let root = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(
      UUID().uuidString, isDirectory: true)
    let first = root.appendingPathComponent("first", isDirectory: true)
    let second = root.appendingPathComponent("second", isDirectory: true)
    try fileManager.createDirectory(at: first, withIntermediateDirectories: true)
    try fileManager.createDirectory(at: second, withIntermediateDirectories: true)
    defer { try? fileManager.removeItem(at: root) }
    let model = "lookup-\(UUID().uuidString).ckpt"
    let list = "custom-lookup-\(UUID().uuidString).json"
    for folder in [first, second] {
      for name in [model, list] {
        XCTAssertTrue(
          fileManager.createFile(atPath: folder.appendingPathComponent(name).path, contents: Data()))
      }
    }
    // Preferred, so Documents/Models isn't looked in first.
    withLookup(externalUrls: [first, second], order: [second], preferred: true) {
      XCTAssertEqual(
        ModelZoo.filePathForModelDownloaded(model), second.appendingPathComponent(model).path)
      XCTAssertEqual(
        ModelZoo.filePathForCustomSpecifications(list), first.appendingPathComponent(list).path)
      let missing = "lookup-missing-\(UUID().uuidString).ckpt"
      XCTAssertEqual(
        ModelZoo.filePathForModelDownloaded(missing), first.appendingPathComponent(missing).path)
    }
  }
}
