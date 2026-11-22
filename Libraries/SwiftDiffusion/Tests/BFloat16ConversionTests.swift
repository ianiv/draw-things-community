import Diffusion
import NNC
import XCTest

final class BFloat16ConversionTests: XCTestCase {
  // ccv_bfloat_to_half_precision reads its output buffer instead of its input, so a direct
  // BFloat16 -> Float16 conversion loses the values; Tensor(bfloat16SafeFrom:) goes by Float32.
  func testBFloat16ToFloat16PreservesValues() {
    let values: [Float] = [0, 1, -1, 0.5, -0.25, 3.140625, 1024, -0.0078125, 65280, 1e-3]
    let bf16 = Tensor<BFloat16>(
      from: Tensor<Float32>(values, .CPU, .C(values.count)))
    let expected = Tensor<Float32>(from: bf16)
    let half = Tensor<Float16>(bfloat16SafeFrom: bf16)
    XCTAssertEqual(half.dataType, .Float16)
    for i in 0..<values.count {
      XCTAssertEqual(Float(half[i]), Float(Float16(expected[i])), "index \(i)")
    }
  }

  func testOtherTypesUnchanged() {
    let values: [Float] = [0.1, -2.5, 7]
    let f32 = Tensor<Float32>(values, .CPU, .C(values.count))
    let half = Tensor<Float16>(bfloat16SafeFrom: f32)
    let direct = Tensor<Float16>(from: f32)
    for i in 0..<values.count {
      XCTAssertEqual(half[i], direct[i])
    }
    let back = Tensor<Float32>(bfloat16SafeFrom: Tensor<BFloat16>(from: f32))
    XCTAssertEqual(back.dataType, .Float32)
  }
}
