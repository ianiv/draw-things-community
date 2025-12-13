import * as flatbuffers from "flatbuffers";

export interface GenerationConfig {
  [key: string]: any;
}

export enum SamplerType {
  DPMPP2MKarras = 0,
  EulerA = 1,
  DDIM = 2,
  PLMS = 3,
  DPMPPSDEKarras = 4,
  UniPC = 5,
  LCM = 6,
  EulerASubstep = 7,
  DPMPPSDESubstep = 8,
  TCD = 9,
  EulerATrailing = 10,
  DPMPPSDETrailing = 11,
  DPMPP2MAYS = 12,
  EulerAAYS = 13,
  DPMPPSDEAYS = 14,
  DPMPP2MTrailing = 15,
  DDIMTrailing = 16,
  UniPCTrailing = 17,
  UniPCAYS = 18,
}

export enum SeedMode {
  Legacy = 0,
  TorchCpuCompatible = 1,
  ScaleAlike = 2,
  NvidiaGpuCompatible = 3,
}

export enum ControlMode {
  Balanced = 0,
  Prompt = 1,
  Control = 2,
}

export enum LoRAMode {
  All = 0,
  Base = 1,
  Refiner = 2,
}

const SAMPLER_MAP: { [key: string]: SamplerType } = {
  dpmpp_2m_karras: SamplerType.DPMPP2MKarras,
  euler_a: SamplerType.EulerA,
  ddim: SamplerType.DDIM,
  plms: SamplerType.PLMS,
  dpmpp_sde_karras: SamplerType.DPMPPSDEKarras,
  unipc: SamplerType.UniPC,
  lcm: SamplerType.LCM,
};

export class ConfigurationBuilder {
  private builder: flatbuffers.Builder;

  constructor() {
    this.builder = new flatbuffers.Builder(2048);
  }

  build(config: GenerationConfig): Buffer {
    // Create string fields first
    let modelOffset: number | null = null;
    if (config.model) {
      modelOffset = this.builder.createString(config.model);
    }

    let upscalerOffset: number | null = null;
    if (config.upscaler) {
      upscalerOffset = this.builder.createString(config.upscaler);
    }

    let faceRestorationOffset: number | null = null;
    if (config.face_restoration) {
      faceRestorationOffset = this.builder.createString(
        config.face_restoration
      );
    }

    let refinerModelOffset: number | null = null;
    if (config.refiner_model) {
      refinerModelOffset = this.builder.createString(config.refiner_model);
    }

    let nameOffset: number | null = null;
    if (config.name) {
      nameOffset = this.builder.createString(config.name);
    }

    // Note: Full FlatBuffer implementation would include controls and loras vectors
    // For now, we're creating a minimal configuration that covers the most common use cases

    // Start building GenerationConfiguration table
    this.builder.startObject(80); // Approximate number of fields

    // Add fields
    if (config.id !== undefined) {
      this.builder.addFieldInt32(1, config.id, 0);
    }

    if (config.start_width !== undefined) {
      this.builder.addFieldInt16(2, config.start_width, 0);
    }

    if (config.start_height !== undefined) {
      this.builder.addFieldInt16(3, config.start_height, 0);
    }

    if (config.seed !== undefined) {
      this.builder.addFieldInt32(4, config.seed, 0);
    }

    if (config.steps !== undefined) {
      this.builder.addFieldInt32(5, config.steps, 0);
    }

    if (config.guidance_scale !== undefined) {
      this.builder.addFieldFloat32(6, config.guidance_scale, 0);
    }

    if (config.strength !== undefined) {
      this.builder.addFieldFloat32(7, config.strength, 0);
    }

    if (modelOffset !== null) {
      this.builder.addFieldOffset(8, modelOffset, 0);
    }

    // Sampler
    const sampler = config.sampler || "dpmpp_2m_karras";
    const samplerVal = SAMPLER_MAP[sampler] || SamplerType.DPMPP2MKarras;
    this.builder.addFieldInt8(9, samplerVal, 0);

    // Batch settings
    this.builder.addFieldInt32(10, config.batch_count || 1, 1);
    this.builder.addFieldInt32(11, config.batch_size || 1, 1);

    if (config.hires_fix) {
      this.builder.addFieldInt8(12, config.hires_fix ? 1 : 0, 0);
    }

    if (config.hires_fix_start_width !== undefined) {
      this.builder.addFieldInt16(13, config.hires_fix_start_width, 0);
    }

    if (config.hires_fix_start_height !== undefined) {
      this.builder.addFieldInt16(14, config.hires_fix_start_height, 0);
    }

    if (config.hires_fix_strength !== undefined) {
      this.builder.addFieldFloat32(15, config.hires_fix_strength, 0);
    }

    if (upscalerOffset !== null) {
      this.builder.addFieldOffset(16, upscalerOffset, 0);
    }

    this.builder.addFieldFloat32(17, config.image_guidance_scale || 1.5, 1.5);

    // Seed mode
    const seedMode = config.seed_mode || SeedMode.Legacy;
    this.builder.addFieldInt8(18, seedMode, 0);

    this.builder.addFieldInt32(19, config.clip_skip || 1, 1);

    if (faceRestorationOffset !== null) {
      this.builder.addFieldOffset(23, faceRestorationOffset, 0);
    }

    this.builder.addFieldFloat32(26, config.clip_weight || 1.0, 1.0);

    if (refinerModelOffset !== null) {
      this.builder.addFieldOffset(28, refinerModelOffset, 0);
    }

    if (config.refiner_start !== undefined) {
      this.builder.addFieldFloat32(36, config.refiner_start, 0);
    }

    if (nameOffset !== null) {
      this.builder.addFieldOffset(38, nameOffset, 0);
    }

    if (config.shift !== undefined) {
      this.builder.addFieldFloat32(45, config.shift, 0);
    }

    if (config.tiled_decoding !== undefined) {
      this.builder.addFieldInt8(50, config.tiled_decoding ? 1 : 0, 0);
    }

    const configObj = this.builder.endObject();

    // Finish the buffer
    this.builder.finish(configObj);

    return this.builder.asUint8Array();
  }

  private startObject(numFields: number): void {
    this.builder.startObject(numFields);
  }

  private endObject(): number {
    return this.builder.endObject();
  }
}
