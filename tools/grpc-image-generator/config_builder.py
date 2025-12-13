"""
FlatBuffer builder for GenerationConfiguration.
Manually constructs the FlatBuffer to avoid schema compilation issues.
"""
import flatbuffers
from typing import Optional, List, Dict, Any


# Enum values from config.fbs
class SamplerType:
    DPMPP2MKarras = 0
    EulerA = 1
    DDIM = 2
    PLMS = 3
    DPMPPSDEKarras = 4
    UniPC = 5
    LCM = 6
    EulerASubstep = 7
    DPMPPSDESubstep = 8
    TCD = 9
    EulerATrailing = 10
    DPMPPSDETrailing = 11
    DPMPP2MAYS = 12
    EulerAAYS = 13
    DPMPPSDEAYS = 14
    DPMPP2MTrailing = 15
    DDIMTrailing = 16
    UniPCTrailing = 17
    UniPCAYS = 18


class SeedMode:
    Legacy = 0
    TorchCpuCompatible = 1
    ScaleAlike = 2
    NvidiaGpuCompatible = 3


class ControlMode:
    Balanced = 0
    Prompt = 1
    Control = 2


class LoRAMode:
    All = 0
    Base = 1
    Refiner = 2


SAMPLER_MAP = {
    "dpmpp_2m_karras": SamplerType.DPMPP2MKarras,
    "euler_a": SamplerType.EulerA,
    "ddim": SamplerType.DDIM,
    "plms": SamplerType.PLMS,
    "dpmpp_sde_karras": SamplerType.DPMPPSDEKarras,
    "unipc": SamplerType.UniPC,
    "lcm": SamplerType.LCM,
}


class ConfigurationBuilder:
    """Builds a GenerationConfiguration FlatBuffer."""

    def __init__(self):
        self.builder = flatbuffers.Builder(2048)

    def build(self, config: Dict[str, Any]) -> bytes:
        """
        Build a GenerationConfiguration FlatBuffer from a configuration dict.

        Args:
            config: Dictionary with generation configuration

        Returns:
            Serialized FlatBuffer bytes
        """
        # Create string fields first
        model_offset = None
        if config.get('model'):
            model_offset = self.builder.CreateString(config['model'])

        upscaler_offset = None
        if config.get('upscaler'):
            upscaler_offset = self.builder.CreateString(config['upscaler'])

        face_restoration_offset = None
        if config.get('face_restoration'):
            face_restoration_offset = self.builder.CreateString(config['face_restoration'])

        refiner_model_offset = None
        if config.get('refiner_model'):
            refiner_model_offset = self.builder.CreateString(config['refiner_model'])

        name_offset = None
        if config.get('name'):
            name_offset = self.builder.CreateString(config['name'])

        # Build controls vector if present
        controls_offset = None
        if config.get('controls'):
            controls_offsets = []
            for ctrl in config['controls']:
                ctrl_offset = self._build_control(ctrl)
                controls_offsets.append(ctrl_offset)
            # Create vector
            flatbuffers.Builder.StartVector(self.builder, len(controls_offsets), 4, 4)
            for ctrl_offset in reversed(controls_offsets):
                self.builder.PrependUOffsetTRelative(ctrl_offset)
            controls_offset = self.builder.EndVector()

        # Build loras vector if present
        loras_offset = None
        if config.get('loras'):
            loras_offsets = []
            for lora in config['loras']:
                lora_offset = self._build_lora(lora)
                loras_offsets.append(lora_offset)
            # Create vector
            flatbuffers.Builder.StartVector(self.builder, len(loras_offsets), 4, 4)
            for lora_offset in reversed(loras_offsets):
                self.builder.PrependUOffsetTRelative(lora_offset)
            loras_offset = self.builder.EndVector()

        # Start building GenerationConfiguration table
        # Table offset 4 (this is approximate, actual offsets depend on schema)
        self.builder.StartObject(80)  # Approximate number of fields

        # Add fields (order doesn't matter for FlatBuffers)
        # Field offsets are based on the schema order

        # Field 0: id (long)
        if 'id' in config:
            self._add_long(1, config['id'])

        # Field 1: start_width (ushort)
        if 'start_width' in config:
            self._add_ushort(2, config['start_width'])

        # Field 2: start_height (ushort)
        if 'start_height' in config:
            self._add_ushort(3, config['start_height'])

        # Field 3: seed (uint)
        if 'seed' in config:
            self._add_uint(4, config['seed'])

        # Field 4: steps (uint)
        if 'steps' in config:
            self._add_uint(5, config['steps'])

        # Field 5: guidance_scale (float)
        if 'guidance_scale' in config:
            self._add_float(6, config['guidance_scale'])

        # Field 6: strength (float)
        if 'strength' in config:
            self._add_float(7, config['strength'])

        # Field 7: model (string)
        if model_offset is not None:
            self._add_offset(8, model_offset)

        # Field 8: sampler (SamplerType)
        sampler = config.get('sampler', 'dpmpp_2m_karras')
        sampler_val = SAMPLER_MAP.get(sampler, SamplerType.DPMPP2MKarras)
        self._add_byte(9, sampler_val)

        # Field 9: batch_count (uint, default 1)
        self._add_uint(10, config.get('batch_count', 1))

        # Field 10: batch_size (uint, default 1)
        self._add_uint(11, config.get('batch_size', 1))

        # Field 11: hires_fix (bool)
        if 'hires_fix' in config:
            self._add_bool(12, config['hires_fix'])

        # Field 12-14: hires_fix dimensions and strength
        if 'hires_fix_start_width' in config:
            self._add_ushort(13, config['hires_fix_start_width'])
        if 'hires_fix_start_height' in config:
            self._add_ushort(14, config['hires_fix_start_height'])
        if 'hires_fix_strength' in config:
            self._add_float(15, config['hires_fix_strength'])

        # Field 15: upscaler (string)
        if upscaler_offset is not None:
            self._add_offset(16, upscaler_offset)

        # Field 16: image_guidance_scale (float, default 1.5)
        self._add_float(17, config.get('image_guidance_scale', 1.5))

        # Field 17: seed_mode (SeedMode, default Legacy)
        seed_mode = config.get('seed_mode', SeedMode.Legacy)
        self._add_byte(18, seed_mode)

        # Field 18: clip_skip (uint, default 1)
        self._add_uint(19, config.get('clip_skip', 1))

        # Field 19: controls
        if controls_offset is not None:
            self._add_offset(20, controls_offset)

        # Field 20: loras
        if loras_offset is not None:
            self._add_offset(21, loras_offset)

        # Field 21: mask_blur (float)
        if 'mask_blur' in config:
            self._add_float(22, config['mask_blur'])

        # Field 22: face_restoration (string)
        if face_restoration_offset is not None:
            self._add_offset(23, face_restoration_offset)

        # Field 25: clip_weight (float, default 1.0)
        self._add_float(26, config.get('clip_weight', 1.0))

        # Field 27: refiner_model (string)
        if refiner_model_offset is not None:
            self._add_offset(28, refiner_model_offset)

        # Field 35: refiner_start (float, default 0.7)
        if 'refiner_start' in config:
            self._add_float(36, config['refiner_start'])

        # Field 37: name (string)
        if name_offset is not None:
            self._add_offset(38, name_offset)

        # Additional common fields
        if 'shift' in config:
            self._add_float(45, config['shift'])

        if 'tiled_decoding' in config:
            self._add_bool(50, config['tiled_decoding'])

        config_obj = self.builder.EndObject()

        # Finish the buffer
        self.builder.Finish(config_obj)

        return bytes(self.builder.Output())

    def _build_control(self, ctrl: Dict[str, Any]) -> int:
        """Build a Control table."""
        file_offset = self.builder.CreateString(ctrl['file'])

        self.builder.StartObject(8)
        self._add_offset(0, file_offset)
        self._add_float(1, ctrl.get('weight', 1.0))
        self._add_float(2, ctrl.get('guidance_start', 0.0))
        self._add_float(3, ctrl.get('guidance_end', 1.0))
        self._add_bool(4, ctrl.get('no_prompt', False))
        self._add_bool(5, ctrl.get('global_average_pooling', True))
        self._add_float(6, ctrl.get('down_sampling_rate', 1.0))
        control_mode = ctrl.get('control_mode', ControlMode.Balanced)
        self._add_byte(7, control_mode)

        return self.builder.EndObject()

    def _build_lora(self, lora: Dict[str, Any]) -> int:
        """Build a LoRA table."""
        file_offset = self.builder.CreateString(lora['file'])

        self.builder.StartObject(3)
        self._add_offset(0, file_offset)
        self._add_float(1, lora.get('weight', 0.6))
        lora_mode = lora.get('mode', LoRAMode.All)
        self._add_byte(2, lora_mode)

        return self.builder.EndObject()

    def _add_byte(self, field: int, value: int):
        self.builder.PrependByteSlot(field, value, 0)

    def _add_ushort(self, field: int, value: int):
        self.builder.PrependUint16Slot(field, value, 0)

    def _add_uint(self, field: int, value: int):
        self.builder.PrependUint32Slot(field, value, 0)

    def _add_long(self, field: int, value: int):
        self.builder.PrependUint64Slot(field, value, 0)

    def _add_float(self, field: int, value: float):
        self.builder.PrependFloat32Slot(field, value, 0.0)

    def _add_bool(self, field: int, value: bool):
        self.builder.PrependBoolSlot(field, value, False)

    def _add_offset(self, field: int, offset: int):
        self.builder.PrependUOffsetTRelativeSlot(field, offset, 0)
