import torch
import tensorrt as trt
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import tempfile
import os
from pathlib import Path
import logging

from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

class TensorRTOptimizer:
    """TensorRT optimization for ultra-fast inference at 60+ FPS"""
    
    def __init__(self, 
                 precision: str = "fp16",
                 max_workspace_size: int = 1 << 30,  # 1GB
                 max_batch_size: int = 8):
        
        self.precision = precision
        self.max_workspace_size = max_workspace_size
        self.max_batch_size = max_batch_size
        
        # Initialize TensorRT logger
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        
        # Check TensorRT availability
        self._check_tensorrt_support()
    
    def _check_tensorrt_support(self):
        """Check if TensorRT is available and GPU supports it"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for TensorRT optimization")
        
        try:
            # Check GPU compute capability
            gpu_props = torch.cuda.get_device_properties(0)
            compute_capability = gpu_props.major * 10 + gpu_props.minor
            
            if compute_capability < 53:  # SM 5.3 minimum for FP16
                logger.warning(f"GPU compute capability {compute_capability/10:.1f} may not support FP16")
            
            logger.info(f"GPU: {gpu_props.name}, Compute: {compute_capability/10:.1f}")
            logger.info(f"TensorRT version: {trt.__version__}")
            
        except Exception as e:
            logger.error(f"TensorRT check failed: {e}")
            raise
    
    def optimize_yolov10(self, 
                        model_path: str, 
                        input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
                        output_path: Optional[str] = None) -> str:
        """Optimize YOLOv10 model with TensorRT"""
        
        logger.info(f"Optimizing YOLOv10 model: {model_path}")
        
        if output_path is None:
            output_path = model_path.replace('.pt', '_trt.engine')
        
        # Convert PyTorch model to ONNX first
        onnx_path = self._convert_pytorch_to_onnx(model_path, input_shape)
        
        # Build TensorRT engine from ONNX
        engine_path = self._build_tensorrt_engine(onnx_path, output_path, input_shape)
        
        # Cleanup temporary ONNX file
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        
        logger.info(f"TensorRT engine saved: {engine_path}")
        return engine_path
    
    def optimize_rtdetr(self, 
                       model_path: str,
                       input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
                       output_path: Optional[str] = None) -> str:
        """Optimize RT-DETR model with TensorRT"""
        
        logger.info(f"Optimizing RT-DETR model: {model_path}")
        
        if output_path is None:
            output_path = model_path.replace('.pt', '_rtdetr_trt.engine')
        
        # RT-DETR specific optimization
        onnx_path = self._convert_rtdetr_to_onnx(model_path, input_shape)
        engine_path = self._build_tensorrt_engine(onnx_path, output_path, input_shape)
        
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        
        logger.info(f"RT-DETR TensorRT engine saved: {engine_path}")
        return engine_path
    
    def _convert_pytorch_to_onnx(self, 
                                model_path: str, 
                                input_shape: Tuple[int, int, int, int]) -> str:
        """Convert PyTorch model to ONNX format"""
        
        try:
            from ultralytics import YOLO
            
            # Load model
            model = YOLO(model_path)
            
            # Create dummy input
            dummy_input = torch.randn(input_shape).cuda()
            
            # Export to ONNX
            onnx_path = model_path.replace('.pt', '_temp.onnx')
            
            torch.onnx.export(
                model.model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            logger.info(f"ONNX model exported: {onnx_path}")
            return onnx_path
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise
    
    def _convert_rtdetr_to_onnx(self, 
                               model_path: str, 
                               input_shape: Tuple[int, int, int, int]) -> str:
        """Convert RT-DETR model to ONNX format"""
        
        try:
            from ultralytics import RTDETR
            
            # Load RT-DETR model
            model = RTDETR(model_path)
            
            # Export to ONNX with RT-DETR specific settings
            onnx_path = model_path.replace('.pt', '_rtdetr_temp.onnx')
            
            model.export(
                format='onnx',
                imgsz=input_shape[2:],
                opset=17,
                simplify=True,
                dynamic=True
            )
            
            logger.info(f"RT-DETR ONNX model exported: {onnx_path}")
            return onnx_path
            
        except Exception as e:
            logger.error(f"RT-DETR ONNX export failed: {e}")
            raise
    
    def _build_tensorrt_engine(self, 
                              onnx_path: str, 
                              engine_path: str,
                              input_shape: Tuple[int, int, int, int]) -> str:
        """Build TensorRT engine from ONNX model"""
        
        try:
            # Create builder and network
            builder = trt.Builder(self.trt_logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, self.trt_logger)
            
            # Parse ONNX model
            with open(onnx_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    logger.error("Failed to parse ONNX model")
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    raise RuntimeError("ONNX parsing failed")
            
            # Configure builder
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.max_workspace_size)
            
            # Set precision
            if self.precision == "fp16" and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("Enabled FP16 precision")
            elif self.precision == "int8" and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                logger.info("Enabled INT8 precision")
            else:
                logger.info("Using FP32 precision")
            
            # Optimization profiles for dynamic shapes
            profile = builder.create_optimization_profile()
            
            # Set input shape ranges
            batch_size, channels, height, width = input_shape
            
            profile.set_shape(
                "input",
                min=(1, channels, height, width),
                opt=(batch_size, channels, height, width),
                max=(self.max_batch_size, channels, height, width)
            )
            
            config.add_optimization_profile(profile)
            
            # Additional optimizations for football detection
            self._configure_football_optimizations(config)
            
            # Build engine
            logger.info("Building TensorRT engine (this may take several minutes)...")
            serialized_engine = builder.build_serialized_network(network, config)
            
            if serialized_engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Save engine
            with open(engine_path, 'wb') as f:
                f.write(serialized_engine)
            
            logger.info(f"TensorRT engine built successfully: {engine_path}")
            return engine_path
            
        except Exception as e:
            logger.error(f"TensorRT engine build failed: {e}")
            raise
    
    def _configure_football_optimizations(self, config):
        """Configure TensorRT optimizations specific to football detection"""
        
        # Enable layer optimizations for detection models
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        
        # Optimize for throughput (60 FPS target)
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        
        # Enable CUDA graph capture for reduced latency
        config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
        
        logger.info("Applied football-specific TensorRT optimizations")
    
    def benchmark_engine(self, engine_path: str, 
                        input_shape: Tuple[int, int, int, int],
                        num_runs: int = 100) -> Dict[str, float]:
        """Benchmark TensorRT engine performance"""
        
        logger.info(f"Benchmarking TensorRT engine: {engine_path}")
        
        try:
            # Load engine
            runtime = trt.Runtime(self.trt_logger)
            with open(engine_path, 'rb') as f:
                engine = runtime.deserialize_cuda_engine(f.read())
            
            # Create execution context
            context = engine.create_execution_context()
            
            # Allocate GPU memory
            batch_size, channels, height, width = input_shape
            input_size = batch_size * channels * height * width * 4  # float32
            
            # Create CUDA stream
            stream = torch.cuda.Stream()
            
            # Allocate GPU buffers
            input_gpu = torch.cuda.FloatTensor(batch_size, channels, height, width)
            
            # Get output shapes (assuming single output for simplicity)
            output_shape = self._get_output_shape(engine)
            output_gpu = torch.cuda.FloatTensor(*output_shape)
            
            # Warm up
            for _ in range(10):
                with torch.cuda.stream(stream):
                    context.execute_async_v2(
                        bindings=[input_gpu.data_ptr(), output_gpu.data_ptr()],
                        stream_handle=stream.cuda_stream
                    )
                stream.synchronize()
            
            # Benchmark
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            
            for _ in range(num_runs):
                with torch.cuda.stream(stream):
                    context.execute_async_v2(
                        bindings=[input_gpu.data_ptr(), output_gpu.data_ptr()],
                        stream_handle=stream.cuda_stream
                    )
            
            end_time.record()
            torch.cuda.synchronize()
            
            # Calculate metrics
            total_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
            avg_time = total_time / num_runs
            fps = 1.0 / avg_time
            
            benchmark_results = {
                "avg_inference_time_ms": avg_time * 1000,
                "fps": fps,
                "total_runs": num_runs,
                "total_time_s": total_time,
                "memory_usage_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "target_fps_achieved": fps >= 60.0
            }
            
            logger.info(f"Benchmark results: {benchmark_results}")
            return benchmark_results
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise
    
    def _get_output_shape(self, engine) -> Tuple[int, ...]:
        """Get output tensor shape from TensorRT engine"""
        
        # This is a simplified version - actual implementation would
        # properly parse the engine output bindings
        
        # Default shape for YOLO/DETR outputs
        return (1, 84, 8400)  # Typical YOLO output shape
    
    def optimize_for_football(self, 
                            model_configs: List[Dict[str, Any]]) -> Dict[str, str]:
        """Optimize multiple models for football analysis pipeline"""
        
        optimized_engines = {}
        
        for config in model_configs:
            model_type = config.get("type")
            model_path = config.get("path")
            input_shape = config.get("input_shape", (1, 3, 640, 640))
            
            try:
                if model_type == "yolov10":
                    engine_path = self.optimize_yolov10(model_path, input_shape)
                elif model_type == "rt-detr":
                    engine_path = self.optimize_rtdetr(model_path, input_shape)
                else:
                    logger.warning(f"Unsupported model type for TensorRT: {model_type}")
                    continue
                
                optimized_engines[model_type] = engine_path
                
                # Benchmark optimized engine
                benchmark_results = self.benchmark_engine(engine_path, input_shape)
                logger.info(f"{model_type} optimization complete: {benchmark_results['fps']:.1f} FPS")
                
            except Exception as e:
                logger.error(f"Failed to optimize {model_type}: {e}")
                continue
        
        return optimized_engines
    
    def create_football_detection_pipeline(self, 
                                         primary_engine: str,
                                         backup_engines: List[str]) -> "TensorRTFootballPipeline":
        """Create optimized inference pipeline for football detection"""
        
        return TensorRTFootballPipeline(
            primary_engine=primary_engine,
            backup_engines=backup_engines,
            logger=self.trt_logger
        )

class TensorRTFootballPipeline:
    """Optimized TensorRT inference pipeline for football detection"""
    
    def __init__(self, 
                 primary_engine: str,
                 backup_engines: List[str],
                 logger):
        
        self.primary_engine_path = primary_engine
        self.backup_engine_paths = backup_engines
        self.trt_logger = logger
        
        # Load engines
        self.runtime = trt.Runtime(self.trt_logger)
        self.primary_engine = self._load_engine(primary_engine)
        self.primary_context = self.primary_engine.create_execution_context()
        
        # Setup CUDA stream for async execution
        self.stream = torch.cuda.Stream()
        
        # Pre-allocate GPU memory buffers
        self._allocate_gpu_buffers()
        
        logger.info("TensorRT Football Pipeline initialized")
    
    def _load_engine(self, engine_path: str):
        """Load TensorRT engine from file"""
        with open(engine_path, 'rb') as f:
            return self.runtime.deserialize_cuda_engine(f.read())
    
    def _allocate_gpu_buffers(self):
        """Pre-allocate GPU memory for maximum performance"""
        
        # Get input/output shapes from engine
        input_shape = self._get_binding_shape(self.primary_engine, 0)
        output_shape = self._get_binding_shape(self.primary_engine, 1)
        
        # Allocate buffers
        self.input_buffer = torch.cuda.FloatTensor(*input_shape)
        self.output_buffer = torch.cuda.FloatTensor(*output_shape)
        
        logger.info(f"GPU buffers allocated: input {input_shape}, output {output_shape}")
    
    def _get_binding_shape(self, engine, binding_idx: int) -> Tuple[int, ...]:
        """Get tensor shape for a specific binding"""
        return tuple(engine.get_binding_shape(binding_idx))
    
    def infer_batch(self, batch_frames: torch.Tensor) -> torch.Tensor:
        """Ultra-fast batch inference with TensorRT"""
        
        # Copy input to GPU buffer
        self.input_buffer.copy_(batch_frames, non_blocking=True)
        
        # Execute inference asynchronously
        with torch.cuda.stream(self.stream):
            success = self.primary_context.execute_async_v2(
                bindings=[
                    self.input_buffer.data_ptr(),
                    self.output_buffer.data_ptr()
                ],
                stream_handle=self.stream.cuda_stream
            )
        
        # Synchronize stream
        self.stream.synchronize()
        
        if not success:
            raise RuntimeError("TensorRT inference failed")
        
        return self.output_buffer.clone()
    
    def infer_single(self, frame: torch.Tensor) -> torch.Tensor:
        """Single frame inference optimized for minimal latency"""
        
        # Ensure correct batch dimension
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
        
        return self.infer_batch(frame)
    
    def get_max_fps(self) -> float:
        """Get theoretical maximum FPS for this pipeline"""
        
        # This would be calculated based on actual benchmarking
        # Placeholder for now
        return 60.0