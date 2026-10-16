"""Recording fakes of vendor SDK modules (TensorRT, MIGraphX, AI Hub, mlx-lm ...).

They implement just the API surface the stages call and record every call in ``calls`` so
tests can assert on behaviour (flags, profiles, options) without the hardware.
"""

from __future__ import annotations

from types import SimpleNamespace


def fake_tensorrt(version: str = "10.13.0", parse_ok: bool = True) -> SimpleNamespace:
    """TensorRT module; ``version`` 10.x keeps precision BuilderFlags, 11.x removes them."""
    calls: dict = {"flags": [], "profiles": {}, "config": {}}
    major = int(version.split(".")[0])
    flag_names = ["VERSION_COMPATIBLE", "GPU_FALLBACK"] + (
        ["FP16", "BF16", "INT8", "FP8"] if major < 11 else []
    )
    creation = {"STRONGLY_TYPED": 5} if major >= 11 else {"STRONGLY_TYPED": 5, "EXPLICIT_BATCH": 0}

    inputs = [SimpleNamespace(name="x")]
    network = SimpleNamespace(num_inputs=1, get_input=lambda i: inputs[i])

    def set_flag(flag):
        calls["flags"].append(flag)

    config = SimpleNamespace(
        set_memory_pool_limit=lambda pool, size: calls["config"].update(workspace=size),
        set_flag=set_flag,
        add_optimization_profile=lambda p: calls.update(profile_added=True),
        builder_optimization_level=None,
        hardware_compatibility_level=None,
        default_device_type=None,
        DLA_core=None,
    )
    profile = SimpleNamespace(
        set_shape=lambda name, mn, opt, mx: calls["profiles"].update({name: (mn, opt, mx)})
    )

    def create_network(flags):
        calls["network_flags"] = flags
        return network

    builder = SimpleNamespace(
        create_network=create_network,
        create_builder_config=lambda: config,
        create_optimization_profile=lambda: profile,
        build_serialized_network=lambda net, cfg: b"ENGINE",
    )
    parser = SimpleNamespace(
        parse_from_file=lambda path: parse_ok,
        num_errors=0 if parse_ok else 1,
        get_error=lambda i: "unsupported op Foo",
    )
    module = SimpleNamespace(
        __version__=version,
        calls=calls,
        config=config,
        Logger=type("Logger", (), {"WARNING": 2, "__init__": lambda self, level=None: None}),
        Builder=lambda logger: builder,
        OnnxParser=lambda net, logger: parser,
        NetworkDefinitionCreationFlag=SimpleNamespace(**creation),
        BuilderFlag=SimpleNamespace(**{n: n for n in flag_names}),
        MemoryPoolType=SimpleNamespace(WORKSPACE="workspace"),
        HardwareCompatibilityLevel=SimpleNamespace(AMPERE_PLUS="ampere_plus"),
        DeviceType=SimpleNamespace(DLA="dla"),
    )
    return module


def fake_migraphx() -> SimpleNamespace:
    calls: dict = {}
    program = SimpleNamespace(compile=lambda target, **kw: calls.update(compile=(target, kw)))
    return SimpleNamespace(
        calls=calls,
        parse_onnx=lambda path, map_input_dims: calls.update(dims=map_input_dims) or program,
        get_target=lambda name: f"target:{name}",
        quantize_fp16=lambda prog: calls.update(quantized="fp16"),
        quantize_bf16=lambda prog: calls.update(quantized="bf16"),
        quantize_int8=lambda prog, target, data: calls.update(quantized="int8", calib=len(data)),
        argument=lambda array: array,
        save=lambda prog, path, format: open(path, "wb").write(b"MXR"),  # noqa: SIM115
    )


def fake_qai_hub(tmp_path) -> SimpleNamespace:
    calls: dict = {}

    def download(path):
        open(path, "wb").write(b"DLC")  # noqa: SIM115
        return path

    target_model = SimpleNamespace(download=download)
    compile_job = SimpleNamespace(
        job_id="j123", get_target_model=lambda: target_model, get_status=lambda: "done"
    )

    def submit_compile_job(**kwargs):
        calls["compile"] = kwargs
        return compile_job

    def submit_profile_job(**kwargs):
        calls["profile"] = kwargs
        return SimpleNamespace(
            download_profile=lambda: {"execution_summary": {"estimated_inference_time": 1234}}
        )

    return SimpleNamespace(
        calls=calls,
        Device=lambda name=None, attributes=None: f"device:{name or attributes}",
        submit_compile_job=submit_compile_job,
        submit_profile_job=submit_profile_job,
    )


def fake_converter_package(accepted: tuple[str, ...]) -> SimpleNamespace:
    """``mlx_lm`` / ``mlx_vlm`` stand-in whose ``convert`` accepts only ``accepted`` kwargs."""
    calls: dict = {}
    params = ", ".join(f"{name}=None" for name in accepted)
    namespace: dict = {"calls": calls}
    exec(f"def convert({params}):\n    calls.update(locals())\n", namespace)  # noqa: S102 - builds a real signature
    return SimpleNamespace(convert=namespace["convert"], calls=calls)
