# create_grpc_proto.py
import subprocess
import sys
from pathlib import Path


def _generated_grpc_path(proto_file: str, output_dir: str) -> Path:
    proto_path = Path(proto_file)
    relative_proto_path = proto_path
    if relative_proto_path.is_absolute():
        try:
            relative_proto_path = relative_proto_path.relative_to(Path.cwd())
        except ValueError:
            relative_proto_path = Path(relative_proto_path.name)
    return Path(output_dir) / relative_proto_path.with_name(
        f"{proto_path.stem}_pb2_grpc.py"
    )


def _patch_package_import(proto_file: str, output_dir: str):
    grpc_file = _generated_grpc_path(proto_file, output_dir)
    pb2_module = f"{Path(proto_file).stem}_pb2"
    old_import = f"import {pb2_module} as model__rpc__service__pb2"
    new_import = (
        f"from rtp_llm.cpp.model_rpc.proto import {pb2_module} "
        "as model__rpc__service__pb2"
    )

    text = grpc_file.read_text()
    if old_import in text:
        grpc_file.write_text(text.replace(old_import, new_import))


def main():
    if len(sys.argv) != 3:
        print("Usage: create_grpc_proto.py <proto_file>")
        sys.exit(1)

    proto_file = sys.argv[1]
    output_dir = sys.argv[2]

    subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            "-I.",
            f"--python_out={output_dir}",
            f"--grpc_python_out={output_dir}",
            proto_file,
        ],
        check=True,
    )
    _patch_package_import(proto_file, output_dir)


if __name__ == "__main__":
    main()
