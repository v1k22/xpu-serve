# Sourced on the remote machines: locates OpenVINO and sets up its runtime environment.
if [ -d /opt/intel/openvino_2026.2.1 ]; then
    export XS_OV_ROOT=/opt/intel/openvino_2026.2.1
else
    export XS_OV_ROOT=$(ls -d "$HOME"/xpu-serve/deps/openvino_toolkit_*_2026.2.1.* 2>/dev/null | head -1)
fi
if [ -z "$XS_OV_ROOT" ] || [ ! -f "$XS_OV_ROOT/setupvars.sh" ]; then
    echo "remote_env.sh: OpenVINO 2026.2.1 not found" >&2
    return 1
fi
# shellcheck disable=SC1091
source "$XS_OV_ROOT/setupvars.sh" >/dev/null
export XS_TEST_MODEL_DIR=${XS_TEST_MODEL_DIR:-$HOME/xpu-serve/models/gemma-4-E2B-it}
