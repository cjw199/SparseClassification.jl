const DIR = @__DIR__

@info "Installing Jupyter kernel..."
using IJulia
installkernel("Julia_SSPOC_demo", "--project=" * DIR * "/../")
@info "Build complete."

