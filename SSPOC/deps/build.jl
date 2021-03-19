const DIR = @__DIR__
using Pkg
Pkg.activate(DIR*"/../")
@info "Installing required packages..."
Pkg.resolve()
Pkg.instantiate()
Pkg.status()

@info "Installing Jupyter kernel..."
using IJulia
installkernel("Julia_SSPOC_demo", "--project=" * DIR * "/../")
@info "Build complete."

