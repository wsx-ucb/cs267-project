{
  description = "Development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    cudtwxx-src = {
      flake = false;
      url = "github:asbschmidt/cuDTW";
    };
  };

  outputs = { self, nixpkgs, flake-utils, cudtwxx-src }:
    flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      cudtwxx = pkgs.stdenv.mkDerivation {
        pname = "cudtw++";
        version = "1.0";
        src = cudtwxx-src;
        installPhase = ''
          mkdir -p $out
          cp -r $src/src/include $out/include
        '';
      };
    in {
      devShells.default = pkgs.mkShell {
        packages = with pkgs; [ cmakeCurses clang-tools_15 ];
        buildInputs = with pkgs; [ llvmPackages_15.openmp mpich hdf5-cpp cista cudaPackages.cudatoolkit cudaPackages.nsight_compute nvidia-thrust cudtwxx ];
      };
    });
}
