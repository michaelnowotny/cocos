#!/usr/bin/env bash
#
# Install only the CPU backend of ArrayFire (about 0.5 GB of the 13.8 GB
# release) into the directory given as the first argument. CI runners have no
# GPU, so they test cocos on the CPU backend.
#
#   install-arrayfire-cpu.sh <prefix> [<local copy of the installer>]

set -euo pipefail

prefix=$1
installer=${2:-}
version=3.10.0
sha256=f1289b0969d1c6db3bf52e961076f7b241764c80e5d509e363a122a219a82288

download_dir=$(mktemp -d)
trap 'rm -rf "$download_dir"' EXIT

if [ -z "$installer" ]; then
    installer=$download_dir/ArrayFire-v${version}_Linux_x86_64.sh
    curl -fL --retry 5 --no-progress-meter -o "$installer" \
        "https://arrayfire.s3.amazonaws.com/${version}/ArrayFire-v${version}_Linux_x86_64.sh"
fi
echo "${sha256}  ${installer}" | sha256sum -c -

# The installer is a shell script followed by a gzipped tar archive.
offset=$(head -c 65536 "$installer" \
         | sed -n 's/^tail \$use_new_tail_syntax +\([0-9]*\) "\$0" | gunzip .*/\1/p')
test "$offset" -gt 0

mkdir -p "$prefix"
tail -n +"$offset" "$installer" | gunzip | tar -x -C "$prefix" --wildcards \
    --exclude='lib64/libmkl_sycl*' \
    'LICENSES/*' 'lib64/libaf.so*' 'lib64/libafcpu.so*' 'lib64/libmkl_*' \
    'lib64/libtbb*' 'lib64/libhwloc*' 'lib64/libimf.so' 'lib64/libsvml.so' \
    'lib64/libintlc.so*' 'lib64/libirng.so'

du -sh "$prefix"
