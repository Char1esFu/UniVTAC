# Header-only library
#
# NOTE: overlay port. Upstream vcpkg's registry entry for tinygltf 2.9.3 fetches the
# GitHub-generated source tarball and checks it against a recorded SHA512. GitHub
# regenerates those tarballs on the fly, and a change in its compression made the
# bytes -- and therefore the hash -- differ from what the registry recorded, so the
# stock port fails with a hash mismatch (upstream fixed this only for newer versions,
# see microsoft/vcpkg#53226).
#
# We fetch over git instead: the commit sha below IS git's own checksum of the tree,
# so no separate SHA512 is needed and repackaging cannot break the build again.
# 14ba271 is the commit tagged v2.9.3.
vcpkg_from_git(
    OUT_SOURCE_PATH SOURCE_PATH
    URL https://github.com/syoyo/tinygltf
    REF 14ba27113ebd507ebeb7cba8ecaeab8df3921b27
    HEAD_REF master
)

# Put the licence file where vcpkg expects it
# Copy the tinygltf header files and fix the path to json
vcpkg_replace_string("${SOURCE_PATH}/tiny_gltf.h" "#include \"json.hpp\"" "#include <nlohmann/json.hpp>")
file(INSTALL "${SOURCE_PATH}/tiny_gltf.h" DESTINATION "${CURRENT_PACKAGES_DIR}/include")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
