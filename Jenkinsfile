def getgputargets() {
    targets="gfx906;gfx908;gfx90a;gfx1030;gfx1100;gfx1101;gfx1102"
    return targets
}

def getnavi3xtargets() {
    targets="gfx1100;gfx1101;gfx1102"
    return targets
}

// Test
// def rocmtestnode(variant, name, body, args, pre) {
def rocmtestnode(Map conf) {
    def variant = conf.get("variant")
    def name = conf.get("node")
    def body = conf.get("body")
    def docker_args = conf.get("docker_args", "")
    def docker_build_args = conf.get("docker_build_args", "")
    def pre = conf.get("pre", {})
    def ccache = "/home/jenkins/workspace/.cache/ccache"
    def image = 'migraphxlib'
    env.CCACHE_COMPRESSLEVEL = 7
    env.CCACHE_DIR = ccache
    def cmake_build = { bconf ->
        def compiler = bconf.get("compiler", "/opt/rocm/llvm/bin/clang++")
        def flags = bconf.get("flags", "")
        def gpu_debug = bconf.get("gpu_debug", "0")
        def cmd = """
            ulimit -c unlimited
            echo "leak:dnnl::impl::malloc" > suppressions.txt
            echo "leak:libtbb.so" >> suppressions.txt
            cat suppressions.txt
            export LSAN_OPTIONS="suppressions=\$(pwd)/suppressions.txt"
            export MIGRAPHX_GPU_DEBUG=${gpu_debug}
            export CXX=${compiler}
            export CXXFLAGS='-Werror'
            env
            rm -rf build
            mkdir build
            cd build
            cmake -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DBUILD_DEV=On -DCMAKE_EXECUTE_PROCESS_COMMAND_ECHO=STDOUT -DMIGRAPHX_DISABLE_VIRTUAL_ENV=ON ${flags} ..
            git diff
            git diff-index --quiet HEAD || (echo "Git repo is not clean after running cmake." && exit 1)
            make -j\$(nproc) generate VERBOSE=1
            git diff
            git diff-index --quiet HEAD || (echo "Generated files are different. Please run make generate and commit the changes." && exit 1)
            make -j\$(nproc) all doc package check VERBOSE=1
            md5sum ./*.deb
        """
        echo cmd
        sh cmd
        // Only archive from master or develop
        if (env.BRANCH_NAME == "develop" || env.BRANCH_NAME == "master") {
            archiveArtifacts artifacts: "build/*.deb", allowEmptyArchive: true, fingerprint: true
        }
    }
    node(name) {
        withEnv(['HSA_ENABLE_SDMA=0']) {
            stage("checkout ${variant}") {
                checkout scm
            }
            gitStatusWrapper(credentialsId: "${env.status_wrapper_creds}", gitHubContext: "Jenkins - ${variant}", account: 'ROCmSoftwarePlatform', repo: 'AMDMIGraphX') {
                pre()
                stage("image ${variant}") {
                    try {
                        docker.build("${image}", "${docker_build_args} .")
                    } catch(Exception ex) {
                        docker.build("${image}", "${docker_build_args} --no-cache .")

                    }
                }
                withDockerContainer(image: image, args: "--device=/dev/kfd --device=/dev/dri --group-add video --cap-add SYS_PTRACE -v=/home/jenkins:/home/jenkins ${docker_args}") {
                    timeout(time: 2, unit: 'HOURS') {
                        body(cmake_build)
                    }
                }
            }
        }
    }
}
def rocmtest(m) {
    def builders = [:]
    m.each { e ->
        def label = e.key;
        def action = e.value;
        builders[label] = {
            action(label)
        }
    }
    parallel builders
}

def rocmnodename(name) {
    def rocmtest_name = "(rocmtest || migraphx)"
    def node_name = "${rocmtest_name}"
    if(name == "fiji") {
        node_name = "${rocmtest_name} && fiji";
    } else if(name == "vega") {
        node_name = "${rocmtest_name} && vega";
    } else if(name == "navi21") {
        node_name = "${rocmtest_name} && navi21";
    } else if(name == "mi100+") {
        node_name = "${rocmtest_name} && (gfx908 || gfx90a) && !vm";
    } else if(name == "cdna") {
        node_name = "${rocmtest_name} && (gfx908 || gfx90a || vega20) && !vm";
    } else if(name == "navi32") {
        node_name = "${rocmtest_name} && gfx1101 && !vm";
    } else if(name == "nogpu") {
        node_name = "${rocmtest_name} && nogpu";
    }
    return node_name
}

def rocmnode(name, body) {
    return { label ->
        rocmtestnode(variant: label, node: rocmnodename(name), body: body)
    }
}

rocmtest clang_debug: rocmnode('mi100+') { cmake_build ->
    stage('hipRTC Debug') {
        // Disable MLIR since it doesnt work with all ub sanitizers
        withEnv(['MIGRAPHX_DISABLE_MLIR=1']) {
            def sanitizers = "undefined"
            def debug_flags = "-g -O2 -fsanitize=${sanitizers} -fno-sanitize-recover=${sanitizers}"
            def gpu_targets = getgputargets()
            cmake_build(flags: "-DCMAKE_BUILD_TYPE=debug -DMIGRAPHX_ENABLE_PYTHON=Off -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}' -DCMAKE_C_FLAGS_DEBUG='${debug_flags}' -DMIGRAPHX_USE_HIPRTC=On -DGPU_TARGETS='${gpu_targets}'", gpu_debug: true)
        }
    }
}, clang_release: rocmnode('mi100+') { cmake_build ->
    stage('Hip Clang Release') {
        def gpu_targets = getgputargets()
        cmake_build(flags: "-DCMAKE_BUILD_TYPE=release -DGPU_TARGETS='${gpu_targets}'")
        stash includes: 'build/*.deb', name: 'migraphx-package'
    }
// }, hidden_symbols: rocmnode('cdna') { cmake_build ->
//     stage('Hidden symbols') {
//         cmake_build(flags: "-DMIGRAPHX_ENABLE_PYTHON=Off -DMIGRAPHX_ENABLE_GPU=On -DMIGRAPHX_ENABLE_CPU=On -DCMAKE_CXX_VISIBILITY_PRESET=hidden -DCMAKE_C_VISIBILITY_PRESET=hidden")
//     }
}, all_targets_debug : rocmnode('mi100+') { cmake_build ->
    stage('All targets Release') {
        def gpu_targets = getgputargets()
        cmake_build(flags: "-DCMAKE_BUILD_TYPE=release -DMIGRAPHX_ENABLE_GPU=On -DMIGRAPHX_ENABLE_CPU=On -DMIGRAPHX_ENABLE_FPGA=On -DGPU_TARGETS='${gpu_targets}'")
    }
}, mlir_debug: rocmnode('mi100+') { cmake_build ->
    stage('MLIR Debug') {
        withEnv(['MIGRAPHX_ENABLE_EXTRA_MLIR=1', 'MIGRAPHX_MLIR_USE_SPECIFIC_OPS=fused,attention,convolution,dot']) {
            def sanitizers = "undefined"
            // Note: the -fno-sanitize= is copied from upstream LLVM_UBSAN_FLAGS.
            def debug_flags_cxx = "-g -O2 -fsanitize=${sanitizers} -fno-sanitize=vptr,function -fno-sanitize-recover=${sanitizers}"
            def debug_flags = "-g -O2 -fsanitize=${sanitizers} -fno-sanitize=vptr -fno-sanitize-recover=${sanitizers}"
            def gpu_targets = getgputargets()
            // Since the purpose of this run verify all things MLIR supports,
            // enabling all possible types of offloads
            cmake_build(flags: "-DCMAKE_BUILD_TYPE=debug -DMIGRAPHX_ENABLE_PYTHON=Off -DMIGRAPHX_ENABLE_MLIR=On -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags_cxx}' -DCMAKE_C_FLAGS_DEBUG='${debug_flags}' -DGPU_TARGETS='${gpu_targets}'")
        }
    }
}, ck_hiprtc: rocmnode('mi100+') { cmake_build ->
    stage('CK hipRTC') {
        withEnv(['MIGRAPHX_ENABLE_CK=1', 'MIGRAPHX_TUNE_CK=1', 'MIGRAPHX_DISABLE_MLIR=1']) {
            def gpu_targets = getgputargets()
            cmake_build(flags: "-DCMAKE_BUILD_TYPE=release -DMIGRAPHX_USE_HIPRTC=On -DGPU_TARGETS='${gpu_targets}'")
        }
    }
}, clang_asan: rocmnode('nogpu') { cmake_build ->
    stage('Clang ASAN') {
        def sanitizers = "undefined,address"
        def debug_flags = "-g -O2 -fno-omit-frame-pointer -fsanitize=${sanitizers} -fno-sanitize-recover=${sanitizers}"
        def gpu_targets = getgputargets()
        cmake_build(flags: "-DCMAKE_BUILD_TYPE=debug -DMIGRAPHX_ENABLE_PYTHON=Off -DMIGRAPHX_ENABLE_GPU=Off -DMIGRAPHX_ENABLE_CPU=On -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}' -DCMAKE_C_FLAGS_DEBUG='${debug_flags}' -DGPU_TARGETS='${gpu_targets}'")
    }
}
//, clang_release_navi: rocmnode('navi32') { cmake_build ->
//    stage('HIP Clang Release Navi32') {
//        def gpu_targets = getnavi3xtargets()
//        cmake_build(flags: "-DCMAKE_BUILD_TYPE=release -DGPU_TARGETS='${gpu_targets}' -DMIGRAPHX_DISABLE_ONNX_TESTS=On")
//    }
//}



def onnxnode(name, body) {
    return { label ->
        rocmtestnode(variant: label, node: rocmnodename(name), docker_args: '-u root', body: body, pre: {
            sh 'rm -rf ./build/*.deb'
            unstash 'migraphx-package'
        })
    }
}

rocmtest onnx: onnxnode('mi100+') { cmake_build ->
    stage("Onnx runtime") {
        sh '''
            apt install half
            #ls -lR
            md5sum ./build/*.deb
            dpkg -i ./build/*.deb
            env
            cd /onnxruntime && ./build_and_test_onnxrt.sh
        '''
    }
}
