
// def rocmtestnode(variant, name, body, args, pre) {
def rocmtestnode(Map conf) {
    def variant = conf.get("variant")
    def name = conf.get("node")
    def body = conf.get("body")
    def docker_args = conf.get("docker_args", "")
    def docker_build_args = conf.get("docker_build_args", "")
    def pre = conf.get("pre", {})
    def ccache = "/var/jenkins/.cache/ccache"
    def image = 'migraphxlib'
    env.CCACHE_COMPRESSLEVEL = 7
    env.CCACHE_DIR = ccache
    def cmake_build = { compiler, flags ->
        def cmd = """
            env
            ulimit -c unlimited
            echo "leak:dnnl::impl::malloc" > suppressions.txt
            export LSAN_OPTIONS="suppressions=\$(pwd)/suppressions.txt"
            rm -rf build
            mkdir build
            cd build
            CXX=${compiler} CXXFLAGS='-Werror' cmake -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache ${flags} ..
            make -j\$(nproc) generate all doc package check VERBOSE=1
        """
        echo cmd
        sh cmd
        if (compiler != "hcc") {
            // Only archive from master or develop
            if (env.BRANCH_NAME == "develop" || env.BRANCH_NAME == "master") {
                archiveArtifacts artifacts: "build/*.deb", allowEmptyArchive: true, fingerprint: true
            }
        }
    }
    node(name) {
        withEnv(['HSA_ENABLE_SDMA=0', 'MIOPEN_DEBUG_GCN_ASM_KERNELS=0']) {
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
                withDockerContainer(image: image, args: "--device=/dev/kfd --device=/dev/dri --group-add video --cap-add SYS_PTRACE -v=/var/jenkins/:/var/jenkins ${docker_args}") {
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
    } else if(name == "nogpu") {
        return rocmtest_name;
    }
    return node_name
}

def rocmnode(name, body) {
    return { label ->
        rocmtestnode(variant: label, node: rocmnodename(name), body: body)
    }
}

rocmtest clang_debug: rocmnode('vega') { cmake_build ->
    stage('Hip Clang Debug') {
        def sanitizers = "undefined"
        def debug_flags = "-g -O2 -fsanitize=${sanitizers} -fno-sanitize-recover=${sanitizers}"
        cmake_build("/opt/rocm/llvm/bin/clang++", "-DCMAKE_BUILD_TYPE=debug -DMIGRAPHX_ENABLE_PYTHON=Off -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}' -DCMAKE_C_FLAGS_DEBUG='${debug_flags}'")
    }
}, clang_release: rocmnode('vega') { cmake_build ->
    stage('Hip Clang Release') {
        cmake_build("/opt/rocm/llvm/bin/clang++", "-DCMAKE_BUILD_TYPE=release")
        stash includes: 'build/*.deb', name: 'migraphx-package'
    }
}, mlir_debug: rocmnode('vega') { cmake_build ->
    stage('MLIR Debug') {
        def sanitizers = "undefined"
        def debug_flags = "-g -O2 -fsanitize=${sanitizers} -fno-sanitize-recover=${sanitizers}"
        cmake_build("/opt/rocm/llvm/bin/clang++", "-DCMAKE_BUILD_TYPE=debug -DMIGRAPHX_ENABLE_PYTHON=Off -DMIGRAPHX_ENABLE_MLIR=On -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}' -DCMAKE_C_FLAGS_DEBUG='${debug_flags}'")
    }
}, clang_asan: rocmnode('nogpu') { cmake_build ->
    stage('Clang ASAN') {
        def sanitizers = "undefined,address"
        def debug_flags = "-g -O2 -fno-omit-frame-pointer -fsanitize=${sanitizers} -fno-sanitize-recover=${sanitizers}"
        cmake_build("/opt/rocm/llvm/bin/clang++", "-DCMAKE_BUILD_TYPE=debug -DMIGRAPHX_ENABLE_PYTHON=Off -DMIGRAPHX_ENABLE_GPU=Off -DMIGRAPHX_ENABLE_CPU=On -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}' -DCMAKE_C_FLAGS_DEBUG='${debug_flags}'")
    }
}//, clang_release_navi: rocmnode('navi21') { cmake_build ->
//    stage('HIP Clang Release Navi') {
//        cmake_build("/opt/rocm/llvm/bin/clang++", "-DCMAKE_BUILD_TYPE=release")
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

rocmtest onnx: onnxnode('rocmtest') { cmake_build ->
    stage("Onnx runtime") {
        sh '''
            apt install half
            ls -lR
            dpkg -i ./build/*.deb
            cd /onnxruntime && ./build_and_test_onnxrt.sh
        '''
    }
}
